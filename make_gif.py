"""
Regenerate the training animation GIF from a fresh DL run.
Run after hopenhayn_VFI.ipynb has produced output_csv/v_VFI.csv.

Usage:  PYTHONPATH=C:\torch_pkg python make_gif.py
"""
import os, copy, time
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import imageio.v2 as imageio

# === Style (matches notebooks) ===
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": False, "legend.frameon": False,
    "figure.constrained_layout.use": True,
})
COLOR_NN = "#0072B2"; COLOR_REF = "#D55E00"

# === Parameters ===
beta = 1.0/1.04; alpha = 0.64; pstar = 1.0; wstar = 1.0
rho = 0.984150757243253; sigmaF = 0.245520815536363
muF = -2.431373086987380; cf = 2.298950284374937; ns = 100

def tauchen(mu, rho_val, sig, N, m=5):
    su = np.sqrt(sig**2/(1-rho_val**2))
    sv = np.linspace(mu-m*su, mu+m*su, N); step = sv[1]-sv[0]
    Pi = np.zeros((N,N))
    for j in range(N):
        for k in range(N):
            if k==0: Pi[j,k]=norm.cdf((sv[0]-(1-rho_val)*mu-rho_val*sv[j]+step/2)/sig)
            elif k==N-1: Pi[j,k]=1-norm.cdf((sv[-1]-(1-rho_val)*mu-rho_val*sv[j]-step/2)/sig)
            else: Pi[j,k]=(norm.cdf((sv[k]-(1-rho_val)*mu-rho_val*sv[j]+step/2)/sig)
                          -norm.cdf((sv[k]-(1-rho_val)*mu-rho_val*sv[j]-step/2)/sig))
    return sv, Pi

svec_np, Pi_np = tauchen(muF, rho, sigmaF, ns)
sMin, sMax = svec_np[0], svec_np[-1]
sigma_uncond = np.sqrt(sigmaF**2/(1-rho**2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
svec_t = torch.tensor(svec_np, dtype=torch.float32, device=device).unsqueeze(1)
Pi_t = torch.tensor(Pi_np, dtype=torch.float32, device=device)

def profit_torch(s):
    return (pstar*torch.exp(s)*(alpha/wstar)**alpha)**(1/(1-alpha))*(1-alpha)-cf
pi_grid = profit_torch(svec_t)

# Load VFI reference
V_vfi = np.genfromtxt("output_csv/v_VFI.csv", delimiter=",")
exit_idx = np.argmax(V_vfi > 0.01)
exit_s = svec_np[exit_idx]

# Network
class NN(nn.Module):
    def __init__(self, dh=128, ly=4):
        super().__init__()
        torch.manual_seed(123)
        m = [nn.Linear(1,dh), nn.SiLU()]
        for _ in range(ly-1): m += [nn.Linear(dh,dh), nn.SiLU()]
        m.append(nn.Linear(dh,1)); m.append(nn.Softplus())
        self.q = nn.Sequential(*m)
    def forward(self,x): return self.q(x)

torch.manual_seed(123); np.random.seed(123)
v_hat = NN().to(device)
target_net = copy.deepcopy(v_hat); target_net.eval()
for p in target_net.parameters(): p.requires_grad_(False)

optimizer = torch.optim.Adam(v_hat.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200001)
tau = 0.005; exit_center = 0.0

def sample_s(bs):
    n1=bs//4; n2=bs//4; n3=bs//4; n4=bs-n1-n2-n3
    x1=sMin+(sMax-sMin)*torch.rand(n1,1,device=device)
    x2=torch.normal(muF,sigma_uncond,size=(n2,1)).clamp(sMin,sMax).to(device)
    x3=torch.normal(muF-2*sigma_uncond,sigma_uncond,size=(n3,1)).clamp(sMin,sMax).to(device)
    x4=torch.normal(exit_center,0.5,size=(n4,1)).clamp(sMin,sMax).to(device)
    return torch.cat([x1,x2,x3,x4])[torch.randperm(bs,device=device)]

snapshots = []
print("Training for GIF generation...")
for epoch in range(200001):
    s_input = sample_s(512)
    optimizer.zero_grad()
    V = v_hat(svec_t)
    with torch.no_grad(): EV_targ = Pi_t @ target_net(svec_t)
    rhs = torch.max(torch.zeros_like(svec_t), pi_grid + beta*EV_targ)
    loss = (torch.log1p(V) - torch.log1p(rhs)).pow(2).mean()
    loss.backward(); optimizer.step(); scheduler.step()
    with torch.no_grad():
        for p,pt in zip(v_hat.parameters(), target_net.parameters()):
            pt.data.mul_(1-tau).add_(p.data*tau)

    if epoch % 500 == 0:
        with torch.no_grad():
            Vs = v_hat(svec_t).cpu().numpy().flatten()
            sb = s_input.cpu().numpy().flatten()
            Vb = v_hat(s_input).cpu().numpy().flatten()
        snapshots.append((epoch, Vs.copy(), loss.item(), sb.copy(), Vb.copy()))
        # Update exit center
        pos = np.where(Vs > 0.01)[0]
        if len(pos)>0: exit_center = svec_np[pos[0]]

    if epoch % 10000 == 0:
        print(f"  epoch={epoch:>6d}, loss={loss.item():.2e}")

    # Early stop
    if epoch > 5000 and loss.item() < 1e-6:
        with torch.no_grad():
            Vs = v_hat(svec_t).cpu().numpy().flatten()
            sb = s_input.cpu().numpy().flatten()
            Vb = v_hat(s_input).cpu().numpy().flatten()
        snapshots.append((epoch, Vs.copy(), loss.item(), sb.copy(), Vb.copy()))
        print(f"  Early stop at epoch {epoch}, loss={loss.item():.2e}")
        break

# === Build GIF ===
print(f"Building GIF from {len(snapshots)} snapshots...")
log_V_vfi = np.log1p(V_vfi)
y_max_log = log_V_vfi.max() * 1.05
active_mask = V_vfi > 0.01
s_active_min = svec_np[active_mask][0] - 0.5
s_active_max = sMax + 0.2

os.makedirs("output_figures", exist_ok=True)
filenames = []

for i, (epoch, V_snap, loss_snap, s_batch, V_batch) in enumerate(snapshots):
    log_V_snap = np.log1p(np.maximum(V_snap, 0.0))
    log_V_batch = np.log1p(np.maximum(V_batch, 0.0))
    abs_rel_snap = np.where(V_vfi > 0.01, np.abs(V_snap - V_vfi) / V_vfi, 0.0)
    err_pct = abs_rel_snap[active_mask] * 100

    if epoch < 500: nr = 12
    elif epoch < 2000: nr = 8
    elif epoch < 10000: nr = 4
    else: nr = 2

    for t in range(nr):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
        ax = axes[0]
        ax.plot(svec_np, log_V_vfi, color=COLOR_REF, linewidth=1.5, label='VFI')
        ax.plot(svec_np, log_V_snap, color=COLOR_NN, linewidth=1.5, linestyle='--', label='Neural network')
        ax.scatter(s_batch, log_V_batch, s=3, color=COLOR_NN, alpha=0.3, zorder=5,
                   label=f'Mini-batch ({len(s_batch)} pts)')
        ax.plot(svec_np, np.zeros_like(svec_np)-0.15, '|', color=COLOR_REF, markersize=3, alpha=0.5)
        ax.set_ylim(-0.3, y_max_log); ax.set_xlim(sMin-0.2, sMax+0.2)
        ax.axvspan(sMin-0.2, exit_s, color="gray", alpha=0.15)
        ax.set_xlabel(r'Productivity, $s$'); ax.set_ylabel(r'$\log(1+V(s))$')
        ax.set_title(f'(a) Epoch {epoch:,d}  |  Loss = {loss_snap:.2e}')
        ax.legend(loc='upper left', fontsize=7, markerscale=3)

        ax = axes[1]
        if np.any(err_pct > 0):
            ax.plot(svec_np[active_mask], np.clip(err_pct, 1e-5, None), color=COLOR_NN, linewidth=1.0)
        ax.set_yscale("log"); ax.set_xlim(s_active_min, s_active_max); ax.set_ylim(1e-3, 500)
        ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.7, alpha=0.7)
        ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.7, alpha=0.7)
        ax.text(s_active_max-0.1, 12, '10%', ha='right', fontsize=7, color='gray')
        ax.text(s_active_max-0.1, 1.2, '1%', ha='right', fontsize=7, color='gray')
        ax.axvline(x=exit_s, color=COLOR_REF, linestyle='--', linewidth=0.8)
        ax.set_xlabel(r'Productivity, $s$'); ax.set_ylabel(r'Relative error (%)')
        ax.set_title(r'(b) Approximation error')

        fname = f"_gif_frame_{i}_{t}.png"
        fig.savefig(fname, dpi=100); plt.close(fig)
        filenames.append(fname)

# Hold final frame
for _ in range(25): filenames.append(filenames[-1])

with imageio.get_writer("output_figures/hopenhayn_DL.gif", mode="I", duration=0.1) as w:
    for f in filenames: w.append_data(imageio.imread(f))
for f in set(filenames): os.remove(f)

print(f"Saved output_figures/hopenhayn_DL.gif ({len(snapshots)} snapshots, {len(filenames)} frames)")
