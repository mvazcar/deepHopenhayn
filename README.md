# Hopenhayn Firm Dynamics: VFI vs Deep Learning

> Inspired by the deep learning approach of [Mahdi Kahou](https://github.com/Mekahou) — check out his [McCall Search Model notebook](https://github.com/Mekahou/Notes/blob/main/deep_learning/McCall_DL.ipynb) for the pedagogical foundation this builds on.

Solving the stationary Hopenhayn (1992) firm dynamics model using both **Value Function Iteration** and a **neural network** that minimizes the Bellman residual.

Based on the calibration in Hopenhayn, Neira & Singhania (2022).

![Training Animation](output_figures/hopenhayn_DL.gif)

---

## Model

### Value function and exit

A firm with log-productivity $s$ chooses to stay or exit:

$$V(s) = \max\left\{0,\; \pi(s) + \beta\, \mathbb{E}\!\left[V(s') \mid s\right]\right\}$$

where static profit (after optimizing labor $n$) is:

$$\pi(s) = \max_{n}\; p\, e^{s}\, n^{\alpha} - w\, n - w\, c_f = \left(p\, e^{s}\, \alpha^{\alpha}\right)^{\frac{1}{1-\alpha}}(1-\alpha) - c_f$$

Firms exit when $V(s) = 0$. The **exit threshold** $s^*$ satisfies $V(s^*) = 0$.

### Productivity process

Log-productivity follows a Tauchen-discretized AR(1):

$$s' = (1-\rho)\,\mu_F + \rho\, s + \sigma_F\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,1)$$

### Free entry

The expected value of an entrant (drawn from $G$) equals the entry cost:

$$\mathbb{E}_G\!\left[V(s)\right] = c_e$$

---

## Methods

### Value Function Iteration (VFI)

Standard contraction mapping on a 100-point Tauchen grid:

$$T\,V(s_j) = \max\left\{0,\; \pi(s_j) + \beta \sum_{k} P(s_k \mid s_j)\, V(s_k)\right\}$$

Iterate $V \leftarrow T\,V$ until $\|T\,V - V\|_\infty < 10^{-8}$.

- **294 iterations**, **0.002 seconds**

### Deep Learning (neural network)

A feedforward network $\hat{V}_\theta(s)$ with 4 hidden layers × 128 neurons (SiLU activation, Softplus output to enforce $\hat{V} \geq 0$) minimizes the **log-space Bellman residual**:

$$\mathcal{L}(\theta) = \frac{1}{B}\sum_{i=1}^{B}\left[\log\!\left(1 + \hat{V}_\theta(s_i)\right) - \log\!\left(1 + \text{RHS}_i\right)\right]^2$$

where the Bellman target uses a **Polyak-averaged target network** $\bar{\theta}$ for stability:

$$\text{RHS}_i = \max\left\{0,\; \pi(s_i) + \beta \sum_k P(s_k \mid s_i)\, \hat{V}_{\bar{\theta}}(s_k)\right\}$$

The conditional expectation $\mathbb{E}[V(s')|s]$ uses the **same Tauchen transition matrix** as VFI — an apples-to-apples comparison.

**Key design choices** (and why they matter):

| Feature | Why |
|---|---|
| **Log-space loss** | $V(s)$ spans 5 orders of magnitude (0 to 186,000); raw MSE is dominated by high-$V$ points |
| **Softplus output** | Guarantees $\hat{V} \geq 0$ without gradient-killing clamps |
| **Target network** (Polyak) | The RHS depends on $\hat{V}$ itself — without a frozen target, it's a moving-target problem that doesn't converge |
| **Adaptive mini-batch sampling** | Concentrates collocation points near $s^*$ where the kink is hardest to learn |
| **Early stopping** | Loss plateaus after ~36k epochs; no need for 200k |

- **~36,000 epochs**, **~90 seconds** (NVIDIA RTX 5080)

---

## Results

| | VFI | Neural Network |
|---|---|---|
| Exit threshold $s^*$ | 0.0160 | 0.0160 |
| Wall time | 0.002 s | 90 s |
| Mean relative error (interior) | — | 4.1% |

![Value Function Comparison](output_figures/value_function_comparison_DL.png)

---

## Project structure

```
deepHopenhayn/
├── hopenhayn_VFI.ipynb              # VFI solution (run first)
├── hopenhayn_DL.ipynb               # Deep learning solution (loads VFI CSVs)
├── make_gif.py                      # Regenerate training animation
├── output_csv/
│   ├── v_VFI.csv, svec_VFI.csv      # VFI value function and grid
│   ├── nstar_VFI.csv, mustar_VFI.csv # Labor demand and firm distribution
│   └── v_nn_DL.csv, svec_nn_DL.csv  # NN value function and grid
└── output_figures/
    ├── value_function_VFI.png        # VFI plots
    ├── value_function_comparison_DL.png  # VFI vs NN
    ├── sampling_distribution_DL.png  # Mini-batch visualization
    └── hopenhayn_DL.gif             # Training animation
```

**Run order:** `hopenhayn_VFI.ipynb` → `hopenhayn_DL.ipynb`

---

## Calibration

| Parameter | Value | Description |
|---|---|---|
| $\beta$ | $1/1.04$ | Discount factor |
| $\alpha$ | $0.64$ | Labor share |
| $\rho$ | $0.984$ | AR(1) persistence |
| $\sigma_F$ | $0.246$ | Std dev of innovations |
| $\mu_F$ | $-2.431$ | Long-run mean |
| $c_f$ | $2.299$ | Fixed operating cost |

## References

- Hopenhayn, H. (1992). Entry, Exit, and Firm Dynamics in Long Run Equilibrium. *Econometrica*, 60(5), 1127–1150.
- Hopenhayn, H., Neira, J., & Singhania, R. (2022). From Population Growth to Firm Demographics: Implications for Concentration, Entrepreneurship and the Labor Share. *Econometrica*, 90(4), 1879–1914.
