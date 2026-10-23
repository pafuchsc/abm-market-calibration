# The original assignment

The project comes from the ST4 integration course "Statistical analysis of financial markets" at CentraleSupélec (2026). The assignment was written by **D. Challet**, who also supervised the project. A group of 5 students did it in one week, and it got 18/20. Below is a short summary of the original assignment, day by day. The course PDF is not included here.

This repository is my **personal rework** of that project. The second section lists what I changed or added.

---

## Summary of the original assignment

**Day 1: calibrating the model.**
- Implement the modified Chiarella model, in which the fundamental value $v_t$ is an exponential moving average of prices.
- Calibrate it with the generalized method of moments (GMM), using as many moments as parameters.
- Repeat the calibration 100 times. Plot the density of each estimated parameter, report its standard deviation and RMSE, and comment on *sloppiness*.

**Day 2: efficient computing.**
- Profile the simulator and speed it up, with numba, multiprocessing or a GPU.
- Replace the simulations with a deep-learning surrogate (or a PINN) to speed up calibration.

**Day 3: investor behaviour (NFT data).**
- Explore the data: CCDFs of activity.
- Classify traders as in Lillo et al., and compute a transition table between periods.
- Predict trades from the inventory.
- Find each trader's reaction horizon T_best, over at least 50 horizons.
- Find out who wins and whether winners persist. Predict returns from the good and bad traders.

**Day 4: prediction.**
- *ABM → ABM*:
  - calibrate the model on T steps of a simulated series;
  - infer the hidden state;
  - predict the next returns;
  - study the hit ratio as a function of the signal-to-noise ratio.
- *ABM → real world*:
  - backtest on real prices with a rolling window;
  - transaction costs are 0.01 % for equities and 0.3 % for crypto.

**Day 5: presentation.**

---

## What I changed or added

**Corrections**
- **NFT.** The group had concluded that "strategies are stable". I compared the transition matrix with the base rates and showed that it is close to random.
- **Prediction.** I showed that the 0.512 hit ratio of the group slides on simulated data was not a failure: it is the ceiling of the model, about 0.52, even for an oracle.

**Additions, part by part**

| Part | What I added |
|---|---|
| 1. Model | A map of the model's regimes for each mechanism (fat tails, volatility clustering, explosions), and a comparison of the stylized facts with the S&P 500 at three time scales. |
| 2. GMM | Moments chosen with a normalized Jacobian and its SVD, which explains why only σ and c − g are identifiable. I added the E-GMM objective (fixed seeds), DIRECT next to Nelder-Mead, Hansen's iterative optimal weighting with the (1 + 1/N) correction, the J-test, and a 50-series Monte Carlo study with a check of its sensitivity to the optimization budget. |
| 3. Surrogate | A noise floor to judge the surrogate's quality, the GMM run with the surrogate on the same 50 series, and a break-even analysis against the numba simulator. |
| 4. Trajectory | Not in the original assignment. I calibrate by exact reconstruction of the state, which turns the problem into a regression (a conditional maximum likelihood). It is much more precise than the GMM. The non-linear version (5 parameters) shows the β–γ degeneracy. |
| 5. NFT | Two collections (main and control), a significance test for T_best based on circular shifts, persistence tested with Spearman's correlation plus a permutation test, and the measured horizons fed back into the model's α and λ. |
| 6. Prediction | Sheppard's formula as the theoretical ceiling, sizing positions by the strength of the signal, a rolling backtest with no look-ahead bias, robust calibration (Huber, winsorization), extreme-day decomposition, volatility targeting, and 9 currency pairs at daily and monthly frequency. On the currencies I added data cleaning, a Bonferroni correction and a data-snooping check. |
