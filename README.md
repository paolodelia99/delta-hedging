# Delta Hedging: A Comparative Study Using Machine Learning and Traditional Methods

>The derivatives market is inherently complex, with risk factors such as cross-gamma exposure posing significant challenges to traders. Cross-gamma, the second-order cross-partial derivatives of the derivative price concerning changes in the values of multiple underlying assets, introduces an intricate layer of risk that demands sophisticated hedging strategies. In this study, we aim to delve into the industry problem of determining the effectiveness of hedging mechanisms in mitigating cross-gamma risks

## Current notebooks

- Call Option
  - [Black-Scholes delta hedging with $\Delta^{BS}$](black-scholes-hedging/call-option/bs_delta_hedging.ipynb)
  - [Black-Scholes gamma hedging](black-scholes-hedging/call-option/bs_gamma_hedging_continuos.ipynb)
  - [Black-Scholes static gamma hedging](black-scholes-hedging/call-option/bs_static_gamma_hedging.ipynb)
  - [Black-Scholes occasionaly gamma hedging](black-scholes-hedging/call-option/bs_gamma_hedging.ipynb)
  - [Heston delta hedging](heston-hedging/heston_delta_hedging.ipynb)
  - [Heston Gamma hedging](heston-hedging/heston_gamma_hedging.ipynb)
- Spread Option
  - [Delta Hedging exchange option $X = max(S_1 - S_2, 0)$](black-scholes-hedging/spread-option/delta_hedging_spread.ipynb)
  - [Gamma Hedging exchange option $X = max(S_1 - S_2, 0)$](black-scholes-hedging/spread-option/gamma_hedging_spread.ipynb)
- Reinforcement Learning
  - [Hedging in BS environment with TD3](rl-hedging/hedging_bs_t3d.ipynb)

## TODOs

- [ ] Add text to notebooks - In Progress
- [x] First draft of Deep Hedging
- [ ] Bidimensional BS 
  - [x] Spread option
- [x] Include transaction costs
- [ ] Heston model
  - [ ] Delta Hedging with $\sigma^{IV}$
  - [ ] Gamma Hedging