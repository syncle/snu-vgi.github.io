---
layout: post
title:  "GenAI Lecture: Rectified Flows"
date:   2025-03-10 14:52:12 +0900
categories: Gen AI Lecture
---

### Transport Mapping Problem

지도 학습과 달리 생성 모델 등의 비지도 학습의 많은 문제는 Unpaired dataset을 다룬다. 이에 많은 비지도 학습, 특히 생성 모델은 대개 두 분포 사이에 유의미한 대응 관계를 만들어내는 데 초점을 맞춘다.

예를 들어 Variational Autoencoder(VAE)[1]와 Generative Adversarial Netowrk(GAN)[2]의 경우, 잠재 변수 공간에서 데이터 공간을 대응하는 함수를 찾아내곤 한다. 이러한 모델은 의미론상 유의미한 대응 관계를 찾아내기도 하여, 다양한 downstream task에도 활용된다.

이를 좀 더 일반화하여, 경험적으로 주어진 두 데이터 분포에 대해 대응 관계를 찾아내는 문제를 Transport Mapping Problem이라 한다.

Transport Mapping Problem: Emperically observed distributions $$X_0\sim\pi_0, X_1\sim\pi_1$$ on $$\mathbb R^d$$에 대해 transport map $$T: \mathbb R^d\to\mathbb R^d$$ such that $$Z_1 := T(Z_0)\sim\pi_1; Z_0\sim\pi_0$$을 찾는 문제를 의미한다. 이때 $$T$$에 의해 결정된 대응 관계 $$(Z_0, Z_1)$$를 Transport plan(or Coupling)이라 한다.

대개 이런 $$T$$는 Neural Network로 가정되어, Min-max algorithm에 따라 GAN의 형태로 학습이 이뤄지기도 하고, likelihood maximization scheme에 따라 VAE나 Normalizing Flows의 형태로 학습이 이뤄지기도 한다.

하지만 GAN의 경우 Mode collapse나 numerical instability를 겪기도 하고, VAE에서 Monte-Carlo estimation을 요구 받거나, Normalizing flows처럼 constrained network architecture를 강제 받기도 한다.

근래에 들어서는 Mathmatical Structure를 충분히 활용하여 continuous-time process(이하 flow models)를 통해 transport map을 학습하는 사례들이 등장하고 있고, Ordinary Differential Equations(이하 ODE) 기반의 Neurl ODE[4]나 FFJORD[3], 혹은 Stochastic Differential Equations(이하 SDE) 기반의 ScoreSDE[6] 등이 있다.

이러한 모델의 가장 큰 문제는 transport map을 simulation 하기 위해 여러 번의 네트워크 forward pass를 요구하고, 이에 비례하여 high computational cost를 요구한다는 것이다.

그와 별개로 Transport Mapping Problem 내에서 Cost function $$c: \mathbb R^d \to \mathbb R$$을 두어, transport map 중 가장 작은 transport cost를 가지는(i.e. $$\mathbb E\left[c(Z_1 - Z_0)\right]$$) map을 찾기도 한다. 이러한 문제를 Optimal Transport Problem이라 한다. 하지만, 대개 OT 문제를 해결하는 방법론은 아직 high dimensional data에서 scalable 하지 않고, transport cost가 learning performance와 완전히 정렬되어 있지 않아, 낮은 transport cost가 좋은 performance를 낸다는 보장을 못 받기도 한다.

Rectified Flows는 ODE 기반의 Continuous process를 가정한다. 그러면서도, 위의 문제를 해결하기 위해 transport map이 ***가능한 직선에 가깝게*** 근사하고자 한다. 직선에 가까워진 궤도는 두 점 사이에서 가장 짧은 경로로 수렴하게 되고(OT), Simulation 중에 발생하는 time-discretized error를 줄여 numerical solver가 요구하는 sampling steps의 수를 줄이는 데에도 긍정적 영향을 미친다. 또한 least square optimization을 통해 학습을 수행하여, min-max algorithm 등에서 발생하는 training instability를 효과적으로 방어한다.

### Rectified Flows

먼저 Continuous process를 가정하자.

Emperically observed distributions $$X_0\sim\pi_0, X_1\sim\pi_1$$와 Independent Coupling $$(X_0, X_1)\sim\pi_0\times\pi_1$$의 ODE $$dZ_t = v_\theta(Z_t; t)dt;\ t\in[0, 1]$$를 정의하면, $$Z_1 = Z_0 + \int^1_{0}v_\theta(Z_t; t)dt$$의 Initial Value Problem으로 정리할 수 있다.

Rectified Flows는 많은 ODE Trajectory 중 가장 단순한 형태인 선형 보간, $$X_t = tX_1 + (1 - t)X_0$$을 채택한다. 이때 velocity는 $$\frac{dX_t}{dt} = X_1 - X_0$$로 정리되고, $$X_1$$과 $$X_0$$을 잇는 가장 짧은 궤적이 된다.

학습은 MSE의 형태로 단순화하여 수행한다.

$$\theta^* = \arg\min_\theta\int^1_0\mathbb E_{(X_0, X_1)\sim\pi_0\times\pi_1}\left[||(X_1 - X_0) - v_\theta(X_t; t)||^2_2\right]dt$$

$$X_t$$와 $$dX_t/dt$$는 $$X_1$$과 $$X_0$$를 모두 포함하고 있기에, 위 학습은 Network가 $$X_1$$의 정보 없이도 velocity를 추정하는 *causalize*의 과정을 겪게 된다.

### Properties

Rectified flow는 Marginal distribution $$\pi_0, \pi_1$$을 유지한다. 또한, 잘 학습된 Rectified Flows는 다음 Theorem에 의해 궤적 사이에 교점이 존재하지 않음을 보장받는다.

Theorem3.6. $$X_t = tX_1 + (1 - t)X_0$$일 때, Uniformly lipschitz continuous $$v$$에 대해 $$(X_0, X_1)$$이 Straight coupling이면 다음을 만족하고, 이는 $$X_t$$가 $$X_1$$과 $$X_0$$에 의해 유일하게 결정됨(궤적 간의 교점이 없음)을 의미한다.

(Straight Coupling: ODE에 의해 구체화된 대응 $$(X_0, X_1)$$ s.t. $$X_1 = X_0 + \int^1_0 v(X_t; t)dt$$)

$$V((X_0, X_1)) := \int^1_0\mathbb E\left[||X_1 - X_0 - \mathbb E[X_1 - X_0|X_t]||^2_2\right]dt = 0$$

이에 Rectified Flows는 Marginal distributions를 유지하면서, 궤적 사이에 교점이 없도록 표본 간의 대응을 Rewiring 한다. 그리고 Rewired Coupling $$\textbf Z^1 = (Z_0, Z_1)\ s.t. Z_0\sim\pi_0, Z_1 = Z_0 + \int^1_0v_\theta(x_t; t)dt$$을 ***Rectify***(to Straight Coupling)라 정의한다.

이렇게 Rectified Coupling을 두고 다시 학습 학습하는 과정을 Reflow라 하고, 다음과 같이 표현한다.

$$\theta^{k+1} = \arg\min_\theta\int^1_0\mathbb E_{(Z^{k+1}_0, Z^{k+1}_1)\sim\mathrm{Rectify}((Z^k_0, Z^k_1))}[||Z^{k+1}_1 - Z^{k+1}_0 - v_\theta(X_t; t)||^2_2]dt$$

$$\implies \textbf Z^{k+1} = \mathrm{RectFlow}((Z^k_0, Z^k_1)) = \mathrm{Rectify}(\textbf Z^k; \theta^{k+1})$$

이렇게 k-번 continuously-trained rectified flow를 k-rectified flows라 하며, 회가 거듭될수록 낮은 convex transport cost를 보인다.

$$\mathbb E[c(Z^{k+1}_1 - Z^{k+1}_0)] \le \mathbb E[c(Z^k_1 - Z^k_0)],\ (Z^0_0, Z^0_1) = (X_0, X_1); k\ge0$$

이는 k가 증가함에 따라 Trajectory는 더 Straight 해지고, Time-discretize Error와 Numerical Error 역시 감소하여 Sampling step을 줄임에도 성능 하락 폭이 다소 작아지는 경향을 만들어낸다.

별개로 직렬화 정도를 다음의 Straightness measure $$S(\textbf Z)$$를 가정하면, 직렬화 정도를 Reflow의 수행 횟수 내에서 bounding 할 수 있다.

$$S(\textbf Z) = \int^1_0\mathbb E\left[||(Z_1 - Z_0) - \hat Z_t||^2_2\right]\implies \min_{k\in\{0,...,K\}}S(\textbf Z^k)\le \frac{\mathbb E[||X_1-X_0||^2_2]}{K}$$

이렇게 학습된 Rectified Flows는 ODE Solver를 통해 Initial value problem을 풀게 된다. 대표적으로 1st-order Euler Solver을 가정한다면 다음의 update rule을 가지게 되고, N-sampling steps라면 $$\Delta t = 1/N$$이다.

$$x_{t + \Delta t} = x_t + v_\theta(x_t; t)\Delta t$$

### Distillation

k-Rectified Flows는 continuous time interval $$t\in[0, 1]$$에서 velocity를 추정하도록 학습하기 때문에, 아직 1-step generation의 성능을 개선할 여지가 남아 있다.

$$T(z_0) = z_0 + v_\phi(z_0; 0)$$의 1-step generator를 가정하고, time interval을 single-point $$t=0$$으로 축약하여 distillation을 수행한다.

$$\mathbb E\left[(Z^k_1 - Z^k_0) - v_\phi(Z^k_0; 0)\right]$$

Reflow(or Rectification)은 학습을 통해 새로운 Rectified Coupling $$\textbf Z^{k+1} = (Z^{k+1}_0, Z^{k + 1}_1)$$를 만드는 것이 목표라면, Distillation은 $$\textbf Z^k$$를 lower-step에서 approximate 하는 것에 초점을 맞추는 차이점을 가진다.

### Relations between Score Models

Rectified flows는 interpolation $$X_t = tX_1 + (1 - t)X_0$$를 time-varying coefficient $$\alpha_t, \beta_t$$의 weighted sum $$X_t = \alpha_tX_1 + \beta_tX_0$$으로 확장할 경우 ScoreSDE[6]의 Probability flow ODE(이하 PF-ODE)와 unified framework로 일반화될 수 있다.

$$\mathrm{RectifiedFlow}:\ \alpha_t = 1;\ \beta_t = 1- t$$

$$\mathrm{VP\ ODE}:\ \alpha_t = \exp\left(-\frac{1}{4}a(1 - t)^2 - \frac{1}{2}b(1 - t)\right);\ \beta_t = \sqrt{1 - \alpha^2_t}$$

$$\mathrm{VE\ ODE}:\ \alpha_t = 1;\ \beta_t = \sigma_\mathrm{min}\sqrt{r^{2(1- t)} - 1}$$

sub-VP ODE의 경우 VP ODE와 $$\alpha_t$$를 공유하고, $$\beta_t = 1 - \alpha_t$$로 나타낸다. VP ODE와 sub-VP ODE의 경우 $$\beta_0\approx 1$$이 되도록 hyperparameter를 설정하고, $$\pi_0=\mathcal N(0, I)$$로 가정한다.

두 경우 모두 velocity가 uniform 하지 않고, Trajectory가 Recitifed flow와 달리 곡선의 형태를 띤다. 그렇기에 Reflow를 수행하여도 convex cost의 감소나 straightness의 개선을 기대할 수 없다. 또한, sampling interval에 관한 engineering을 요구하고, step-size가 커질수록 approximation error가 증가하게 된다.

VE-ODE의 경우 Trajectory는 직선이나, velocity는 $$\beta_t$$에 의해 uniform 하지 않아 여전히 sampling interval에 관한 engineering을 요구할 수 있다. 

### References

[1] Auto-Encoding Variational Bayes, Kingma & Welling, 2013. [[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)] \
[2] Generative Adversarial networks, Goodfellow et al., 2014. [[arXiv:1406.2661](https://arxiv.org/abs/1406.2661)] \
[3] FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models, Grathwohl et al., 2018. [[arXiv:1810.01367](https://arxiv.org/abs/1810.01367)] \
[4] Neural Ordinary Differential Equations, Chen et al., 2018. [[arXiv:1806.07366](https://arxiv.org/abs/1806.07366)] \
[5] Denoising Diffusion Probabilistic Models, Ho et al., 2020. [[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)] \
[6] Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020. [[arXiv:2011.13456](https://arxiv.org/abs/2011.13456)]
