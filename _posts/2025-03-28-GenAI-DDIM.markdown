---
layout: post
title:  "GenAI Lecture: DDIM"
date:   2025-03-28 16:17:00 +0900
categories: Gen AI Lecture
---

DDPM은 Diffusion Model의 시초를 열었다. 기존에 state-of-the-art로 사용되던 GAN의 adversarial training 방법과는 다른 새로운 방법으로 그와 비슷한 퀄리티의 이미지를 생성했고, 이에 많은 주목을 받았다.

하지만 GAN의 샘플링이 한번의 forward pass를 통해 이루어지는 반면, DDPM의 샘플링은 Markov chain을 따라 진행되며, 그 과정에서 많은 iteration을 필요로 한다. 이 때문에 DDPM의 샘플링 속도는 GAN에 비해 현저히 느리며, 이미지 크기가 커질수록 DDPM의 샘플링 시간은 더 큰 문제가 된다.

DDIM은 이 같은 DDPM과 GAN 사이의 효율 차이를 줄이기 위해 고안되었다. DDIM은 implicit probabilistic model로, DDPM의 아이디어를 기반으로 하지만, DDPM의 Markovian forward process를 non-Markovian으로 일반화하며 동시에 적합한 reverse generative Markov chain을 설정하였다.

#### Background - DDPM
현실의 데이터 분포 $$q(x_0)$$가 주어졌을 때, DDPM의 목표는 $$q(x_0)$$를 잘 근사하면서도 샘플링이 쉽게 가능한 모델 분포 $$p(x_0)$$를 학습시켜, 학습된 모델 분포에서 샘플링하는 것이다.

이를 위해 DDPM에서는 먼저 이미지를 latent variable로 바꾸는 inference procedure(=forward process)을 다음과 같은 Markov chain으로 고정한다.

$$
q (x_{1:T} | x_0) := \prod_{t=1}^T q (x_t | x_{t-1}), \quad q (x_t | x_{t-1}) := \mathcal{N} \bigg(\sqrt{\frac{\alpha_t}{\alpha_{t-1}}} x_{t-1}, \bigg( 1 - \frac{\alpha_t}{\alpha_{t-1}} \bigg) I \bigg)
$$
이때 $$x_0$$는 원본 이미지이고, $$x_1, ..., x_T$$는 latent variable이다.
$$x_{t-1}$$를 $$x_t$$로 바꾸는 스텝마다 Gaussian noise가 섞이며 원본 이미지의 정보가 희석된다.

이때 DDPM이 학습하는 것은 forward process의 최종 결과물인 $$x_T$$를 원본 이미지인 $$x_0$$로 되돌리는 $$p_\theta (x_{0:T})$$이며, 이를 generative process라고 부른다. generative process는 다음과 같이 Markov chain으로 표현된다.
$$
p_\theta (x_0) = \int p_\theta (x_{0:T}) dx_{1:T}, \quad p_\theta (x_{0:T}) := p_\theta (x_T) \prod_{t=1}^T p_\theta^{(t)} (x_{t-1} | x_t)
$$
다시 말하자면, generative process 학습의 목적은 intractable한 reverse process $$q (x_{t-1} | x_t)$$을 근사하는 것이다.

이때, Gaussian noise의 특성에 의해 다음과 같은 식이 성립한다.

$$
q (x_t | x_0) := \int q(x_{1:t} | x_0) dx_{1:(t-1)} = \mathcal{N} (x_t ; \sqrt{\alpha_t} x_0, (1-\alpha_t)I)
$$
따라서 $$x_t$$를 $$x_0$$와 noise 변수 $$\epsilon$$의 선형결합으로 나타낼 수 있다.
$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N} (\textbf{0}, I)
$$

이때 $$\alpha_T$$가 충분히 작다면 $$q(x_T|x_0)$$가 표준 정규 분포로 수렴하며, $$p_\theta (x_T) := \mathcal{N} (\textbf{0},I)$$에서 샘플링을 시작하는 것이 자연스러워진다. 즉, 모델을 잘 학습시킨다면 랜덤 Gaussian noise에서 이미지를 생성할 수 있게 된다.
<br>

모델의 학습은 negative log likelihood에 대한 variational bound를 최적화하는 방향으로 진행된다.

$$
\mathbb{E}_{q(x_0)} [-\log p_\theta (x_0)] \le \mathbb{E}_{q(x_0, x_1, \cdots, x_T)} [\log q(x_{1:T} | x_0) - \log p_\theta (x_{0:T}) ]
$$

이때 조건을 적절히 맞추어주면 목적함수를 다음과 같이 간단하게 표현할 수 있다.
$$
L_\gamma (\epsilon_\theta) := \sum_{t=1}^T \gamma_t \mathbb{E}_{x_0 \sim q(x_0), \epsilon_t \sim \mathcal{N} (\textbf{0}, I)}
\bigg[ \| \epsilon_\theta^{(t)} (\sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon_t) - \epsilon_t \|_2^2 \bigg], \quad
\epsilon_\theta := \{\epsilon_\theta^{(t)}\}_{t=1}^T
$$
이렇게 단순화된 목적함수를 사용하는 DDPM 모델은 실제로는 각 상태에서 원본 이미지 상태로 되돌리기 위해 없애야 할 노이즈를 예측한다.

### DDIM

DDIM의 아이디어는 inference process에 대한 고찰로부터 시작한다. DDPM의 단순화된 목적함수 $$L_\gamma$$의 식을 살펴보면, $$L_\gamma$$이 결합분포 $$q(x_{1:T}|x_0)$$의 값과는 직접적인 연관이 없으며, 주변분포 $$q(x_t|x_0)$$에만 의존하는 것을 알 수 있다. 같은 주변분포를 가지는 결합분포는 수없이 많이 존재한다. 따라서 DDPM의 Markovian process와 같은 주변분포를 가지는 non-Markovian inference process를 생각해볼 수 있고, 이때의 목적함수는 DDPM의 것과 같다.
<br>

#### Forward Process

우선, DDIM에서는 다음과 같은 non-Markovian forward process를 정의했다.
$$
q_\sigma (x_{1:T} | x_0) := q_\sigma (x_T | x_0) \prod_{t=2}^T q_\sigma (x_{t-1} | x_t, x_0) \\
$$
이때 주변분포 $$q_\sigma (x_t \vert x_0) = \mathcal{N} (\sqrt{\alpha_t} x_0, (1-\alpha_t)I)$$를 보장하기 위해 다음과 같은 정의가 따라온다.
$$
q_\sigma (x_T | x_0) = \mathcal{N} (\sqrt{\alpha_t} x_0, (1-\alpha_t)I) \; \textrm{and for all} \; t > 1, \\
q_\sigma (x_{t-1} | x_t, x_0) = \mathcal{N} \bigg( \sqrt{\alpha_{t-1}} x_0  + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1-\alpha_t}}, \sigma_t^2 I \bigg) 
$$ 

#### Generative Process

DDIM에서 정의하는 학습 가능한 generative process $$p_\theta (x_{0:T})$$에서 각 $$p_\theta^{(t)} (x_{t-1} \vert x_t)$$는 $$q_\sigma (x_{t-1} \vert x_t, x_0)$$의 정보를 활용한다.
$$x_t$$가 주어졌을 때 모델은 $$x_0$$를 예측하고, 주어진 $$x_t$$와 예측한 $$x_0$$를 $$q_\sigma (x_{t-1} \vert x_t, x_0)$$에 대입하여 $$x_{t-1}$$을 샘플링한다.
실제로 모델이 예측하는 것은 $$x_0$$와 $$x_t$$ 사이의 noise $$\epsilon_t \sim \mathcal{N} (\textbf{0},I)$$이다. 모델 $$\epsilon_\theta^{(t)} (x_t)$$가 $$x_0$$에 대한 정보 없이  $$\epsilon_t$$를 예측하면, 이를 이용해 $$x_t$$에 대한 $$x_0$$의 예측 $$f_\theta^{(t)} (x_t)$$(=denoised observation)을 구할 수 있다.
$$
f_\theta^{(t)} (x_t) := \frac{1}{\sqrt{\alpha_t}} (x_t - \sqrt{1-\alpha_t} \epsilon_\theta^{(t)})
$$

이때 고정된 prior $$p_\theta (x_T) = \mathcal{N} (\textbf{0}, I)$$에서 시작하는 generative process가 다음과 같이 정의된다.
$$
p_\theta^{(t)} (x_{t-1} | x_t) =
\begin{cases}
\mathcal{N}(f_\theta^{(t)} (x_1), \sigma_1^2 I) & \textrm{if} \;t = 1 \\
q_\sigma (x_{t-1} | x_t, f_\theta^{(t)} (x_t)) & \textrm{otherwise,}
\end{cases}
$$
$$q_\sigma (x_{t-1} \vert x_t, f_\theta^{(t)} (x_t))$$은 위에서 정의한 $$q_\sigma (x_{t-1} \vert x_t, x_0)$$ 식에 주어지지 않은 $$x_0$$ 대신 예측한 $$f_\theta^{(t)} (x_t)$$를 넣어 얻을 수 있다. 또한 모든 시점에 generative process를 보장하기 위해 $$t=1$$인 경우에 Gaussian noise를 추가했다.

#### Training Model

모델 파라미터 $$\theta$$는 다음과 같이 일반적인 variational inference objective에 따라 최적화된다.
$$
J_\sigma (\epsilon_\theta) = \mathbb{E}_{x_{0:T} \sim q_\sigma (x_{0:T})} [\log q_\sigma (x_{1:T} | x_0) - \log p_\theta (x_{0:T})] \\
= \mathbb{E}_{x_{0:T} \sim q_\sigma (x_{0:T})} \bigg[ \log q_\sigma (x_T | x_0) + \sum_{t=2}^T \log q_\sigma (x_{t-1} | x_t, x_0) - \sum_{t=1}^T \log p_\theta^{(t)} (x_{t-1} | x_t) - \log p_\theta (x_T) \bigg]
$$
언뜻 보기엔 $$\sigma$$가 바뀌면 $$J_{\sigma}$$가 변하기 때문에 $$\sigma$$를 선택할 때마다 새로운 모델을 훈련해야 할 것처럼 보인다.
하지만 다음과 같은 사실을 증명할 수 있다.

$$
For\; all\; \sigma > 0,\; there\; exists\; \gamma \in \mathbb{R}_{> 0}^T\; and\; C \in \mathbb{R},\; such\; that\; J_\sigma = L_\gamma + C
$$

$$L_\gamma$$을 잘 살펴보면, $$\epsilon_\theta^{(t)}$$의 parameter $$\theta$$가 서로 다른 $$t$$에서 공유되지 않을 때는 모델의 optimal solution이 가중치 $$\gamma$$에 의존하지 않는다는 것을 알 수 있다. 이는 $$L_\gamma$$을 이루는 각 항을 최적화시키는 과정이 서로에게 영향을 주지 않기 때문이다. 이럴 경우, $$L_\gamma$$를 목적함수로 사용하던 모델은 $$L_1$$를 대신 사용해 훈련해도 같은 결과를 얻을 수 있다.
또한 앞서 살펴본 대로 모든 $$J_{\sigma}$$는 어떤 $$L_{\gamma}$$와 같기 때문에, DDIM 모델이 위 가정을 만족한다면 $$J_{\sigma}$$ 대신 $$L_1$$을 목적함수로 사용할 수 있고, $$L_1$$으로 훈련시킨 DDPM 모델을 최적해로 대신 사용할 수 있다.

#### Sampling Procedure

DDIM에서는 $$\sigma$$로 매개화한 수많은 non-Markovian process를 정의했으며, 따라서 $$\sigma$$를 변화시키며 개중에서 더 나은 generative process를 찾아볼 수 있다.

다음 식으로 $$x_t$$에서 $$x_{t-1}$$을 샘플링할 수 있다.

$$
x_{t-1} = \sqrt{\alpha_{t-1}} \underbrace{\bigg( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta^{(t)} (x_t)}{\sqrt{\alpha_t}} \bigg)}_{\textrm{predicted } x_0} +
\underbrace{\sqrt{1-\alpha_{t-1} - \sigma_t^2} \cdot \epsilon_\theta^{(t)}(x_t)}_{\textrm{direction pointing to }{x_t}} +
\underbrace{\sigma_t \epsilon_t}_{\textrm{random noise}}
$$

이때 $$\epsilon_t \sim \mathcal{N} (0, I)$$는 $$x_t$$에 독립적인 노이즈이며, $$\alpha_0:=1$$로 정의한다.
$$\sigma$$의 변화는 generative process의 변화로 이어지지만, 여전히 같은 모델 $$\epsilon_\theta$$를 사용할 수 있기 때문에 모델을 재학습할 필요는 없다.
주목할 만한 점은 $$\sigma_t$$를 작게 설정할수록 generative process에서 확률적인 부분이 줄어들고, 점점 deterministic 해진다는 것이다. 특히 $$\sigma_t=0$$인 경우에서는 generative process에서 random noise의 영향이 아예 사라지며, $$x_T$$에서 $$x_0$$로 샘플링하는 과정이 고정된다. 즉, 모델이 implicit probabilistic model이 된다. 이를  Denoising Diffusion Implicit Model(DDIM)이라고 부른다.

#### Accelerated Sampling

generative process는 reverse process를 근사하기 때문에, forward process가 $$T$$ step에 거쳐 진행되면 샘플링도 $$T$$ step을 거쳐야 한다. 하지만 $$L_1$$은 주변분포가 고정되어 있는 한 특정 forward process에 종속적이지 않으며, 따라서 생성과정을 가속하기 위해 $$T$$보다 짧은 길이의 forward process를 고려해 볼 수 있다.
$$x_{1:T}$$ 대신 그 부분집합 $$\{ x_{\tau_1}, \cdots, x_{\tau_S} \}$$를 forward process로 정의하자. $$\tau$$는 $$[1, \cdots, T]$$의 부분수열이다. 이때 샘플링은 $$reverse(\tau)$$를 따라 진행할 수 있으며, 이를 샘플링 궤적이라고 부른다. 이때 샘플링 궤적이 $$T$$보다 많이 짧으면 샘플링 효율을 크게 증가할 수 있다.
이때 $$q(x_{\tau_i}) = \mathcal{N} (\sqrt{\alpha_{\tau_i}} x_0 , (1-\alpha_{\tau_i})I)$$만 만족시켜주면, 앞선 내용과 마찬가지로, $$L_1$$ 목적함수로 학습한 모델을 바뀐 샘플링 과정에 그대로 사용할 수 있다. 기존에 $$x_t$$에서 $$x_{t-1}$$을 샘플링했다면, 이제는 $$x_{\tau_i}$$에서 $$x_{\tau_{i-1}}$$을 샘플링하게끔 공식을 살짝 바꿔주기만 하면 된다.

샘플링 스텝 수(혹은 연산량)와 샘플 퀄리티는 서로 trade-off 관계에 있다. 샘플링 스텝이 많을수록 샘플 퀄리티가 좋아진다. 하지만 DDIM에서는 샘플링 과정이 deterministic하기 때문에 적은 스텝으로 샘플링하면서도 샘플 퀄리티를 DDPM에 비해 비교적 잘 유지할 수 있다. 실제로 1000 step으로 훈련된 DDPM 모델에서는 샘플링 궤적이 100 step 이하로 내려가면 샘플 퀄리티가 심각하게 떨어지지만, $$\sigma=0$$인 DDIM 모델의 경우에는 20 step만으로도 1000 step의 경우와 비교 가능한 퀄리티의 이미지를 생성할 수 있다. DDPM과 DDIM의 iterative한 샘플링 과정을 생각해보면, 50배만큼 짧은 시간에 양질의 이미지를 얻을 수 있는 것이다.

#### Connection with ODE
DDIM의 deterministic한 샘플링 과정은 효과적인 quality-time trade-off 외에도 여러 특성을 지니며, 이는 상미분 방정식(ODE)과의 연관성에서 비롯된다.
$$x_t$$에서 $$x_{t-1}$$을 샘플링하는 과정을 재배열하면
$$
\frac{x_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} = \frac{x_t}{\sqrt{\alpha_t}} + \bigg( \sqrt{\frac{1 - \alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}} - \sqrt{\frac{1-\alpha_t}{\alpha_t}} \bigg) \epsilon_\theta^{(t)} (x_t)
$$
과 같이 표현할 수 있고, 이때 $$\sigma = \sqrt{(1-\alpha) / \alpha}, \bar{x} = x / \sqrt{\alpha}$$로 치환하면 위 식은 다음과 같은 ODE에 Euler method를 적용한 형태와 같다는 것을 알 수 있다.
$$
d \bar{x} (t) = \epsilon_\theta^{(t)} \bigg( \frac{\bar{x} (t)}{\sqrt{\sigma^2 + 1}} \bigg) d \sigma (t)
$$
이때 Euler method의 초기 조건은 $$x(T) \sim \mathcal{N} (0, \sigma(T))$$이다.
따라서 충분한 discretization steps을 거치면, ODE를 뒤집어서 $$x_0$$를 $$x_T$$로 encode할 수도 있다.
이를 통해 DDIM에서는 기존 DDPM에서는 할 수 없었던 semantic interpolation, image reconstruct 등 latent encoding을 통해 할 수 있는 다양한 작업을 시도해볼 수 있다.
DDIM의 generative process와 ODE와의 연관성은 후에 score-based generative model를 통해 더욱 자세히 설명된다.