---
layout: post
title: "GenAI Lecture: Score-based Generative Modeling with SDE (Part 2)"
date: "2025-01-15 15:30:00 +0900"
categories: Gen AI Lecture
---

# Score SDE: Score-based Generative Modeling with SDE (Part 2)

*Presented by 서민균*  
*Based on work by Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole*

---

### Recap from Part 1

앞서 Part 1에서는 확률 미분 방정식(SDE)을 기반으로 한 생성 모델링의 개요와 주요 구성 요소들을 다뤘다.

- **SDE의 정의**  
  확률적 생성 과정을 다음과 같은 형태로 표현할 수 있다:

  $$
  dx = f(x, t) \, dt + g(t) \, dw
  $$

  여기서 $$f(x, t)$$는 drift, $$g(t)$$는 diffusion 계수, $$w$$는 Wiener process이다.

- **Forward vs Reverse SDE**  
  Forward SDE는 데이터를 점점 노이즈화시키는 과정이고,  
  Reverse-time SDE는 이 과정을 거꾸로 되돌려 샘플을 생성하는 방향이다.

- **VP SDE / VE SDE**  
  - **VP SDE (Variance Preserving)**: DDPM 기반
  - **VE SDE (Variance Exploding)**: SMLD 기반  
  - 이외에도 저자들이 제안한 중간 형태인 sub-VP SDE가 있다.

이제 이러한 SDE를 실제로 어떻게 풀 것인가, 즉 샘플링하는 다양한 방법들을 살펴본다.
### Diverse Samplers for Solving SDE

SDE를 샘플링하는 방식은 여러 가지가 있다. 대표적으로는 일반적인 수치 해석 기반의 방법과, 이를 생성 모델에 맞춰 응용한 방식들이 있다.

---

#### General-purpose Sampler

가장 기본적으로 사용되는 방법은 **Euler-Maruyama**다.  
다음과 같은 형태의 확률 미분 방정식이 주어졌다고 하자:

$$
dx(t) = f(x(t), t) \, dt + g(x(t), t) \, dw(t)
$$

정확한 해는 다음과 같은 적분으로 표현된다:

$$
x(t + \Delta t) = x(t) + \int_t^{t+\Delta t} f(x(t), t)\,dt + \int_t^{t+\Delta t} g(x(t), t)\,dw(t)
$$

Euler-Maruyama는 이 적분을 간단히 근사해서 다음과 같이 계산한다:

- $$\Delta w \sim \mathcal{N}(0, \Delta t)$$일 때,

$$
\hat{x}(t_{k+1}) = \hat{x}(t_k) + f(\hat{x}(t_k), t_k) \, \Delta t + g(\hat{x}(t_k), t_k) \, \Delta w
$$

이 방법은 구현이 간단하고 빠르다는 장점이 있지만, 정밀도가 낮다.

좀 더 정밀한 해를 원한다면 **Stochastic Runge-Kutta** 계열의 방법을 사용할 수 있다.  
이 방식은 deterministic Runge-Kutta 방법에 Wiener process 항을 추가한 형태로,  
복잡하지만 더 정확한 근사를 제공한다.  
계산량이 많고 구조가 복잡해 실제로는 잘 쓰이지 않지만, 논문에서는 참고용으로 소개된다.

---

#### Reverse-time SDE 기반 샘플링

학습된 score 모델을 이용해 **역방향 SDE(reverse-time SDE)**를 구성할 수 있다.  
이 구조를 따라가면 노이즈로부터 데이터를 생성할 수 있다.

DDPM에서 사용하는 **Ancestral sampling**은 VP SDE에 기반한 샘플러다.  
하지만 Sub-VP나 VE SDE와 같은 다른 SDE에 대해서는 동일한 방식으로 풀 수 없고,  
score function을 사용해 reverse-time SDE를 다시 구성해야 한다.

예를 들어, DDPM의 경우 역방향 SDE는 다음과 같이 쓸 수 있다:

$$
dx = -\beta(t) \left[ \frac{x}{2} + \nabla_x \log p_t(x) \right] dt + \sqrt{\beta(t)} \, d\bar{w}
$$

여기서 $$\nabla_x \log p_t(x)$$는 확률 분포의 gradient이며, 실제로는 학습된 score function $$s_\theta^*(x, t)$$로 근사한다.

---

#### Reverse Diffusion Sampler

실제 구현에서는 forward SDE를 다음과 같은 형태로 표현하고:

$$
dx = f(x, t)\,dt + G(t)\,dw, \quad x_{i+1} = x_i + f_i(x_i)\Delta t + G_i z_i, \quad z_i \sim \mathcal{N}(0, I)
$$

이를 역방향으로 discretize해서 다음과 같이 샘플을 생성한다:

$$
x_i = x_{i+1} - f_{i+1}(x_{i+1}) \Delta t + G_{i+1} G_{i+1}^\top s_\theta^*(x_{i+1}, t_{i+1}) \Delta t + G_{i+1} z_{i+1}
$$

여기서 $$s_\theta^*$$는 학습된 score 모델로, 역방향 확률 흐름을 따라가며 샘플을 생성하게 된다.

---

정리하자면, 다양한 샘플러들은 forward SDE를 수치적으로 풀거나,  
score function을 기반으로 역방향 경로를 복원하는 방식으로 동작한다.  
사용하는 SDE의 종류와 원하는 정밀도에 따라 적절한 샘플러를 선택하면 된다.

---


### Predictor-Corrector 샘플러

Reverse-time SDE를 그대로 수치적으로 푸는 것도 가능하지만, **예측 + 보정** 과정을 분리해서 수행하면 더 안정적이고 성능이 좋다. 이를 **Predictor-Corrector (PC) 샘플러**라고 한다.

---

#### Predictor 단계

우선 SDE Solver(Euler-Maruyama 등)를 사용해서 다음 위치를 예측한다.  
DDPM의 ancestral sampling이나 VE SDE의 reverse diffusion 방식이 여기에 해당한다.

예를 들어 VE SDE에서는 다음과 같이 update 된다:

$$
x_i' = x_{i+1} + (\sigma_{i+1}^2 - \sigma_i^2) \cdot s_\theta^*(x_{i+1}, \sigma_{i+1})
$$

그리고 노이즈 샘플링 후:

$$
x_i = x_i' + \sqrt{\sigma_{i+1}^2 - \sigma_i^2} \cdot z, \quad z \sim \mathcal{N}(0, I)
$$

---

#### Corrector 단계

예측된 샘플이 타겟 분포와 얼마나 잘 맞는지를 평가해서 보정한다.  
여기에는 **Langevin Dynamics**가 주로 사용된다.

보정은 다음과 같은 방식으로 반복적으로 수행된다:

$$
x \leftarrow x + \epsilon \cdot s_\theta^*(x, t) + \sqrt{2 \epsilon} \cdot z, \quad z \sim \mathcal{N}(0, I)
$$

$\epsilon$은 step size이며, 보통 adaptive하게 정해진다. 예를 들어 아래와 같이:

$$
\epsilon = 2\alpha \cdot \frac{\|r\|_2^2}{\|g\|_2^2}, \quad r \sim \mathcal{N}(0, I), \quad g = s_\theta^*(x, t)
$$

이렇게 하면 샘플이 현재 score 방향으로 이동하며 실제 분포에 더 가까워지도록 보정된다.

---

#### 전체 PC 알고리즘

하나의 스텝에서 predictor와 corrector가 함께 사용되며, 이를 반복해서 최종 샘플 $$x_0$$을 생성한다.  
논문에서는 **Algorithm 2~5**로 VE SDE와 VP SDE 각각의 PC 샘플링이 소개되어 있다.

- Predictor는 Euler나 Ancestral 방식
- Corrector는 Langevin 보정
- 두 단계를 N번 반복 (예: PC1000이면 predictor+corrector를 각각 1000번)

---

#### 왜 Corrector가 중요한가?

VE SDE에서는 Corrector가 성능 개선에 큰 역할을 한다. 예를 들어:

- **P2000** (predictor 2000번만): FID ↑
- **PC1000** (predictor+corrector 1000번씩): FID ↓

비슷한 계산량임에도 PC1000이 더 좋은 결과를 보여준다.

| Sampler      | VE SDE (SMLD) | VP SDE (DDPM) |
|--------------|---------------|---------------|
| P1000        | 4.98 ± .06     | 3.24 ± .02     |
| P2000        | 4.88 ± .06     | 3.24 ± .02     |
| PC1000       | **3.62 ± .03** | **3.21 ± .02** |
| C2000        | bad            | bad            |

- 보정만 해도 성능이 안 좋고  
- 예측만 늘리는 것보다 보정을 같이 넣는 게 효과적임

---

#### 요약

- Predictor는 SDE Solver로 다음 상태를 예측
- Corrector는 Langevin Dynamics로 score 방향 보정
- 특히 VE SDE에서는 큰 효과를 보이며, 적은 스텝으로도 좋은 샘플링 결과를 낼 수 있음

---

### Probability Flow ODE

SDE는 본질적으로 stochastic한 경로를 따르지만, 그와 동일한 분포를 갖는 deterministic한 ODE도 존재한다. 이걸 **Probability Flow ODE**라고 부른다.

이 ODE는 다음과 같이 쓸 수 있다:

$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)
$$

이걸 사용하면 여러 가지 장점이 생긴다:

- 모델의 정확한 likelihood 값을 계산할 수 있고,
- latent representation을 명확하게 조작할 수 있다 (예: interpolation),
- 각 데이터는 하나의 trajectory로 인코딩되기 때문에 latent 표현이 모호하지 않다,
- SDE보다 빠르게 샘플링할 수 있다 (확률적 요소가 없으니까).

---  
#### 기본 개념

주어진 SDE:

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

에 대해, 동일한 분포를 따르는 ODE는 다음과 같이 표현할 수 있다:

$$
dx = \left[ f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x) \right] dt
$$

이 식은 stochasticity 없이도 동일한 확률 분포를 유지하게 해준다.

---

#### Continuous Objective로의 확장

기존 DDPM은 discrete noise schedule을 따라가며 loss를 정의했지만,  
Probability Flow ODE는 **continuous time loss**를 가능하게 한다:

- 기존 (Discrete DDPM Loss):

$$
\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

- 연속적 objective:

$$
\mathcal{L}_{\text{cont}} = \mathbb{E}_{x_0, t, \epsilon} \left[ \left\| s_\theta(x_t, t) - \nabla_x \log p_t(x) \right\|^2 \right]
$$

이로 인해 시간에 대한 연속적 학습이 가능해지며, DDPM보다 더 일반화된 프레임워크로 확장된다.

---

#### Exact Likelihood Computation

기존 diffusion 방식은 ELBO 기반 추정만 가능했지만,  
Probability Flow ODE는 **정확한 likelihood 계산**을 지원한다.

이는 다음과 같은 **Instantaneous Change of Variables Formula**를 통해 이루어진다:

$$
\log p(x_0) = \log p(x_T) + \int_0^T \nabla \cdot f_\theta(x(t), t) dt
$$

이 방식은 실제 likelihood 값을 정확히 계산할 수 있어, 모델 성능 비교에 있어 더 신뢰할 수 있는 수치를 제공한다.

- 실제 실험에서도 ELBO 기반보다 NLL(negative log-likelihood)이 더 낮게 나온다.

---

#### Latent Representation 조작

Probability Flow ODE의 가장 큰 장점 중 하나는 **데이터 인코딩 경로가 deterministic**하다는 점이다.

- 모든 데이터 포인트 $$x(0)$$는 고유한 latent point $$x(T)$$로 매핑된다.
- 따라서 같은 $$x(0)$$은 항상 같은 $$x(T)$$로 가며, 반대 방향도 마찬가지이다.

이를 활용하면 다음과 같은 조작이 가능해진다:

- Latent 공간에서의 **interpolation**  
- **temperature scaling**  
- **conditional decoding**  
- 이미지 **편집** 및 **inpainting**

---

#### Uniquely Identifiable Encoding

기존 invertible models처럼 $$x(0)$$과 latent trajectory 간의 1:1 대응이 보장된다.

왜 그럴까?

- Probability Flow ODE는 학습된 **score function**을 따라 동일한 trajectory를 구성한다.
- stochastic한 노이즈 없이 deterministic한 update만 존재하므로 trajectory가 고정된다.

---

#### Efficient Sampling

- 확률적 SDE는 noise sampling이 필수지만,  
  Probability Flow ODE는 단순한 ODE integration만 수행하면 된다.

- 따라서 sampling 속도 측면에서 **Neural ODE 기반 방식이 더 빠르다.**

다만 정확도 vs 효율성 사이의 trade-off가 있으므로,  
적절한 상황에서 선택적으로 사용하는 것이 좋다.

---

### Controllable Generation

Probability Flow ODE는 **조건부 생성 (Conditional Generation)**도 지원한다.  
기존 unconditional score에 조건부 분포의 gradient를 더해준다:

$$
dx = \left\{ f(x, t) - g(t)^2 \left( \nabla_x \log p_t(x) + \nabla_x \log p_t(y \mid x) \right) \right\} dt + g(t)\,d\bar{w}
$$

이때:

- 첫 번째 항: 학습된 unconditional score function
- 두 번째 항: 조건 분포에 대한 gradient

이렇게 하면 다음과 같은 생성이 가능하다:

- 클래스 조건 생성 (예: "말" 이미지만 생성)
- Inpainting (이미지의 일부분만 채우기)
- Colorization (흑백 → 컬러 전환)

---

### 요약

| 항목                     | DDPM                      | SDE + PF ODE                    |
|--------------------------|---------------------------|----------------------------------|
| 샘플링 경로              | Stochastic                 | Deterministic                   |
| Likelihood 계산          | Approx. (ELBO)            | Exact (ODE 기반)               |
| Latent 표현              | 모호함                    | 고유한 인코딩 가능             |
| 조건 생성                | 추가 구조 필요            | Gradient 추가로 가능           |
| 샘플링 속도              | 느림 (확률성)             | 빠름 (ODE solver)              |

---

Probability Flow ODE는 단순히 stochastic한 diffusion의 대체가 아니라,  
**더 정밀하고 효율적인 generative modeling**의 가능성을 보여준다.

### 정리

이번 Part 2에서는 score-based generative modeling의 핵심 구성 요소인 SDE를 **어떻게 실제로 풀고 샘플링할 수 있는가**에 초점을 맞췄다.

- 먼저, Euler-Maruyama와 같은 **기초적인 수치 해석 방법**부터 시작해,
- 학습된 score function을 활용한 **reverse-time SDE 구성 방식**,
- 그리고 이를 더 정교하게 풀어내는 **Predictor-Corrector 샘플러**까지 살펴봤다.

또한, 확률적인 SDE와 동일한 분포를 가지면서도 결정론적으로 동작하는 **Probability Flow ODE**를 통해:

- 정확한 likelihood 계산이 가능하고,
- latent representation을 조작하거나,
- 조건부 생성까지 가능한 **강력한 프레임워크**가 된다는 사실을 확인했다.

결과적으로 score-based 모델은  
**확률적/결정적 방법을 넘나드는 유연성**과  
**고정밀 생성 성능**을 동시에 갖춘 현대적인 생성 모델링 접근 방식임을 보여준다.
