---
layout: post
title:  "GenAI Lecture: Improved DDPM"
date:   2025-04-01 15:25:00 +0900
categories: Gen AI Lecture
---

### 1. 개요 (Introduction)

* **DDPM**은 Forward Process를 통해 원본 이미지를 점차적으로 노이즈화 시키고, Reverse Process에서 가우시안 노이즈로 부터 원본 이미지를 샘플링하는 방법으로 이미지를 생성함.
* DDPM은 이미지 생성 퀄리티 자체는 좋았으나, **log-likelihood** 수치가 정말 좋은지 확인이 되지 않았음. Log-likelihood는 생성모델에서 중요한 수치로, 만약 높다면 data distribution을 잘 캡처하고, feature representation이 좋아짐.
* DDPM을 개선하기 위해 Reverse Process에서 **variance**를 fix 하는 것이 아닌 학습 가능하도록 하여 최적의 값을 찾음.

---

### 2. 배경 지식 (Background)

#### 2.1 DDPM[1]

- **Forward Process**
  - 원본 이미지를 점차적으로 노이즈화: $$x_0 \to x_1 \to \dots \to x_T$$
  - $$x_T$$는 거의 가우시안 잡음 상태가 됨.
  - noise scheduler beta를 **linear 하게 증가**시킴.
- **Reverse Process**
  - 학습된 노이즈 제거 모델 $$\epsilon_\theta$$가 역방향으로 이미지를 복원함.
  - noise scheduler **beta를 고정함**. Ho et al.[2]은 beta가 퀄리티에 미치는 영향이 없다고 생각 했음 -> 이는 log-likelihood를 고려하지 않은 결과로 beta를 잘 설정하는 것이 관건임.

### 3. 메소드 (Method)

#### 3.1 Learning Variance

* diffusion step의 초기 단계가 VLB에 가장 많이 기여함.
* 따라서 고정하기 보단, 최적의 variance를 최적화함.
* 다만, variance 자체는 매우 작은 값이기 때문에 $$\beta_t$$와 $$\hat{\beta}_t$$ 사이 보간 값을 결정하는 v를 최적화 하는 방식을 선택함.
* 기존 $$\mu_\theta$$를 최적화하는 $$L_\text{simple}$$에 variance를 최적화하는 $$L_\text{vlb}$$를 추가하여 **Hybrid한 objective를 선택**함.

#### 3.2 Noise Scheduling

* 이미지에 노이즈가 linear하게 추가되는 과정을 plot 해보면 빠르게 노이즈가 추가되어서 후반에는 노이즈가 큰 영향을 끼치지 못함.
* 초반에 과도하게 노이즈가 더해지는 과정을 방지하게 Linear 스케줄링 대신 **Cosine 스케줄링**을 선택함.

#### 3.3 Reducing Gradient Noise

- $$L_\text{simple}$$, $$L_\text{vlb}$$, $$L_\text{hybrid}$$ 비교
  - $$L_\text{vlb}$$만 단독 사용했을 때 성능이 좋을 것이라 판단했었으나, 실전에는 부적합했음. Why?
  - gradient를 plot하니 noisy해서 optimization에 어려움을 겪었음.
  - **Importance Sampling**을 적용하면 smooth한 gradient를 만들 수 있음. 이전 10개 값의 유지하면서 Loss에 비례하게 sampling. Loss가 초반에 큰 상황을 방지하고 gradient를 smooth하게 만들 수 있음.
  - $$L_\text{vlb}$$(resampled)은 Log-likelihood를 개선 시키지만, 다른 Metric인 FID score는 Hybrid Loss가 성능이 좋았음.

---

### 4. 실험 결과 (Experiments)

#### 4.1 ImageNet & CIFAR-10

- $$L_\text{simple}$$, $$L_\text{vlb}$$(resampled), $$L_\text{hybrid}$$ 비교
- ImageNet, CIFAR-10 실험
- **Cosine + $$L_\text{vlb}$$**는 NLL(Negative Log Likelihood)을 개선하지만 FID가 떨어짐.
- Cosine + $$L_\text{hybrid}$$는 NLL(Negative Log Likelihood), FID 둘 다 개선함.

#### 4.2 Convolution-based & Transformer-based

- Convolution-based 보다 나은 성능, Transformer-based보다 떨어지는 성능을 보임.

#### 4.3 Sampling Speed

- $$L_\text{simple}$$은 sample step을 줄이면 성능이 급격하게 떨어짐.
- DDIM[3]에 $$L_\text{hybrid}$$ 적용한 결과는 좋지 않았음.
- Step 수가 적으면 DDIM 보다 성능이 안좋지만, Step 수가 많아지면 DDIM을 역전하여 성능 좋아짐.

#### 4.4 GAN

- FID, Precision, Recall을 비교함.
- 더 높은 FID, Recall을 기록함.
- data distribution을 잘 커버함.

#### 4.4 Scalability

* computation 양과 모델 크기를 증가할 때 성능(FID, NLL)도 증가함.

---

### 5. 결론 (Conclusion)

- **핵심 요약**: DDPM의 log-likelihood 수치를 개선함.

  1) Reverse Process의 노이즈를 학습 가능하도록 Hyrid Objective를 설정함.
  2) Forward Process의 noise sceduling을 linear가 아닌 cosine이도록 수정함.
  3) $$L_\text{vlb}$$만 사용하는 경우, importance sampling을 하여 smooth 한 gradient 개선 = FID 성능 증가, NLL 성능 하락함.
- **의의**

  - 단순한 방법으로 DDPM을 개선하였고, 다수의 실험으로 증명함.
  - Sampling 수를 줄여도 성능이 방어되는 모습을 보임.
  - Scalable 한 모습 확인함.

---

### 6. 참고문헌 (References)

1. **Denoising Diffusion Probabilistic Models**, Ho et al., NeurIPS 2020.
2. **Denoising diffusion implicit models**, Song et al. ICLR 2021