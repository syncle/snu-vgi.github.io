---
layout: post
title: "GenAI Lecture: Denoising Diffusion Probabilistic Models"
date: "2025-01-08 15:30:00 +0900"
categories: Gen AI Lecture
---

# DDPM: Denoising Diffusion Probabilistic Models

*Presented by 황민혁*  
*Based on work by Jonathan Ho, Ajay Jain, Pieter Abbeel*

---

## 1. 서론

본 자료는 **Denoising Diffusion Probabilistic Models (DDPM)** 의 핵심 개념과 수식을 정리한 발표 자료입니다.  
데이터에 점진적으로 노이즈를 추가하는 *forward process*와 이를 역으로 제거하는 *reverse process*를 통해 원본 데이터 분포를 복원하는 모델의 원리를 소개합니다.

---

## 2. 모델 개요

### 2.1 핵심 아이디어

- **Forward Process**  
  데이터에 점진적으로 노이즈를 추가하는 과정  
  > *데이터 분포에 서서히 노이즈를 주입하여 복원 가능성을 높임.*

- **Reverse Process**  
  손상된 데이터에서 노이즈를 제거해 원본 분포를 복원하는 과정  
  > *Conditional Gaussian 모델을 통해 각 단계에서 평균과 분산을 추론함.*

- **Denoising Score Matching (DSM)**  
  다양한 노이즈 레벨에서 로그확률의 기울기(스코어)를 추정하여 안정적인 학습 환경을 마련함.

---

## 3. 모델 상세 설명

### 3.1 Forward Process

데이터 \( x_0 \)에 대해 단계별로 노이즈를 추가하는 과정을 다음과 같이 모델링합니다.

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\Big(x_t; \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t \mathbf{I}\Big)
$$

- **노이즈 스케줄 \( \beta_t \)**: 각 시간 \( t \)에서의 노이즈 강도 조절  

- **누적 효과**:  
$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\quad \text{with} \quad \bar{\alpha}_t = \prod_{s=1}^{t} (1-\beta_s)
$$

### 3.2 Reverse Process

노이즈 제거 과정은 학습 가능한 네트워크로 파라미터화되며, 다음과 같이 표현됩니다.

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\Big(x_{t-1}; \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\Big)
$$

- **네트워크 예측**:  
  - \( \mu_\theta(x_t, t) \): 평균 예측  
  - \( \Sigma_\theta(x_t, t) \): 분산 예측  
- **ε-prediction** 기법을 통해 직접 \( x_0 \)이나 \( \mu_\theta \) 대신 \( \epsilon_\theta(x_t, t) \)를 학습할 수 있음.

### 3.3 Loss Function

단순화된 학습 목표는 네트워크가 추가된 노이즈 \( \epsilon \)를 정확히 예측하도록 유도합니다.

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\,x_0,\,\epsilon} \Big[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \Big]
$$

- **의의**: 다양한 노이즈 레벨에서의 denoising 성능을 극대화하여, 안정적이고 효율적인 학습을 달성함.

---

## 4. 실험 결과

### 4.1 샘플 품질

- **평가지표**:  
  - 높은 FID 점수  
  - 우수한 Inception score  
- **시각적 평가**:  
  생성된 샘플이 실제와 유사하며, 디테일이 우수함.

### 4.2 Progressive Coding

- **Rate-Distortion Trade-off**:  
  전 단계별 rate와 distortion 간의 상호관계를 분석하며, 중간 단계에서의 복원 과정에서 원본과의 근접성이 점진적으로 향상됨.

### 4.3 Interpolation

- **이미지 간 선형 보간**:  
  두 이미지 사이의 latent space를 선형 보간하여, 자연스러운 변화와 연속적인 전이를 확인할 수 있음.  
  적절한 단계 수 (\( t \approx 500 \))에서 가장 의미 있는 결과를 도출함.

---

## 5. 관련 연구

- **Generative Adversarial Networks (GANs)**  
  경쟁 학습 구조(Generator와 Discriminator)로 고품질 이미지를 생성하지만, mode collapse 등의 문제가 있음.

- **Variational Autoencoders (VAEs)**  
  확률론적 잠재 공간 모델링을 통해 데이터를 복원하나, 생성된 이미지의 선명도에서 한계가 있음.

- **Energy-Based Models (EBM)**  
  에너지 함수를 통해 데이터 분포를 직접 모델링하며, DSM과 밀접한 연관성을 가짐.

---

## 6. 확장 및 발전 방향

### 6.1 Fast Sampling Techniques

- **최적화 기법**:  
  ODE 기반 샘플링 등으로 샘플링 단계를 단축하여, 수치적 안정성을 향상시킴.

### 6.2 Classifier-Free Guidance

- **조건부 생성 개선**:  
  추가 분류기 없이 조건부 및 비조건부 학습을 동시에 수행하여, 생성 품질 및 유연성을 크게 향상시킴.

### 6.3 Latent Diffusion Models (LDM)

- **저차원 latent space에서의 diffusion**:  
  계산 효율성을 높이고 고해상도 이미지 생성을 가능하게 함.

---

## 7. 결론

- **DDPM의 강점**:  
  노이즈 추가 및 제거라는 단순한 아이디어를 바탕으로 안정적이고 고품질의 생성 모델을 구현함.  
  DSM 및 ε-prediction 기법을 통해 역과정의 학습 효율을 극대화함.
  
- **향후 전망**:  
  Fast sampling 및 조건부 생성 기술과의 결합을 통해, DDPM은 다양한 생성 모델 연구 및 실제 응용에 중요한 역할을 할 것으로 기대됨.

---

*참고문헌*  
- Jonathan Ho, Ajay Jain, Pieter Abbeel, "Denoising Diffusion Probabilistic Models".  
- 발표 자료: 황민혁