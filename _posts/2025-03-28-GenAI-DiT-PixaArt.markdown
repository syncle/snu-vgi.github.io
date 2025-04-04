---
layout: post
title:  "GenAI Lecture: DiT and PixArt Alpha"
date:   2025-03-28 14:52:12 +0900
categories: Gen AI Lecture
---

# DiT

## Background

해당 논문의 이해를 위해서는 Transformer 모델과 U-Net Architecture에 대한 기초적인 이해가 필요하다. 따라서 필요한 경우 아래의 논문들을 참조하면 된다.

[Attension is All You Need (2017.6)](https://arxiv.org/abs/1706.03762)
[U-Net: Convolutional Networks for Biomedical Image Segmentation (2015.5)](https://arxiv.org/abs/1505.04597)

간단히 살펴보자면, Transformer는 Self-Attention 매커니즘을 통해 token을 받아 다음에 올 token을 예측하는 방식으로 작동하고, U-Net의 경우 latent space를 이용해 크기를 줄였다가 다시 복원하는 방식으로 작동한다고 이해할 수 있다.

![]({{"/assets/Pasted image 20250328120521.png"}})
![]({{"/assets/Pasted image 20250328120536.png"}})
Transformer는 자연어 처리, 컴퓨터 비전, 강화학습 등 다양한 분야에 사용되어 왔지만 유독 Diffusion 분야에서는 기존의 U-Net이 범용적으로 사용되고 있었다. 이전까지 다루었던 Classifier-Free Guidance 등도 전체적인 U-Net 구조를 바꾸는 것이 아닌, sampling technique을 개선하는 방식으로 이루어진 연구였다.

이에 이 논문에서는 U-Net을 Transformer로 대체할 수 있는 Diffusion 모델의 구조를 제시하였고, 이것이 이 논문의 핵심적인 내용이다.

Transformer는 데이터와 계산량을 늘릴수록 직관적으로 performance가 증가하는 성질을 지니고 있으므로, 이를 도입한다면 Diffusion 분야에도 이러한 성질을 유용하게 적용할 수 있을 것이라는 아이디어이다. 실제로 뒤의 결과에서도, 계산량과 성능 사이에 유의미한 상관관계가 나타났다.

## Architecture Complexity Measures

본격적인 논의를 시작하기에 앞서, 다음 질문을 생각해 보자.

'우리는 어떻게 어떤 architecture의 복잡도를 계산할 수 있을까?'

가장 간단히 떠오르는 답은 '사용되는 parameter의 개수를 센다' 이다. 그러나, 이를테면 이미지를 처리할 때 해상도의 차이 등은 parameter count에 제대로 반영되지 않기 때문에 이 논문에서는 parameter count 대신 Gflops로 측정한 직접적인 계산량을 복잡도의 단위로 사용한다. 

## Diffusion Recap

Diffusion에 관해 간단하게 복습하면, 우리가 가지고 있는 ELBO를 최대화하는 것이 목적이다. ELBO는 $$\mathcal{L}(\theta)=-p(x_0 | x_1) + \sum_t \mathcal{D}_{KL} (q^*(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))$$
로 나타낼 수 있고, 여기서 $$q^*, p_\theta$$가 모두 Gaussian이므로 우리는 $$D_{KL}$$ 항을 평균 $$\mu_\theta$$, 분산 $$\Sigma_\theta$$를 가지는 Gaussian으로 생각할 수 있다.

또한, 우리는 $$\mu_\theta$$를 $$\epsilon_\theta$$로 reparameterize함으로서 위의 ELBO를 보다 간단하게 만들어
$$\mathcal{L}_{simple}(\theta)=\|\epsilon_\theta(x_t)-\epsilon_t\|^2$$로 쓸 수 있다. 

여기서는 $$\epsilon_\theta$$를 훈련시킬 때는 $$\mathcal{L}_{simple}$$을 사용하고, $$\Sigma_t$$를 훈련시킬 때는 위의 full $$\mathcal{L}(\theta)$$를 사용한다.

또한, 아래와 같은 [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) 역시 사용한다.
$$\nabla_x \log p(x|c) \propto \epsilon_\theta (x_t, \emptyset) + s(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$
여기서 parameter $$s$$가 guidance의 scale을 결정함에 유의하자.

마지막으로, 우리는 [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) 역시 사용할 것인데, 즉 Transformer를 전체에 적용하는 것이 아니라 Autoencoder를 사용해 압축을 먼저 진행한 후 그 이후에 Transformer를 적용한다.

## Diffusion Transformer Design Space

그렇다면 이제 핵심적인 DiT Block이 어떻게 설계되었는지 보자. 저자는 아래 사진과 같은 4개의 Setting을 제시했는데, 결과적으로 그 중 adaLN-Zero가 가장 좋은 성능을 보였다.

이 장에서는 각각의 구조에 대해 간단히 설명하고, 특징을 서로 비교해 보려고 한다.

![]({{"/assets/Pasted image 20250328122431.png"}})

그 전에, 먼저 Patchfying에 대해 알아보자. Transformer는 Token을 입력으로 받으므로 입력을 1차원으로 만들 필요가 있다. 우리가 가지고 있는 이미지는 2차원 형태이므로, 이것을 일정한 크기로 잘라서 linearly embedding하면 되는데 이를 Patchfying이라 한다.

$$p$$는 patch size를 나타내는 hyperparameter이고, $$I$$는 latent image, 즉 autoencoder를 거쳐 압축된 이미지의 (가로) 길이이다. 당연히 $$p$$를 절반으로 줄이면 각각의 조각이 더 작아지게 되므로 더 나은 성능이 나오지만, 2차원 이미지를 절반의 길이로 잘게 자르는 것이므로 계산량은 (최소) 4배 증가하게 된다. (이 경우에도 전체 parameter 수는 일정하므로, 왜 위에서 총 계산량을 Complexity measure로 두었는지 알 수 있다.)

![]({{"/assets/Pasted image 20250328122918.png"}})

그렇다면 이제 각 DiT Block의 구조를 살펴보자.

### In-Context Conditioning

![]({{"/assets/Pasted image 20250328123000.png"}})
해당 구조는 condition($$t, c$$)을 그냥 input 뒤에 그대로 붙여서 처리한 다음, 모든 처리가 끝난 이후에 해당 conditioning token들을 sequence에서 제거하는 방식을 사용한다.

단지 두 개의 token만이 처리에 추가적으로 사용될 뿐이므로 추가적인 overhead는 적지만, 성능이 높지 않다.

### Cross-Attention

![]({{"/assets/Pasted image 20250328123149.png"}})
이 경우에는 condition($$t, c$$)을 input token들에 바로 붙이는 것이 아니라, input만 Self-Attention을 통과시킨 뒤 Condition을 (Multi-Head) Cross-Attention Block에 넣는다.

성능은 앞의 In-context conditioning보다 좋지만, 15% 정도의 가장 큰 overhead를 추가한다는 점에서 이 역시 실용적이지 않다.

### adaLN(Adaptive Layer Norm) / adaLN-Zero

![]({{"/assets/Pasted image 20250328123400.png"}})
이 구조에서는 각 Block에 들어가기 전에 Scale/Shift를 수행하는데, 이것은 parameter $$\gamma ,\beta$$에 의해 이루어진다. (논문에 adaLN에 대한 그림이 없어, adaLN-Zero의 그림을 대신 첨부하였음.)

여기서 $$\gamma, \beta$$가 직접적으로 학습되는 것이 아닌, condition을 이용해 $$t+c$$로부터 Regression한다는 점에 주의하자. 단지 새로운 scaling/shifting parameter를 추가하는 것뿐이므로 adaLN은 위의 두 개의 모델과 비교해 가장 적은 overhead를 만든다.

adaLN-Zero의 경우 Block의 처리 이후에 추가적인 Scaling parameter $$\alpha$$를 배치한다는 것이 특징이다. 이름에 'Zero'가 포함된 이유는 초기에 MLP가 Zero를 내뱉도록 Initialization을 수행하는 Zero-Initialization 때문이다. (즉, 전체 DiT Block이 identity function으로 Initialize된다.)

### Block Architecture Comparison

이제 위에 제시된 네 개의 구조를 비교해 보자. FID Score는 낮을수록 결과가 좋은 것인데, 거의 제시된 순서를 따라 성능이 분포하고 있음을 확인할 수 있다.
![]({{"/assets/Pasted image 20250328124152.png"}})

## DiT Model Settings

이제, 논문에서 수행된 실험에 대한 Model Setting을 알아보자. 맨 앞의 $$N$$은 위의 DiT Block을 몇 번 반복했는지, $$d$$는 latent space의 size, 이외의 값은 Attention Head 수/계산량(Gflops)이다. 아래의 자료에서도 알 수 있듯, XL이 넷 중 가장 큰 모델임을 확인할 수 있다.
![]({{"/assets/Pasted image 20250328124555.png"}})
또한, Patchify 과정에서 사용한 hyperparameter $$p$$도 결합하여 $$p=2$$일 때 DiT-XL/2와 같이 나타낼 것이다.

## Experimental Setup

아래는 이 논문의 실험 과정에서 사용한 Setup인데, 지엽적인 내용이 많으므로 필요한 부분만 확인하여도 된다.

- Zero-initialize the final linear layer
- Use Standard weight initialization techniques from [ViT](https://arxiv.org/abs/2010.11929) 
- AdamW optimizer, horizontal flips, constant Adam $$\beta_1, \beta_2$$ parameters
- Constant Learning rate of $$1\times 10^{-4}$$, no weight decay
- Batch Size = 256
- Maintain an EMA of DiT weights with a decay of 0.9999
- Training hyperparameters are almost entirely retained from [ADM](https://arxiv.org/abs/2105.05233) : such as $$t_{max}=1000$$, Linear variance schedule from $$1\times 10^{-4}$$ to $$2\times 10^{-2}$$ 
- Use pre-trained VAE model from Stable Diffusion : encoder has downsample factor of 8 

## Evaluation Metrics

평가 기준은 250 DDPM Sampling step을 기준으로 FID-50K를 사용하였다.

## Results

![]({{"/assets/Pasted image 20250328125336.png"}})
위의 사진에서도 알 수 있듯, Scaling effect가 확연히 나타나고 있다. Model Size를 늘리거나 (S -> XL), Patch size를 줄이는 (8 -> 2) 것으로 성능이 눈에 띄게 증가하는 모습을 확인할 수 있었다.

![]({{"/assets/Pasted image 20250328125440.png"}})
이것은 계산량에 크게 의존하는데, 위에서 보이듯 서로 다른 Config에서도 계산량이 비슷하면 비슷한 performence를 보이는 것을 확인할 수 있었다. 이는 Transformer의 '더 많은 계산량이 더 나은 performance를 만든다'라는 특징과도 부합한다고 할 수 있다.

![]({{"/assets/Pasted image 20250328125549.png"}})
마지막으로, 결국 많은 계산량을 투입할수록 더 큰 DiT 모델이 보다 효율적인 계산을 수행한다는 것을 확인할 수있었다. 예를 들어, $$10^{10}$$ Gflops 이후부터 XL/2가 XL/4보다 효율적임을 확인할 수 있다.

이러한 벤치마크 결과를 가장 뛰어난 모델인 DiT-XL/2를 기준으로 다른 모델들과 비교하면, 아래와 같이 뛰어난 metric을 보인다. (각각 $$256\times 256$$, $$512\times 512$$ 사이즈의 이미지)

![]({{"/assets/Pasted image 20250328125751.png"}})

![]({{"/assets/Pasted image 20250328125806.png"}})
심지어 2.35M Step만 train한 경우에도 2.55의 FID로 여전히 비교군의 다른 모델들을 압도하는 점수를 보였다.

Compute Efficiency 관점에서 보더라도, 아래와 같이 SOTA Diffusion Model들에 비해 더 나은 결과를 확인할 수 있었다.

![]({{"/assets/Pasted image 20250328125927.png"}})

## Scaling Effect vs. Sampling Effect

그런데, Transformer의 Scaling Effect와 별개로, Diffusion model에서 sampling step을 늘리면 성능이 증가하는 Sampling Effect 역시 존재한다. 둘 중 어떤 것이 영향이 클지 알아보기 위해 실험한 결과, Scaling Effect가 (훨씬) 큰 영향을 미침을 확인할 수 있었다. (XL/2, L/2를 비교해 보자.)

![]({{"/assets/Pasted image 20250328130057.png"}})

마지막으로, DiT를 이용해 생성된 이미지를 보며 마무리하자. 오른쪽으로 갈수록 더 큰 모델 (S -> XL)을 사용한 것이고, 아래로 갈수록 더 작은 Patch size $$p$$를 사용한 것이다. (8 -> 2)

![]({{"/assets/Pasted image 20250328130202.png"}})


# PixArt-$$\alpha$$ 

이제 위의 DiT를 적용한 사례로 PixArt-$$\alpha$$를 알아보도록 하자.

## Abstract / Introduction 

해당 논문에서는 text-to-image 생성 모델의 computational resource를 획기적으로 줄일 수 있는 모델을 제시하였다. 아래 그림에서도 확인할 수 있듯, PixArt-$$\alpha$$는 타 모델들에 비해 훨씬 적은 가격만 소모하였다.

![]({{"/assets/Pasted image 20250328130433.png"}})

이 논문에서는 이미지 생성 작업을 세 단계로 구분하여 제시하였는데, 각각의 단계를 서로 다른 모듈에 배분함으로서 효율성을 향상시켰다.
- Natural Image로부터 pixel distribution 학습
- text-image alignment 학습
- image의 aesthetic quality 향상

또한, 아래와 같은 예시를 통해 text-image pair로 구성된 데이터셋의 문제점을 지적하며 Vision-Language model을 통한 Auto-Labeling이 보다 효율적임을 주장하였다. 아래 그림에서도 확인할 수 있듯, VLM인 LLaVA를 활용해 만든 caption이 보다 이미지와 잘 맞고, 널리 활용 가능하다.

![]({{"/assets/Pasted image 20250328130752.png"}})

## T2I Transformer

우리가 살펴볼 T2I Transformer 모델은 앞선 DiT에서 살펴본 Cross-Attention과 adaLN-Zero를 합쳐 놓은 모양새이다. 아래 그림을 참조하자.
![]({{"/assets/스크린샷 2025-03-28 오후 1.08.45.png"}})

일반적인 adaLN에서는 $$S(i) = f^{(i)}(c+t)$$의 방식을 사용하지만, 여기서는 condition이 parameter의 27% 정도라는 큰 overhead를 만드는 것에서 착안하여 $$S=f(t)$$만 계산하고, 작은 trainable embedding만을 사용해 $$S(i)=g(S, E(i))$$와 같은 방식으로 계산한다. 즉, 큰 MLP를 전체 condition에 대해 적용하는 대신 작은 adjusting vector만을 사용하는 방식으로 computing resource를 절약할 수 있는 것이다.

여기서 additional vector인 $$E(i)$$들은 $$c$$가 주어지지 않은 상태에서 특정 $$t$$에 대해 DiT와 같은 output을 내도록 initialize되었다. (이 논문에서는 $$t=500$$을 사용하였다.) 이러한 Initialization으로, 우리는 미리 훈련된 DiT weight들을 그대로 사용할 수 있다.

이러한 구조를 저자들은 AdaLN-single이라 정의하였다.

## Dataset Construction

또한, 앞서 살펴본 것처럼 저자들은 image-text pair에서 주어진 caption을 그대로 사용하지 않고 VLM인 LLaVA를 사용해 생성된 caption을 사용하였다. 이러한 방법을 통해 전체 Noun에 대한 Valid Noun의 비율을 아래처럼 크게 끌어올릴 수 있었다.

![]({{"/assets/Pasted image 20250328131744.png"}})

마지막으로 aesthetic quality를 끌어올리는 데에는 JourneyDB를 사용하였다. 이렇게 세 가지의 목표를 서로 다른 모듈로 분리함으로써 저자들은 보다 효율적인 구현을 할 수 있었다.

## Implementation Details

아래는 본 논문의 실험 과정에서 사용된 Implementation Detail들인데, 필요한 부분만 참고하여 보면 된다.

- Utiilze DiT-XL/2 as the base network
- Adopt a 4.3B Flan-T5-XXL language model for text encoding
- Process 120 text tokens for more detailed captions (usually 77)
- Employ a pre-trained, frozen VAE (Rombach, 2022) for latent extraction
- Image resizing & center-cropping for a uniform size before entering VAE
- Uses Multi-Aspect augmentation to support arbitrary aspect ratios
- Optimized with AdamW, at a constant learning rate of $$2\times 10^{-5}$$ and weight decay of $$0.03$$ 
- Trained on 64 V100 GPUs for ~26 days

## Evaluation Metrics

본 논문에는 총 세 가지의 평가 기준이 제시되어 있는데, 각각은 아래와 같다.

- MSCOCO Dataset을 이용한 FID Score
- T2I-CompBench에서의 Compositionality
- User Study에서의 human-preference rate

## Results

![]({{"/assets/Pasted image 20250328132239.png"}})
결과적으로 다른 모델들과 비교했을 때, PixArt-$$\alpha$$는 눈에 띄게 적은 계산량과 dataset만을 사용하여 좋은 결과를 얻었다. 이를테면 SDv1.5와 비교했을 때, PixArt는 12% 정도의 훈련 시간, 1.25% 정도의 training sample을 사용했지만 FID Score 면에서는 더 좋았다.

![]({{"/assets/Pasted image 20250328132355.png"}})
특히 User Study에서 사람들의 반응 역시 좋았기 때문에, 생성된 이미지의 품질이 우수했음을 확인할 수 있다.

마지막으로, 여러 가지 세팅을 이용해 생성된 이미지들을 확인하고 마무리하도록 하자.
![]({{"/assets/Pasted image 20250328132459.png"}})