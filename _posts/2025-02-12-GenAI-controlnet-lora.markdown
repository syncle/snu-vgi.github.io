---
layout: post
title:  "GenAI Lecture: ControlNet & LoRA"
date:   2025-02-12 15:30:00 +0900
categories: Gen AI Lecture
---

## ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models (ICCV 2023)

### Motivation

Stable Diffusion 1.5[3]와 같은 Text-to-Image Diffusion Model은 텍스트 프롬프트만으로 높은 quality의 이미지를 생성할 수 있지만, 유저 입장에서 이미지의 공간적 구성이나 구조적 특성을 세밀하게 제어하는 것은 여전히 어려운 일이다. 사용자가 원하는 정확한 레이아웃, 포즈, 형태 등을 텍스트만으로 표현하는 것은 한계가 있으며, 이상적인 결과를 얻기 위해서는 프롬프트를 반복적으로 수정해보며 시행착오를 거쳐야 하는 경우가 많기 때문이다. 따라서, diffusion model을 이용한 text-to-image generation에 추가적으로 이미지 형태의 spatial conditioning을 반영해줄 필요가 있는데, 이러한 필요성이 ControlNet[1]의 motivation이 된다.

### Key Idea

ControlNet[1]은 pretrained text-to-image diffusion model에 spatial conditional control을 효율적으로 추가할 수 있도록 한 아키텍처이다. 우선 neural block 한 개에 ControlNet을 적용한 상황을 살펴보자.

<div align="center">
  <img src="https://github.com/lllyasviel/ControlNet/raw/main/github_page/he.png" width="70%">
  <p>ControlNet Figure 2</p>
</div>

Figure 2는 neural block 한 개에 ControlNet을 적용하는 방법을 도식화한 것이다. (a)에서와 같이 Feature map $$x$$(shape: [h, w, c])를 input으로 받고, 또다른 feature map $$y$$(shape: [h, w, c])를 output하는 neural block이 있다고 하자. 이 block에 대해 (b)와 같이 ControlNet을 적용하려면, 우선 block을 copy한 후, 복제본은 trainable, 원본은 not trainable하게 설정한다. 또, trainable한 zero convolution 두 개를 추가한다. (이 둘은 파라미터를 공유하지는 않는 별개의 block이다!) 이제 control feature $$c$$를 neural block에 적용할 준비가 끝났다. 

우리는 $$c$$를 첫 번째 zero convolution에 통과시킨 후, 이를 $$x$$와 합하여 trainable copy의 input으로 삼는다. 이후, trainable copy의 output을 두 번째 zero convolution에 통과시킨 후, 원본 block의 출력 $$y$$와 더해준다. 이렇게 만들어진 $$y_c$$가 ControlNet이 적용된 neural block의 output이 된다. 이를 수식으로 나타내면 다음과 같다.

$$y_c = F(x; \Theta) + Z(F(x + Z(c; \Theta_{z1}); \Theta_c); \Theta_{z2})$$

여기서 $$F(x; \Theta)$$는 원본 모델 블록의 출력, $$\Theta_c$$는 복제된 블록의 파라미터, $$Z(\cdot; \Theta_z)$$는 zero convolution 레이어, $$c$$는 condition feature에 해당한다. (여기서, 두 zero convolution layer는 각각 서로 다른 parameter $$\Theta_{z1}$$와 $$\Theta_{z2}$$를 가지는 독립적인 network임에 유의하자.)

ResNet block, Conv-BN-ReLU block, multi-head attention block, transformer block 등 다양한 block에 이러한 ControlNet 아키텍처를 적용할 수 있다.

이러한 설계는 여러 장점을 가진다. 

먼저, ControlNet을 적용할 때 원본을 freeze하기 때문에 원래의 generation 능력이 그대로 보존한다. 직관적으로 말하면, ControlNet의 output은 initialize될 때부터 원본의 output에서 출발하며, 그에 따라 train 과정에서 condition을 반영하면서 더 개선될 일만 남은 셈이다. 이는 절대 trivial한 장점이 아니다. Spatial control을 잘 반영하는 모델을 만들기 위해서는 (training free 방법에는 한계가 있으므로) data-driven approach를 택할 수밖에 없는데, 이때 naïve하게 pretrained diffusion model을 fine-tuning하게 될 경우 자칫하면 model이 fine-tuning data에 overfitting되어 도리어 원래 갖고 있던 일반적인 image domain knowledge를 잊어버리는 catastrophic forgetting이 발생할 수 있기 때문이다. 이러한 걱정이 없다는 것은 ControlNet의 가장 큰 장점이자, 똑같은 block을 2개 복제하면서 GPU memory 요구량이 늘어나는 부분에 대한 정당성을 부여해주는 요소가 된다.

다음으로, 원본 모델과 copy된 부분이 zero convolution layer로 연결되기 때문에 오는 이점이 있다. Zero Convolution은 1×1 convolution layer로, weight과 bias 모두 0으로 초기화된 것을 의미한다. 즉, $$Z(\cdot; \Theta_{z1}) = Z(\cdot; \Theta_{z2}) = 0$$로 initialize되는 것이다. 따라서, 학습 시작 시점에는 $$y_c = y$$가 되므로 ControlNet에서 제안하는 zero convolution은 학습 초기 시점에 원본 모델의 출력에 아무 영향도 주지 않는 것을 알 수 있다. 이후 학습이 진행됨에 따라 점진적으로 가중치가 조정되면서 zero convolution은 원본 블록에 유용한 정보만 전달하게 된다. 만약 우리가 이 convolution network의 weight를 random initialize했더라면, 학습 초기에 이 network는 random output 즉 noise를 생성했을 것이고, 이것이 pretrained model에 주입되면 모델의 성능이 원래에 비해 저하되었을 것이다.

또, 앞에서 확인했다시피 ControlNet의 아키텍처는 control feature vector $$c$$의 종류에 무관하게 정의되며, 다양한 conditional control(edge, depth map, pose 등)에 적용했을 때 좋은 성능을 보여준다는 것이 실험적으로 증명된 바 있다.

마지막으로, 50k 미만 등 적은 양의 데이터로도 효율적인 학습이 가능하다는 점 역시 ControlNet의 중요한 장점이다. 처음부터 pretrained weight를 copy해온 덕분에 적은 양의 데이터로도 충분히 잘 학습되고, zero convolution 덕분에 catastrophic forgetting도 없기 때문으로 이해할 수 있다.

### Application to Stable Diffusion 1.5

ControlNet은 다양한 diffusion model에 적용 가능하지만, 논문에서는 우선 Stable Diffusion 1.5[3]에의 적용 사례를 소개하고 있다. Stable Diffusion 1.5의 U-Net 아키텍처는 encoder, middle block, 그리고 decoder로 구성되어 있으며, ControlNet은 이 중 encoder와 middle block을 복제하여 train한다.


<div align="center">
  <img src="https://github.com/lllyasviel/ControlNet/raw/main/github_page/sd.png" width="100%">
  <p>ControlNet Figure 3</p>
</div>

먼저, 512×512 사이즈의 pixel space image가 input condition으로 들어온다. 이후, SD와의 호환을 위해서 별도의 4x4 convolution layer를 4개 사용한 작은 network를 통해 이를 64×64 크기의 condition feature로 변환해주는 구조이다. 이것이 Figure 3에 Condition으로 표기되어 있는 부분이다.

다음으로, ControlNet 구성을 위해 Stable Diffusion 1.5의 encoder(64x64, 32x32, 16x16, 8x8 4종류가 각 3개씩 있으므로 12개 block)와 middle block(1개)을 복제하여 trainable parameter로 설정했다. Stable Diffusion의 encoder는 특성 상 타양한 conditional control을 학습하는 데도 강력한 성능을 보여주는 좋은 backbone으로 기능할 수 있다. 참고로 decoder는 copy 대상이 아니다.

마지막으로, copy된 블록의 output이 zero convolution layer를 통해 freeze된 원본 모델의 각 block들의 output에 합산되도록 Figure 3과 같이 zero convolution을 배치했다. 이들은 서로 독립적인 layer이며, copy된 block들과 마찬가지로 trainable하고, zero convolution이므로 train 시작 시점에 weight와 bias가 0으로 initialize된다. 결국 크게 보면 Figure 2와 같은 구조가 중첩되어 있는 것이라고 볼 수 있다.

### Training

ControlNet의 training objective는 diffusion model의 것과 마찬가지로 다음과 같이 정의된다.

$$L = \mathbb{E}_{z_0,t,c_t,c_f,\epsilon \sim \mathcal{N}(0,1)}[||\epsilon - \epsilon_\theta(z_t, t, c_t, c_f)||^2_2]$$

여기서 $$z_0$$는 원본 이미지, $$z_t$$는 perturbed image, $$t$$는 diffusion timestep, $$c_t$$는 텍스트 프롬프트, $$c_f$$는 conditioning image(예: edge, depth map), $$\epsilon$$은 noise,
$\epsilon_\theta$는 noise prediction network에 해당한다.

학습 중에는 50%의 확률로 텍스트 프롬프트 $$c_t$$를 빈 문자열로 대체하여, 모델이 condition image로부터 직접 의미론적 정보를 인식하도록 유도한다.

한편, ControlNet의 학습 과정에서 발견되는 특징 중 하나는 sudden convergence phenomenon이다. Zero convolution 덕분에 모델은 학습 초기부터 항상 뛰어난 quality의 이미지를 생성하지만, 특정 시점에서 갑자기 control을 따르는 output을 만들어내기 시작한다는 것이 요지이다. 이 현상은, 학습 초반에는 원본 모델의 성능이 그대로 유지되다가, 학습이 진행됨에 따라 gradient가 쌓이면서 어느 임계점에서 copy된 블록이 condition을 반영하는 역할을 갑자기 습득하는 것으로 해석할 수 있다.

### Inference

ControlNet에서는 Classifier-Free Guidance(CFG)[5]에 Resolution Weighting이라는 개념을 추가로 도입했다. 일반적인 CFG는 $$\epsilon_{prd} = \epsilon_{uc} + \beta_{cfg}(\epsilon_c - \epsilon_{uc})$$ 형태로 표현되는데, 여기서 $$\epsilon_{prd}$$는 모델의 최종 output, $$\epsilon_{uc}$$는 unconditional output, $$\epsilon_c$$는 conditional output, $$\beta_{cfg}$$는 사용자가 지정한 가중치이다. 따라서 ControlNet을 통해서 image condition이 추가되면, 이 이미지를 $$\epsilon_{uc}$$와 $$\epsilon_c$$ 모두에 추가할지, 아니면 $$\epsilon_c$$에만 추가할지는 design choice가 된다. 프롬프트 없이 이미지를 생성하는 경우 등의 까다로운 상황에서는 두 output에 image를 모두 추가하면 CFG 효과가 사라질 우려가 있고, 한편 image를 $$\epsilon_c$$에만 추가하면 가이드가 너무 강해질 여지가 있다. 따라서 ControlNet은 이를 해결하기 위해 CFG Resolution Weighting 기법을 제안한다. 이 방법은 image condition을 $$\epsilon_c$$에 추가하고, 각 블록의 해상도에 따라 ControlNet과 Stable Diffusion 사이의 연결에 가중치 $$w_i = 64/h_i$$를 곱하는 방식이다. 여기서 $$h_i$$는 i번째 블록의 크기이다. (즉, 해상도가 낮은 UNet 가운데 부근의 블록일수록 image control이 더 강도 높게 반영되는 것이다!) 이를 통해 CFG strength를 적정 수준으로 조정하면서도 conditional control을 유지할 수 있다.

또 다른 중요한 기능은 Multiple Conditional Composition이다. 하나의 Stable Diffusion instance에 여러 control image(예: Canny 엣지와 포즈)를 동시에 적용하려면, 해당 조건에 맞게 학습된 여러 ControlNet의 output을 Stable Diffusion 모델에 직접 더할 수 있다. 이때 추가적인 가중치를 설정해주거나 linear interpolation 등을 적용해주지 않아도 합성이 가능하다는 편리함이 있다. 이를 통해 유저는 더욱 정교하고 세밀한 control을 할 수 있게 된다.

마지막으로, ControlNet을 이용하면 텍스트 프롬프트 없이도 control image만으로 의미 있는 이미지를 생성할 수 있다. 이는 앞서 언급했다시피 학습 과정에서 50%의 확률로 텍스트 프롬프트를 빈 문자열로 대체한 것이 효과를 발휘한 결과이다. 모델이 image condition에서 직접 의미론적 정보를 추출하는 능력을 갖추게 되어, 프롬프트 없이도 image condition의 구조적 특성과 의미를 반영한 이미지를 생성할 수 있게 된 것이다. 이는 텍스트로 표현하기 어려운 복잡한 구조나 개념을 시각적으로 control하고자 할 때 특히 유용하다.

### Experiments

ControlNet의 실험 결과는 다양한 종류의 condition input에서 모두 뛰어난 성능을 보여주었으며, 기존 Image-to-Image translation 기법들보다 fidelity나 quality 면에서 더 우수하다. 또, Multiple Conditional Composition 실험에서는 pose와 depth 정보를 동시에 적용하여 3D 공간에 정확히 배치된 특정 포즈의 인물 이미지를 생성하거나, edge map과 segmentation을 함께 사용하여 형태와 색상을 동시에 제어하는 등의 복합적인 제어가 가능함을 보여주었다. 또한 프롬프트 없이 control image만으로 생성한 실험에서는 모델이 control image에서 의미론적 정보를 잘 추출하여 합리적인 이미지를 생성할 수 있음을 확인했다. 이러한 다양한 실험 결과는 ControlNet이 높은 수준의 유연성과 적응성을 갖추고 있으며, text-to-image diffusion model의 controllability 면에서 큰 성과를 거두었음을 보여준다.

### Limitation
상술한 여러 장점에도 불구하고, ControlNet은 Diffusion Model 아키텍처의 절반 이상을 copy하고 train해줘야 한다는 연산 부담과, inference 시에도 Diffusion Model 원본의 절반 이상의 GPU 메모리가 추가적으로 요구된다는 부담을 유저에게 안겨주게 된다는 한계가 있다.

---

## LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)

### Motivation

LLM은 일반적인 데이터에 대한 사전 학습 후, 특정 작업이나 도메인에 맞게 adapt시키는 방식으로 활용된다. 하지만 GPT-3 175B와 같이 모델 크기가 커질수록 전통적인 fine-tuning 방식은 여러 현실적 문제에 직면하게 된다. 모든 파라미터를 업데이트하는 full fine-tuning은 막대한 계산 자원과 메모리를 요구할 뿐만 아니라, 다양한 downstream task마다 거대한 모델 전체를 별도로 저장해야 하므로 배포 측면에서도 매우 비효율적이기 때문이다. 또한, 적은 양의 데이터로 fine-tuning할 경우 catastrophic forgetting이 발생할 위험도 있다. 이러한 문제를 해결하기 위해 parameter-efficient한 fine-tuning 방법이 필요하며, 이것이 LoRA[2]의 핵심 motivation이다.

한편, LoRA는 이처럼 NLP 도메인에서 처음 제안되었으나, LLM의 weight matrix의 특질에 기반한 parameter-efficient fine-tuning 방식이 Computer Vision 도메인에서도 좋은 성능을 보인다는 것이 발견되었고, 지금은 Computer Vision 도메인에서도 가장 널리 쓰이고 있는 fine-tuning 방법 중 하나가 되었다.

### Key Idea

LoRA(Low-Rank Adaptation)란 pre-trained 모델의 weight은 freeze한 상태에서, trainable low rank matrix를 통해 모델을 효율적으로 adapt하는 방법을 말한다. 기존의 dense weight를 직접 업데이트하는 대신, 이 업데이트를 low rank matrix들의 곱으로 근사한다는 것이 핵심 아이디어이다. 이는 [8]에서 pretrained language model이 낮은 intrinsic dimensionality를 가진다고 주장한 점에 착안하여 제안된 방법이다.

구체적으로, pre-trained weight matrix $$W_0 \in \mathbb{R}^{d \times k}$$가 있을 때, 이 weight에 대한 업데이트 $$\Delta W$$를 두 개의 low rank matrix의 곱 $$BA$$로 표현한다. 여기서 $$B \in \mathbb{R}^{d \times r}$$, $$A \in \mathbb{R}^{r \times k}$$이며, rank $$r \ll \min(d, k)$$이다. 따라서 adapt된 weight는 다음과 같이 나타낼 수 있다.

$$W = W_0 + \Delta W = W_0 + BA$$

따라서 input $$x$$에 대해 다음과 같은 과정으로 output $$h$$가 도출된다.

$$h = Wx = W_0x + BAx$$

여기서 주목할 점은 $$B$$는 0으로, $$A$$는 Gaussian distribution으로 초기화된다는 것이다. 따라서 fine-tuning 초기에는 $$\Delta W = BA = 0$$이므로, 원래 모델의 output $$W_0 x$$가 그대로 유지된다. (실제 구현에서는 수치 안정성을 위해 $$\Delta W x$$에 scaling hyperparameter $$\frac{\alpha}{r}$$를 곱해준다.)

이러한 설계는 여러 장점을 가진다.

첫째, 원본 모델 weight를 frozen 상태로 유지하므로 전체 모델을 공유하고, task별로 필요한 적은 수의 LoRA 파라미터만 교체하면 되므로 메모리와 저장 공간이 대폭 절약된다. 예를 들어, GPT-3 175B 모델에 $$r=4$$인 LoRA를 적용하면 체크포인트 크기가 약 350GB에서 35MB로 줄어든다. 이 덕분에 원본 모델 한 개에 대해 여러 개의 LoRA를 train해야 하는 경우에도 모델 추가 학습에 따른 한계비용이 크지 않다.

둘째, adaptive optimizer를 사용할 때 대부분의 파라미터, 즉 $$W_0$$가 frozen이므로 train 속도가 향상된다. Optimizer state를 유지할 필요가 없으므로 GPU 메모리 사용량도 줄어든다. GPT-3 175B의 경우 LoRA 적용 시 train 속도가 약 25% 빨라지고, GPU VRAM 사용량은 1.2TB에서 350GB로 감소한다.

셋째, 간단한 linear 설계 덕분에 inference 시에는 $$W = W_0 + BA$$를 미리 계산해두면 추가 레이어 없이 원래의 아키텍처를 그대로 사용할 수 있으므로, adapter[6, 7] 방식과 달리 inference latency가 전혀 발생하지 않는다.

넷째, LoRA는 prefix-tuning과 같은 다른 parameter-efficient fine-tuning 방법과도 쉽게 결합될 수 있다.

### Training and Inference

LoRA의 train은 기존의 일반적인 모델 fine-tuning과 유사하나, 원래 weight $$W_0$$는 freeze 상태이고 $$A$$와 $$B$$ matrix만 업데이트된다는 차이가 있다. 이 방식은 주로 parameter 수를 크게 줄이면서도 모델의 성능을 유지할 수 있다는 장점이 있다.

Loss 함수는 일반적인 LLM train과 동일하며, 다음과 같다.

$$L = -\sum_{t=1}^{|y|} \log p_{W_0 + \Delta W}(y_t|x, y_{<t})$$

여기서 $$x$$는 input context, $$y$$는 output sequence이며, $$W_0 + \Delta W$$는 LoRA로 adapt된 모델을 나타낸다.

train이 완료된 후, inference 단계에서는 $$W = W_0 + BA$$를 미리 계산해두고 일반 모델처럼 사용할 수 있다. 따라서 inference 속도는 full fine-tuning과 동일하게 유지된다.

### Application to Transformer Architecture

LoRA는 Transformer[4] 아키텍처의 여러 weight matrix에 적용될 수 있지만, 모든 matrix에 적용할 필요가 있는 것은 아니다. 실험 결과에 따르면, 특히 self-attention 모듈에서 query($$W_q$$)와 value($$W_v$$) weight matrix에만 LoRA를 적용하는 것이 메모리 효율성과 성능 사이에서 좋은 균형점을 제공한다.

먼저, Transformer 아키텍처의 각 층에는 self-attention 모듈($$W_q$$, $$W_k$$, $$W_v$$, $$W_o$$)과 MLP 모듈의 weight matrix들이 있다. 이론적으로는 모든 weight matrix에 LoRA를 적용할 수 있지만, 파라미터 수를 최소화하기 위해 일부만 선택적으로 적용하는 것이 효율적이다.

LoRA를 적용하면, 각 matrix $$W_i$$는 $$W_i = W_{0,i} + B_iA_i$$로 분해된다. 예를 들어 query weight에 LoRA를 적용한다면 $$W_q = W_{q,0} + B_qA_q$$가 된다. 따라서 transformer 모델의 forward pass 과정은 다음과 같이 계산된다.

$$q = (W_{q,0} + B_qA_q)x$$
$$k = W_{k,0}x$$
$$v = (W_{v,0} + B_vA_v)x$$

여기서 $$W_{q,0}$$, $$W_{k,0}$$, $$W_{v,0}$$는 pre-trained weight로 freeze 상태이고, $$B_q$$, $$A_q$$, $$B_v$$, $$A_v$$만 train의 대상이다.

### Experiments

LoRA는 RoBERTa, DeBERTa, GPT-2, GPT-3 등 다양한 크기의 모델에서 평가되었다. 실험 결과, 놀랍게도 매우 작은 rank($$r = 1$$ 또는 $$r = 2$$)만으로도 대부분의 task에서 full fine-tuning과 비슷하거나 더 나은 성능을 보였다. 이는 model adaptation에 필요한 업데이트의 실제 intrinsic rank가 매우 낮을 수 있다는 함의를 지닌다.

특히 GPT-3 175B에서 MNLI, WikiSQL, SAMSum 등 다양한 태스크에 대해 LoRA를 적용한 결과, 단 몇 MB의 파라미터만으로도 full fine-tuning과 동등하거나 더 나은 성능을 달성했다. 또한 데이터가 적은 상황, 즉 low-data regime에서도 LoRA는 다른 parameter-efficient 방법들보다 우수한 성능을 보였다.

한편, 분석 결과, $$\Delta W$$는 $$W_0$$와 높은 상관관계를 가지지 않는다는 것이 발견되었다. 대신 $$\Delta W$$는 pre-trained 모델에 이미 존재하는 특정 방향(direction)을 강화하는 역할을 하는 것으로 보인다. 이는 LoRA가 단순히 모델 weight를 미세 조정하는 것이 아니라, 특정 task에 중요한 방향을 효율적으로 증폭시키는 메커니즘으로 작동한다는 것을 시사한다.

---

### Reference

[1] Adding Conditional Control to Text-to-Image Diffusion Models, Zhang et al., 2023. [[arXiv:2302.05543](https://arxiv.org/abs/2302.05543)] \
[2] LoRA: Low-Rank Adaptation of Large Language Models, Hu et al., 2021. [[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)] \
[3] High-Resolution Image Synthesis with Latent Diffusion Models, Rombach et al., 2021. [[arXiv:2112.10752](https://arxiv.org/abs/2112.10752)] \
[4] Attention Is All You Need
, Vaswani et al., 2017. [[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)] \
[5] Classifier-Free Diffusion Guidance, Ho et al., 2022, [[arXiv:2207.12598](https://arxiv.org/abs/2207.12598)] \
[6] T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models, Mou et al., 2023. [[arXiv:2302.08453](https://arxiv.org/abs/2302.08453)] \
[7] IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models, Ye et al., 2023. [[arXiv:2308.06721](https://arxiv.org/abs/2308.06721)] \
[8] Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning, Aghajanyan et al., 2020. [[arXiv:2012.13255](https://arxiv.org/abs/2012.13255)]
