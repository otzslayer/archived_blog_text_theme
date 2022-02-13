---
title: Learning to Rank
tags: [LTR, Learning to Rank, Recsys, 추천 시스템]
category: ML
aside:
  toc: true
show_category: true
---


<!--more-->

## Learning to Rank?

>   :bulb: Learning to Rank (LTR)는 검색 결과의 순서를 정하는 것과 같은 **랭킹 문제 (ranking problem)를 해결하기 위해 지도 학습(supervised learning)을 적용**하는 머신러닝 방법론

전통적인 지도 학습 방법인 회귀(regression)나 분류(classification)와 LTR 간의 차이는 다음과 같습니다.

-   회귀 (Regression)
    -   주어진 피처 $\mathbb{X}$에 대해 실수값 $y \in \mathbb{R}$을 예측하기 위해 함수 $f(\mathbb{X})$를 학습
-   분류 (Classification)
    -   주어진 피처 $\mathbb{X}$에 대해 $N$ 개의 클래스 $y \in {1, 2, \cdots, N}$ 을 예측하기 위해 함수 $f(\mathbb{X})$를 학습
-   Learning to Rank (LTR)
    -   **주어진 쿼리 $q$, 관련 있는 아이템 목록 $D$에 대해 ==목록 내의 아이템 순서를 예측==하기 위해 함수 $f(q, D)$를 학습**
    -   따라서 학습/검증/테스트 데이터 모두 피처 $\mathbb{X}$에 대해 관련성 $y$을 Label로 하는 데이터 형태를 갖춰야 함

가장 고전적인 LTR 문제는 **웹 검색 랭킹**이 있습니다. 웹 검색 결과의 순위를 매기는 것은 주어진 검색 쿼리에 대해 결과적으로 일치하는 문서 URL의 관련성 순위를 매겨 더 관련성이 높은 문서를 상위에 표시하는 것입니다. 쿼리 $q$와 $n$ 개의 결과 문서 $D = \{d_1, d_2, \cdots, d_n\}$가 주어졌을 때 쿼리와 주어진 문서간 연관성을 예측하기 위해 함수 $f(q, D)$를 학습하는거죠. 이상적으로 $f(q, D)$는 정렬된 목록 $D^*$를 반환해야 하고 주어진 쿼리 $q$에 대해 연관성이 높은 순서대로 정렬되어야 합니다.

최근 쉽게 접할 수 있는 LTR 문제는 **추천 시스템**이 있습니다. 대부분의 추천 서비스들은 많은 트래픽 때문에 2-stage approach로 모델을 개발하는 경우가 많은데, 많은 아이템에서 추천 후보를 추린 다음 LTR을 통해 해당 후보의 순위를 조정하여 결과를 서빙합니다.

## Approaches to LTR

LTR을 해결하기 위해 사용하는 세 가지 접근법은 Pointwise, Pairwise, Listwise 입니다. 아래 도표는 각 접근법의 특징을 잘 묘사하고 있습니다. 일반적으로 Pointwise < Pairwise < Listwise 의 순서대로 복잡한 방법을 사용하지만 더 좋은 성능을 보입니다.

<center>
  <figure>
    <img src="/assets/images/2022-02-13-learning-to-rank/learning_to_rank.png" alt="Approaches to LTR" style="zoom:50%;" />
    <figcaption style="text-align: center;">Image from <a href="https://jobandtalent.engineering/learning-to-retrieve-and-rank-intuitive-overview-part-iii-1292f4259315">[1]</a></figcaption>
  </figure>
</center>

### Pointwise

Pointwise approach는 모든 데이터 인스턴스를 개별로 보는 접근법입니다. 예를 들어 두 개의 쿼리가 있고, 각각의 쿼리에 두 개, 세 개의 문서가 매칭되었다고 가정해보겠습니다.

```
q1 -> d1, d2
q2 -> d3, d4, d5
```

각 쿼리에 매칭된 문서들을 풀어 헤쳐서 (쿼리, 문서) 형태로 재구성 합니다.

```
x1: q1, d1
x2: q1, d2
x3: q2, d3
x4: q2, d4
x5: q3, d5
```

각각의 문서들은 모두 독립적인 label 값을 갖게 되고 이를 학습하게 됩니다.

이는 결국 전통적인 회귀나 분류 문제와 크게 다른 것이 없습니다. 매우 단순한 방법으로 이미 다른 분야에서 많이 사용하고 있는 ML 모델들을 그대로 적용할 수 있다는 장점이 있습니다. 하지만 각각의 쿼리에서 매칭되는 문서 목록의 전체 정보를 제대로 활용하지 못하기 떄문에 **최적의 결과를 얻지 못하는 경우가 많습니다.**

### Pairwise

Pairwise approach는 Pointwise 비슷하지만 데이터를 구성하는 방법이 많이 다릅니다. 학습 데이터를 구성할 때 (쿼리, 문서쌍) 형태로 만듭니다.

```
x1: q1, (d1, d2)
x2: q2, (d3, d4)
x3: q2, (d3, d5)
x4: q2, (d4, d5)
```

이렇게 데이터를 구성하게 되면 각 데이터 인스턴스에서 두 문서의 연관 정도를 비교하는 **새로운 Pairwise binary label**을 얻을 수 있습니다. 예를 들어 첫 번째 쿼리에서 문서 $d_1$의 연관 정도는 $y_1 = 0$ (전혀 관계 없음), 문서 $d_2$의 연관 정도는 $y_2 = 3$ (연관성이 높음)일 때 $x_1$의 label은 $y_1 < y_2$ 가 됩니다. 이 경우 완전히 binary classification 문제가 됩니다.

Pairwise approach에 대한 가장 유명한 알고리즘인 **RankNet** [2] 에 기반을 두고 설명해보겠습니다. 학습은 여전히 Pointwise function $f(q, d_i) = s_i$에 대해 이루어지는데, 다음과 같이 확률적인 방법을 사용할 수 있습니다.

$$
Pr(i \succ j) \equiv \frac{1}{1 + exp^{-(s_i - s_j)}}
$$

문서 $i$가 문서 $j$보다 더 연관성이 높다면 스코어링 함수는 $f(q, d_i) = s_i > s_j = f(q, d_j)$이므로 $Pr(i \succ j)$는 1에 가까운 값이 됩니다. 학습 시에는 이 문서들의 연관성 순서를 최대한 맞추는 것을 목표로 합니다. 이를 위한 손실 함수는 우리가 익히 아는 Logloss 형태가 됩니다. $Pr(i \succ j)$를 $P_{ij}$ 로 두고 $\bar{P}_{ij}$를 실제 label이라고 할 때 손실 함수는 아래와 같습니다.

$$
L(i, j) = - \bar{P}_{ij} \log(P_{ij}) - (1 - \bar{P}_{ij})\log(1 - P_{ij})
$$

Pairwise approach는 Pointwise approach에 비해서 많은 장점이 있습니다. 우선 모델이 직접 순위를 학습합니다. 이론적으로 일반적인 ranking task 성능에 근사(approximate)합니다. 또한 명시적인 pointwise label이 필요가 없습니다. Explicit한 정보가 없더라도 두 문서의 비교를 통해서 새로운 label을 만들 수 있습니다. 하지만 근본적으로 실제 순위를 직접 최적화하지 않는다는 문제가 있습니다.

### Listwise

위에서 언급한 순위를 직접 최적화하지 못하는 문제를 해결하기 위해 등장한 Listwise approach는 LTR 알고리즘 중에서 가장 뛰어난 성능을 보이지만 가장 복잡한 접근법입니다. 이름에서 알 수 있듯이 전체 목록의 순위를 최적화합니다. 

데이터의 경우 각 쿼리에 매칭된 문서 목록을 그대로 사용합니다.

```
x1: q1, (d1, d2)
x2: q2, (d3, d4, d5)
```

Listwise approach로 문제를 해결할 때 **LambdaRank** [3] 라는 알고리즘을 많이 사용하는데 이 알고리즘은 일반적인 ranking loss 중 하나인 **Normalized Discounted Cumulative Gain (NDCG)**를 직접 최적화합니다. NDCG와 같은 ranking loss들은 불연속(discontinuous)하여 그라디언트를 정의할 수 없는데  LambdaRank는 NDCG를 최적화할 수 있는 그라디언트를 설계하여 학습을 합니다. 이 알고리즘을 Gradient Boosting에 맞춘 LambdaMART 라는 알고리즘은 LightGBM에서도 제공하고 있습니다. 이외에도 ListMLE [4], ListNet [5] 등 다양한 알고리즘이 있습니다. 각각의 알고리즘에 대한 설명은 나중에 기회가 되면 포스팅하겠습니다.

## Evaluation of LTR Models

LTR 모델을 평가할 때 일반적으로 사용하는 메트릭(metric)은 **Mean Average Precision (MAP)**와 **Normalized Discounted Cumulative Gain (NDCG)** 입니다. MAP는 문서의 연관성 여부 (binary relevance)만을 따집니다. NDCG는 연관성의 수준 (graded relevance)을 고려합니다.

### Mean Average Precision (MAP)

MAP를 계산하기 위해서는 우선 *Precision at $k$*를 계산해야 합니다. 쿼리 $q$가 주어졌을 때 모든 $k$ 개의 아이템에 대해 예측값 $r_i$이라고 할 때 $P@k$는 다음과 같습니다.

$$
P@k(q) = \frac{1}{k} \sum^k_{i=1} r_i
$$

여기에서 *Average Precision at $k$*를 계산해야 합니다.

$$
AP(q)@k = \frac{1}{\sum^k_{i=1} r_i} \sum^k_{i=1} P@i(q) \times r_i
$$

이제 여기에 쿼리 개수로 나눠 *Mean Average Precision at $k$*를 계산하면 됩니다.

$$
MAP@k = \frac{1}{Q} \sum^Q_{q=1} AP(q)@k
$$

<center>
  <figure>
    <img src="/assets/images/2022-02-13-learning-to-rank/queries.png" alt="Example" style="zoom:25%;" />
    <figcaption style="text-align: center;">An Example for evaluating LTR models</figcaption>
  </figure>
</center>

위와 같이 두 개의 쿼리가 있다고 가정해봅시다. 색이 칠해져있는 문서가 연관이 있는 경우입니다. 문서 목록 밑에 쓰여져 있는 Precision은 $P@k(q)$ 입니다. 

-   첫 번째 쿼리에 대한 $AP(q)@10$은 $AP(q_1) = (1.0 + 0.67 + 0.5 + 0.44 + 0.5) / 5 = 0.62$
-   두 번째 쿼리에 대한 $AP(q)@10$은 $AP(q_2) = (0.5 + 0.4 + 0.43) / 3 = 0.44$

두 쿼리의 평균이 바로 $MAP@10$ 이 됩니다.

$$
MAP@10 = (AP(q_1) + AP(q_2)) / 2 = (0.62 + 0.44) / 2 = 0.53.
$$

MAP는 순서에 매우 민감한 메트릭입니다. 연관 없는 문서의 위치에 페널티를 주는 개념입니다. 하지만 연관성 여부만 따지기 때문에 실제로 연관성 수준에 대해선 감안하지 않습니다.

### Normalized Discounted Cumulative Gain (NDCG)

NDCG를 계산하기 위해서는 *Discounted Cumulative Gain at $k$*를 계산해야 합니다.  $i$ 번째 문서의 연관성 수준을 $l_i$라고 할 때 $DCG@k$는 다음과 같습니다.

$$
DCG@k = \sum_{i=1}^k\frac{2^{l_i} - 1}{\log_2(i + 1)}
$$

연관성 수준 $l_i$를 연관 여부로 단순하게 생각하면 분자는 연관성이 있을 때는 $1$, 없을 때는 $0$ 입니다.

이 값을 $0$ 에서 $1$로 스케일링하기 위해서 가장 최적의 순서인 경우의 $DCG$인 $IDCG@k$를 계산하여 $DCG@k$를 나눈 것이 $NDCG@k$ 입니다. 

$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

위 이미지로 $NDCG@k$를 계산해 본다고 하면 연관이 있는 경우는 $l_i = 1$, 연관이 없는 경우는 $l_i = 0$ 로 두고 다음과 같이 계산할 수 있습니다.

-   $IDCG@10 = \sum^k_{i=1} \frac{1}{\log_2(i+1)} = 6.55$
-   $DCG@10(q_1) = \sum_i \frac{1}{\log_2{(i+1)}} = 3.53 \quad \text{for } i \in \{ 1, 3, 6, 9, 10 \}$
-   $DCG@10(q_2) = \sum_i \frac{1}{\log_2{(i+1)}} = 1.95 \quad \text{for } i \in \{ 2, 5, 7 \}$
-   $NDCG@10 (q_1) = \frac{DCG@10(q_1)}{IDCG@10} = 0.54$
-   $NDCG@10 (q_2) = \frac{DCG@10(q_21.)}{IDCG@10} = 0.30$



## References

[1] [Learning to (Retrieve and) Rank - Intuitive Overview - part III by Michele Trevisiol](https://jobandtalent.engineering/learning-to-retrieve-and-rank-intuitive-overview-part-iii-1292f4259315)

[2] Burges, Chris, et al. "Learning to rank using gradient descent." *Proceedings of the 22nd international conference on Machine learning*. 2005.

[3] Burges, Christopher JC. "From ranknet to lambdarank to lambdamart: An overview." *Learning* 11.23-581 (2010): 81.

[4] Lan, Yanyan, et al. "Position-Aware ListMLE: A Sequential Learning Process for Ranking." *UAI*. 2014.

[5] Cao, Zhe, et al. "Learning to rank: from pairwise approach to listwise approach." *Proceedings of the 24th international conference on Machine learning*. 2007.

