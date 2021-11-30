---
title: 데코레이터를 이용한 간단한 로깅하기
tags: [Decorator, Logging, Python]
category: Python
aside:
  toc: true
show_category: true
---

데코레이터를 이용해서 테스트를 대체할 간단한 로깅을 해보겠습니다. 🪵

<!--more-->

## 🏞 배경

개인적으로 추천 시스템은 다른 ML 애플리케이션보다 도메인에서 요구하는 비즈니스 로직의 양이 더 많다고 생각합니다. 그러다보니 예외로 두어야할 것들도 더 많고, 테스트해야 하는 것들도 더 많습니다. 지금 제가 개발해서 사내에서 운영하고 있는 학습과정 추천 시스템도 마찬가지였습니다.  많은 조건들이 있었습니다. 추천이 제공되어야 할 사용자에 대한 필터링이 있어야 하고, 추천 콘텐츠에 대한 정렬 조건이 있어야 하고, 권장되는 추천 콘텐츠의 개수 등.

엄격하게 지켜져야 하는 조건들은 당연히 운영 시스템에 올라가기 전에 테스트 되어야 합니다. 예를 들어 추천 결과에 포함되면 안되는 것들이 있는지 확인하거나, 사용자가 올바르게 필터링되지 않았는지 말이죠. 이런 경우는 조건을 만족하지 못하는 경우 빌드가 안되게 하거나 오류를 발생시켜 수정을 해야 합니다. 하지만 권장되는 추천 콘텐츠의 개수가 지켜지지 않은 경우에 대해서까지 엄격하게 할 필요는 없었습니다. 확인 후에 조치를 하면 되는 부분이니까요. 모델 운영 중 발생하는 로그에 `WARNING` 수준으로 기록하기만 하면 되었습니다. 

이런 경우 방법은 여러 가지일겁니다. 단순히 추천 콘텐츠 개수가 권장하는 개수와 다른 경우에만 로깅을 하는 것도 방법입니다. 이번 포스트에서는 데코레이터를 이용하여 추상화한 로깅에 대해 이야기 해보려 합니다.

## 📝  데코레이터 사용하기

### 개요

각각의 추천 알고리즘은 클래스로 구성되어 있습니다. Scikit-learn의 인터페이스를 따라하고자 `fit()`, `generate()` 메서드를 갖고 있습니다. 각각은 데이터에 맞춰 학습하고, 추천 리스트를 생성합니다. 결국 다음의 구조로 데코레이터를 작성하고자 합니다.

```python
logger = logging.getLogger()

class Recommender():
	def __init__(self, **kwargs):
		...
	
	def fit(self, X):
		return self._fit(X)

	@check_recommendation_validity
	def generate(self, **kwargs):
		return self._generate(**kwargs)

	...
```

### 테스트 요소

로깅을 통해 확인해야 할 것은 '사용자에게 보여지는 추천 콘텐츠의 수가 사전에 정한 개수와 동일한가'와 '추천을 제공해야 하는 사용자에게 모두 추천이 제공되는가' 입니다. 

우선 첫 번째 테스트 요소는 추천 결과 내의 사용자 수와 전체 추천 목록의 길이를 사전에 정한 개수로 나눈 것과 같은지를 확인하면 됩니다. 두 번째 요소는 실제 사용자 정보 내의 사용자 수와 추천 결과 내의 사용자 수가 같은지 확인하면 됩니다.

```python
def _check_wrong_recom_size(recom_list):
	users_get_recommendation = recom_list[USER_ID].nunique()
	avg_users_over_recom_size = int(len(recom_list) / RECOM_SIZE)

	return users_get_recommendation != avg_users_over_recom_size

def _check_every_user_get_recommendation(user_info, recom_list):
	num_users = len(user_info)
	users_get_recommendation = recom_list[USER_ID].nunique()

	return num_users == users_get_recommendation
```

### 데코레이터 작성

데코레이터를 작성해야 합니다. 데코레이터에 대한 자세한 설명은 본 포스트에서 다루지 않겠습니다. 다음 [링크](https://wikidocs.net/83687)를 확인해주세요.

위에서 작성한 두 개의 함수를 포함하여 작성을 할텐데 알아두어야 할 점이 있습니다. 일단 사용할 데코레이터는 클래스 내에서 사용합니다. 따라서 데코레이터 내의 Wrapper 함수에 `self`를 인자로 받으면 클래스 내의 인스턴스에 접근할 수 있습니다. 현재 시스템에서 사용자 정보의 경우 클래스 내의 인스턴스라 `self`를 추가해주도록 하겠습니다.

Wrapper 함수의 인자에 들어갈 메서드는 추천 결과를 생성하는 메서드입니다. `generate()` 메서드가 사용될 것을 감안하여 작성합니다.

```python
def check_recommendation_validity(generator) -> Callable:
    @functools.wraps(generator)
    def wrapper(self, *args, **kwargs):
        recom_list = generator(self, *args, **kwargs)
    
        if _check_wrong_recom_size(recom_list):
            msg = "There are too many or too few recommendations for each user."
            logger.warning(msg)
        
        if not _check_every_user_get_recommendation(
            user_info=self._user_info, recom_list=recom_list
        ):
            msg = (
                "The user coverage is not 100%."
                "There are users cannot get recommendation."
            )
            logger.warning(msg)
        return recom_list
    return wrapper

```

### 결과 확인

이렇게 코드를 다 작성하고 나서 추천 리스트를 생성합니다. 데코레이터가 올바르게 작성하는지 확인하기 위해서 일부러 추천 결과를 일부만 반환하게 해봤습니다. 로깅이 올바르게 되었습니다.

![](/assets/images/2021-11-30-simple-logging-using-decorator/result.png)