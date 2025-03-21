# 추천 시스템
## 추천 시스템 개요
사용자의 취향을 이해하고 맞춤 상품과 콘텐츠를 제공해 조금이라도 사이트에 오래 머무르게 하기 위한 수단 중 하나이다.
많은 서비스가 추천 시스템을 도입함으로써 매출을 큰 폭으로 증가시켰다.
많은 데이터가 추천 시스템에 축적되면서 추천이 더욱 정확해지고 다양한 결과를 얻을 수 있는 좋은 선순환 시스템을 구축할 수 있게 된다.
데이터 기반의 추천 시스템은 오늘날 필수 요소인 셈이다.

## 추천 시스템의 유형
1. 콘텐츠 기반 필터링(Contents based filtering)

사용자가 특정한 아이템을 매우 선호하는 경우, 그 아이템과 비슷한 콘텐츠를 가진 다른 아이템을 추천하는 방식이다. 예를 들어 특정 영화에 높은 평점을 주었다면 그 영화의 장르, 배우, 감독, 키워드 등의 유사한 다른 영화를 추천 해주는 방식이다.

2. 협업 필터링(Collaborative filtering)

협업 필터링 방식은 두 방식 모두 사용자-아이템 행렬 데이터에 의지해 추천을 수행한다. 행(Row)에는 사용자(User), 열(Column)에는 개별 아이템(Entity)으로(또는 그 역행렬의 형태로) 구성된다.
    
 - 최근접 이웃 협업 필터링(Nearest neighbor collaborative filtering)

    최근접 이웃 협업 필터링(또는 메모리 협업 필터링 이라 함)은 친구들에게 물어보는 것과 유사한 방식으로, 사용자가 아이템에 매긴 평점 정보나 상품 구매 이력과 같은 사용자 행동 양식(User behavior)만을 기반으로 추천을 수행하는 방식이다. 대표적으로 amazon에서 사용 중이다. 일반적으로 사용자 기반과 아이템 기반으로 다시 나눌 수 있다.
    - 사용자 기반(User-User): 당신과 비슷한 고객들이 다음 상품도 구매했습니다.
    - 아이템 기반(Item-Item): 이 상품을 선택한 다른 고객들은 다음 상품도 구매했습니다.
    
    아이템 기반 최근접 이웃 방식은 아이템 간의 유사도와는 상관 없이 사용자들이 그 아이템을 좋아하는지 선호하는지의 평가 척도가 유사한 아이템을 추천하는 알고리즘이다. 일반적으로 사용자 기반보다는 아이템 기반 협업 필터링이 정확도가 더 높다.
    
- 잠재 요인 협업 필터링(Latent factor collaborative filtering)
    
    사용자-아이템 평점 매트릭스 속제 숨어 있는 잠재 요인을 추출해 추천 예측을 할 수 있게 하는 기법이다. 대규모 다차원 행렬을 차원 감소 기법으로 분해하는 과정에서 잠재 요인을 추출하는 방식이 사용된다. '잠재 요인'이 정확히 어떤 것인지는 알 수 없지만, 가령 영화 평점 기반의 사용자-아이템 평점 행렬 데이터라면 장르별 선호도로 가정할 수도 있다.

> 저차원 매트릭스로 분해하는 기법으로서 대표적으로 SVD(Singular vector decomposition), NMF(Non-negative matrix factorization) 등이 있습니다.



추천 시스템은 서비스하는 아이템의 특성에 따라 결정된다.