# 딥러닝 정리들

keras.dataset 로드함. 각 이미지는 28\*28 흑백 이미지로 구성

    - 총 구성 -> 6만개의 28*28 이미지들

    레이블 0~9 정수로 각 클래스를 나타냄

    정규화 구조 :
    train_scaled = train_input / 255.0


    train_scaled = train_scaled.reshape(-1, 28*28)

    - 2D 이미지28*28= > 1D 784 벡터 로 변경

    reshape 옵션 설명
    reshape (rows(행방향),cols(열방향))
    reshape(-1(자동계산))

    train_scaled = train_scaled.reshape(-1, 28 * 28)
    ex: 전체 크기를 (-1) 28*28을 평탄화 784 백터로 만듬

    reshape 를 사용한 flattening
    ex:
    image = np.random.randint(0, 255, (28, 28))  # 28x28 이미지
    flattened = image.reshape(-1)  # 1D 벡터로 변환

    : 결과 = 1D 이미지

    vector = np.arange(784)  # 길이 784의 벡터
    image = vector.reshape(28, 28)  # 28x28 2D 이미지로 변환

    : 결과 = 2D 이미지

    data = np.arange(100).reshape(20, 5)  # 20개의 샘플, 각 샘플 5개 특징
    batch_data = data.reshape(5, 4, 5)  # 배치 크기 5, 각 배치에 4개의 샘플

    : 결과 = 5x4x5 3D 배치

# 연관분석

    연관성(corr) 파악하는 행위

    지지도(support) : A , B 교집합
    신뢰도 (confidence) : A,B 교집합 나누기 A
    향상도(Lift) : A,b 교집합 나누기 A*B

    Apriori 사용 (항목 집합 찾기)

    사용데이터: Groceries_dataset.csv (특징: int64(1)object(2))

    문제: itemDescription 안의 아이템이 문자임 :
        -> 해결 : get_dummies() 를 사용해 0,1 수치형 데이터로 변환
        ; 날짜별 고객별 거래 데이터를 합산하여 연관 분석에 사용


    itemDescription 의 항목 파악 : Unique() 함수로 파악
    예시 : products = df['itemDescription'].unique()




    -> 하고자 하는것 :
    (원 데이터)
    customerID       date itemDescription

    0 1 2025-01-01 Milk
    1 2 2025-01-01 Bread
    2 3 2025-01-02 Butter
    3 4 2025-01-03 Milk

    (원핫 인코딩 상태)

    Bread Butter Milk
    0 0 0 1
    1 1 0 0
    2 0 1 0
    3 0 0 1

    (결합된 최종 데이터)

    customerID date Bread Butter Milk
    0 1 2025-01-01 0 0 1
    1 2 2025-01-01 1 0 0
    2 3 2025-01-02 0 1 0
    3 4 2025-01-03 0 0 1

# FP-Growth 알고리즘을 사용한 반발항목 집합 계산

    예시 : transaction2 데이터
    상품A 상품B 상품C
    0.067767 0.157923 0.049054
    0.251354 0.364681 0.215989
    0.000000 0.000000 0.000000

- FP-Growth 적용

  frequent_itemsets = fpgrowth(transaction2, min_support=0.01, max_len=3, use_colnames=True)
  frequent_itemsets.sort_values(by=['support'], ascending=True).head(10)

지지도란? 특정 집합이 거래에 얼마나 나오는지 비율
// 설명 : min_support= 지지도 최소 , max_lens=3 = 아이템이 최소 3개 있어야함 , 컬럼네임 = 컬럼네임
//
// 정렬 방법: support(정렬 columns)기반으로 ascending : 오름차순 정리(10개 출력 )

간단한 예제들 :

- metric='support': 데이터에서 가장 빈도가 높은 규칙을 찾고 싶을 때.
- metric='confidence': 특정 조건에서 결과가 발생할 가능성을 보고 싶을 때.
- metric='lift': 규칙의 상관관계를 깊이 분석하고 싶을 때.
- antecedents: 선행 항목 , consequents : 후행 항목
- ; 즉
  antecedents (선행 항목)
  연관 규칙의 "If" 조건에 해당합니다.
  선행 항목이 발생하면, 후행 항목이 발생할 가능성을 측정합니다.
  예: "빵을 구매한 사람".

consequents (후행 항목)
연관 규칙의 "Then" 조건에 해당합니다.
선행 항목이 발생한 경우 발생 가능성을 측정하는 항목.
예: "우유도 구매했다".

## 전체 평가

- 요거트와 소시지를 동시에 구매한 후 우유를 구매할 확률이 높다.
- 또한 소시지와 빵을 동시에 구매한 후 우유를 구매할 확률이 높다.
- 따라서 고객의 장바구니에 요거트와 소시지가 담겨 있다면 우유를 추천해주고, 상품이 진열된 굿을 쉽게 찾을 수 있도록 유도한다.
- 빵을 구매한 후 가공 치즈를 구매할 가능성이 가장 높고, 빵을 구매한 후 적포도주를 구매할 확률이 높다.
- 따라서 빵과 치즈, 와인을 가까이 진열한다.

# 합성곱 신경망(사과)

빅데이터 : 이미지 확인 1과 0 으로

1. 썩은 사과 테스트: 1/0 으로 구분

- 데이터 모델 과정
  model = Sequential()
  model.add(vgg)
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(2, activation='softmax'))

test_set.class_indices : 신선 = 0 썩은 = 1

예측값 : print(np.argmax(predictions, axis=1)) # 앞에 숫자가 0.8 -> 0, 뒤의 숫자가 0.8 -> 1

np.expand_dims(img, axis=0):
모델 입력으로 사용하기 위해 배치 차원을 추가합니다.
입력 크기가 (64, 64, 3)인 이미지를 (1, 64, 64, 3)으로 변경.

model.predict():
모델이 이미지를 예측합니다. 출력은 각 클래스에 대한 확률 분포입니다.
예: [0.1, 0.7, 0.2].

np.argmax(prediction):
가장 높은 확률을 가진 클래스의 인덱스를 배열로 가저옵니다.
예: 앞에 숫자가 0.8 -> 0, 뒤의 숫자가 0.8 -> 1

predicted_class:
예측된 클래스의 이름.

# Numpy axis 정리

- axis 0 = rows
- axis 1 = cols

# Pandas axis 정리

- axis 0 = cols
- axis 1 = rows

input dims = 입력 뉴런

파라미터 계산 = 가중치(입력 뉴런\*출력 뉴런(편향)) + 편향

coef*
intercept*
regression\*

## 선형, 절편 vs 회귀 , 절편

- 선형 : 일차 방정식(직선){기하학 관계 설명}

- 회귀 : 미지의 값을 학습-> 최적의 직선을 찾음 { 예측 모델}

- 선형 절편 vs 회귀 절편

선형 절편: 그냥 y 축

회귀 절편: 모든 입력 특성이 0 일떄 모델 예측 값
: 데이터의 영향을 받음 , 초기상태를 조정하는 역할임

Sequential 데이터 처리 단계

1. 데이터 준비
2. 데이터 잔처리(결측치처리, 정규화/스케일링, 슬라이싱)
3. 데이터 분할
4. 모델설계
5. 모델예측
6. 데이터를 Sequential 데이터로 변환하고 처리
7. 결과 시각화

# 지도 데이터를 활용한 군집 분석

- 비지도 학습과 군집의 개념 사용.
- 군집를 이용한 머신러닝 알고리즘 중 k-Means 이용.

- 비지도 학습 : 문제는 알려주되 정답(레이블)은 알려주지 않음
  (여러 데이터를 학습함으로써 미처 알지 못했던 것을 발견하는 데 목적)
- 군집 : 유사도가 높은 집단끼리 그룹을 만들고 분류된 그룹 간 특징을 파악하는 분석 방법
- K-means : 군집의 개수를 정하는 알고리즘 (가장 많이 사용)

      # 넘파이 배열로 만들고 의미 없는 일련번호를 뺸 위도와 경도만 불러서 X 에 저장

      XY = np.array(df)

      X = XY[:, 1:] # 모든 rows 선택 1번 cols 제외
      X[:10]

K 평균 알고리즘의 엘보우 기법을 이용하여 최적 군집점 개수 수집
KMeans() 메서드를 사용하고, fit() 로 X 데이터 학습
KMeans() 클러스터 내 오차 제곱합 인 sse.append()의 inertia\_ 로 시각화

( 예시 그래프)
택배 수치 데이터가 저장된 X 에서 군집점을 랜덤으로 골라서 C_x, C_y 변수에 저장

- 초기 설정하기
- k-Means의 과정은 모든 데이터의 거리를 계산하여 가까운 군집에 그룹을 할당하고 각 군집에 속한 데이터의 평균을
- 계산하여 군집의 중심을 갱신하기 때문에 거리를 구하는 함수를 설정
- 거리 함수는 유클리드 거리 공식을 이용:

      def distance(A,B):
      return np.sqrt(np.sum(np.power((A-B),2))) # (스꼴라 제곱) 제곱

      # [ [(A-B)2제곱 ] 제곱근

      - 예시
      - value = 4
      - np.sqrt(value)
      - 2.0

      - np.power(4,2) 4 제곱
      - p.power(27, 1/3) 27의 세제곱근

# 탈모 데이터를 이용한 K-NN 분석

    데이터 정규화 (df.columns 로 먼저 확인)

    문자로 이루어진 데이터를 숫자로 라벨링한다.
    replace() 함수로 각 문자를 0, 1, 2의 숫자로 대체한다.(데이터 수치화)
    비듬 정도, 머리숱, 수영 여부, 머리 감기 여부 : 숫자 라벨링
    남은 두피압과 스트레스 정도는 한 번에 라벨링

    ex: df["dandruff"] = df["dandruff"].replace({"None": 0, "Few": 1, "Many": 2})

    데이터 분리 후 K-NN 모델 피팅 후 적용 ->

    # K-NN모델 실패적 -> 원-핫 인코딩 사용

    ex: 탈모정도를 4가지 척도로 나눠서 백터안에 담는법->라이브러리 이름이 to_categorical 인 이유
    없음	[1, 0, 0, 0]
    경미함	[0, 1, 0, 0]
    중간 정도	[0, 0, 1, 0]
    심각함	[0, 0, 0, 1]

    -> y(타겟) 데이터를 원 핫 인코드 -> asarray 사용해서 배열객체변환
    -> 재 피팅 후 모델 테스트 ( 딥러닝모델 1입력 2은닉 1 출력 사용)
    사용 활성화 함수: 렐루 -> 소프트 맥스
    모델 컴파일 : 손실율 측정: categorical_crossentropy , 옵티마이저: adam
    (학습속도 최적화) , 평가지표: 정확도


    값 출력: argmax 함수를 사용해서( 가장 큰값 반환) 가장 높은 클래스값 반환
