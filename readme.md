# 딥러닝 정리들:#

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

# 연관분석#

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
가장 높은 확률을 가진 클래스의 인덱스를 가져옵니다.
예: 앞에 숫자가 0.8 -> 0, 뒤의 숫자가 0.8 -> 1

predicted_class:
예측된 클래스의 이름.

# Numpy axis 정리

- axis 0 = rows
- axis 1 = cols

# Pandas axis 정리

- axis 0 = cols
- axis 1 = rows

input dim = 입력 뉴런

파라미터 계산 = 가중치(입력 뉴런 \* 출력 뉴런(편향)) + 편향
