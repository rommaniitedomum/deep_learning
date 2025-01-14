import json

import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model("model/abalone_model.h5")

# 입력 데이터 받기
# input_data = json.loads(sys.argv[1])
input_data = [0.440, 0.365, 0.125, 0.5160, 0.1140, 0.155, 10]
# data_array = input_data["data"]
test_data = np.array(input_data).reshape(1, -1)

prediction = model.predict(test_data).tolist()

# 예측 결과 출력
print(json.dumps({"prediction": prediction[0][0]}))
