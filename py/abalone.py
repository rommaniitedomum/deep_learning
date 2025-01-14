import koreanize_matplotlib  # 한글 지원
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Streamlit 제목
st.title("Streamlit: H5 모델로 예측하기 - 실제값과 예측값 비교")

# Sidebar에 모델 로드 상태 표시
MODEL_PATH = "model/abalone_model.h5"  # 모델 경로
try:
    model = load_model(MODEL_PATH)
    st.sidebar.success(f" 모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    st.sidebar.error(f" 모델 로드 실패: {str(e)}")

# 사용자 입력 섹션
st.header("실제값 입력")
st.write("아래 필드를 사용해 여러 개의 실제값을 입력하세요.")

# Form을 사용한 실제값 입력
with st.form("input_form"):
    # 다중 입력 필드
    actual_values = st.text_area(
        "실제값 입력 (쉼표로 구분)", "0.440, 0.365, 0.125, 0.5160, 0.1140, 0.155, 10"
    )
    submitted = st.form_submit_button("예측 및 비교")

# 예측 및 결과 비교
if submitted:
    # 실제값 파싱
    try:
        actual_values = [float(val.strip()) for val in actual_values.split(",")]

        if len(actual_values) != 7:
            st.error(f"값을 7개 입력하세요.{len(actual_values)}")
            st.stop()

    except ValueError:
        st.error("실제값 입력이 잘못되었습니다. 숫자를 쉼표로 구분하여 입력하세요.")
        st.stop()

    # 입력 데이터 생성 (랜덤 입력 데이터 생성)
    num_samples = len(actual_values)
    sample_input = pd.DataFrame(
        {
            "길이": np.random.rand(num_samples),
            "직경": np.random.rand(num_samples),
            "두께": np.random.rand(num_samples),
            "전체무게": np.random.rand(num_samples),
            "내장무게": np.random.rand(num_samples),
            "껍질무게": np.random.rand(num_samples),
            "나이테": np.random.rand(num_samples),
        }
    )

    # 모델 예측
    try:
        predictions = model.predict(sample_input.values).flatten()

        # 실제값과 예측값 비교 DataFrame 생성
        comparison_df = pd.DataFrame({"실제값": actual_values, "예측값": predictions})

        # 결과 출력
        st.subheader("결과 비교")
        st.dataframe(comparison_df)

        # 상위 10개 결과 출력
        st.write("상위 10개 결과:")
        for i in range(min(10, len(actual_values))):
            st.write(f"실제값: {actual_values[i]:.3f}, 예측값: {predictions[i]:.3f}")

        # 차트로 시각화
        st.subheader("실제값과 예측값 시각화")
        koreanize_matplotlib.koreanize()
        st.bar_chart(comparison_df.head(10).set_index("실제값"))

    except Exception as e:
        st.error(f"예측 중 오류 발생: {str(e)}")
