import streamlit as st
import numpy as np
import random

st.set_page_config(
    page_title="LOTTO GPT V25.0",
    layout="wide"
)

st.title("🎯 LOTTO GPT V25.0")
st.write("AI 가중치 기반 로또 분석 테스트")

st.sidebar.header("가설 가중치")

weights = []

for i in range(1, 12):

    w = st.sidebar.slider(

        f"가설 {i}",

        0,

        100,

        50

    )

    weights.append(w)


def generate_numbers(weights):

    prob = np.array(weights)

    seed = sum(weights)

    np.random.seed(seed)

    numbers = sorted(

        random.sample(

            range(1, 46),

            6

        )

    )

    return numbers


if st.button("번호 생성"):

    nums = generate_numbers(weights)

    cols = st.columns(6)

    for i, n in enumerate(nums):

        cols[i].metric(

            label=f"No.{i+1}",

            value=str(n)

        )

    st.success("생성 완료!")
