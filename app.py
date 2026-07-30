import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="LOTTO GPT V25.1",
    layout="wide"
)


st.title("🎯 LOTTO GPT V25.1")
st.subheader("AI 로또 데이터 분석 엔진")


# =========================
# 엑셀 업로드
# =========================

st.sidebar.header("📂 데이터 입력")


uploaded_file = st.sidebar.file_uploader(
    "로또 회차 엑셀 업로드",
    type=["xlsx"]
)


if uploaded_file:

    df = pd.read_excel(uploaded_file)


    st.success("엑셀 데이터 로딩 완료")


    st.write("데이터 미리보기")

    st.dataframe(
        df.head(),
        use_container_width=True
    )


    st.divider()


    # =========================
    # 번호 빈도 분석
    # =========================


    st.subheader("📊 번호 출현 빈도 분석")


    numbers = []


    for col in df.columns:

        if "번호" in str(col):

            numbers.extend(
                df[col].dropna().astype(int).tolist()
            )


    freq = pd.Series(numbers).value_counts()


    result = pd.DataFrame(
        {
            "번호":freq.index,
            "출현횟수":freq.values
        }
    )


    st.bar_chart(
        result,
        x="번호",
        y="출현횟수"
    )


    st.subheader("🔥 TOP 10 빈도 번호")


    top10 = result.head(10)


    st.dataframe(top10)


else:

    st.info(
        "왼쪽 메뉴에서 로또 분석용 엑셀 파일을 업로드하세요."
    )
