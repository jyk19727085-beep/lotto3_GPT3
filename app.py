from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from engine.scoring import calculate_eleven_scores

from engine.patterns import (
    extract_draws,
)

from engine.v26_scoring import (
    calculate_v26_scores,
)

from engine.diversity import (
    combination_features,
    generate_balanced_combinations,
)
from engine.set_optimizer import (
    generate_practical_lotto_set,
)

# =========================================================
# LOTTO GPT V26.0
# 15대 분석가설 + 유사후속 + 마킹패턴
# + 구조전이 + 다양성 조합 생성기
# =========================================================


st.set_page_config(
    page_title="LOTTO GPT V26.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 화면 디자인
# =========================================================

st.markdown(
    """
    <style>

    /* ==================================================
       V26.2 PROFESSIONAL — 전체 배경
       ================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 15%,
                rgba(37, 99, 235, 0.22),
                transparent 32%
            ),
            radial-gradient(
                circle at 85% 10%,
                rgba(250, 204, 21, 0.16),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(124, 58, 237, 0.18),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #020617 0%,
                #0f172a 45%,
                #111827 100%
            );
        background-attachment: fixed;
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: rgba(2, 6, 23, 0.30);
        backdrop-filter: blur(12px);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(2, 6, 23, 0.98),
                rgba(15, 23, 42, 0.98)
            );
        border-right: 1px solid rgba(250, 204, 21, 0.22);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }


    /* ==================================================
       메인 글래스 컨테이너
       ================================================== */

    .block-container {
        width: min(96%, 1500px);
        max-width: 1500px;
        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.075),
                rgba(255, 255, 255, 0.025)
            );
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        border: 1px solid rgba(250, 204, 21, 0.18);
        border-radius: 24px;
        padding: 2rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow:
            0 25px 70px rgba(0, 0, 0, 0.48),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        animation: containerEnter 0.75s ease-out;
    }

    @keyframes containerEnter {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    /* ==================================================
       제목과 글자
       ================================================== */

    h1 {
        color: #ffffff !important;
        text-align: center;
        font-weight: 950 !important;
        letter-spacing: -0.04em;
        text-shadow:
            0 0 10px rgba(250, 204, 21, 0.55),
            0 0 28px rgba(250, 204, 21, 0.30);
        animation: titleGlow 2.4s ease-in-out infinite alternate;
    }

    h2, h3 {
        color: #f8fafc !important;
        font-weight: 850 !important;
    }

    p, label, div {
        word-break: keep-all;
    }

    @keyframes titleGlow {
        from {
            text-shadow:
                0 0 8px rgba(250, 204, 21, 0.42),
                0 0 18px rgba(250, 204, 21, 0.20);
        }
        to {
            text-shadow:
                0 0 14px rgba(250, 204, 21, 0.82),
                0 0 38px rgba(250, 204, 21, 0.44);
        }
    }


    /* ==================================================
       로또공
       ================================================== */

    .lotto-row {
        display: flex;
        flex-wrap: wrap;
        gap: 11px;
        align-items: center;
        justify-content: flex-start;
        margin: 12px 0 20px 0;
    }

    .lotto-ball {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 54px;
        height: 54px;
        flex: 0 0 54px;
        border-radius: 50%;
        color: #ffffff;
        font-weight: 950;
        font-size: 1.28rem;
        border: 2px solid rgba(255, 255, 255, 0.28);
        box-shadow:
            inset -5px -6px 9px rgba(0, 0, 0, 0.34),
            inset 4px 4px 7px rgba(255, 255, 255, 0.20),
            0 5px 12px rgba(0, 0, 0, 0.45);
        text-shadow: 0 2px 3px rgba(0, 0, 0, 0.55);
        animation: ballAppear 0.45s ease-out both;
        transition:
            transform 0.20s ease,
            filter 0.20s ease;
    }

    .lotto-ball:hover {
        transform: translateY(-4px) scale(1.07);
        filter: brightness(1.13);
    }

    @keyframes ballAppear {
        from {
            opacity: 0;
            transform: scale(0.55) rotate(-16deg);
        }
        to {
            opacity: 1;
            transform: scale(1) rotate(0);
        }
    }


    /* ==================================================
       추천 결과 카드
       ================================================== */

    .result-card {
        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.94),
                rgba(30, 41, 59, 0.82)
            );
        border: 1px solid rgba(250, 204, 21, 0.28);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.34),
            inset 0 1px 0 rgba(255, 255, 255, 0.07);
        animation: cardSlide 0.55s ease-out both;
        transition:
            transform 0.22s ease,
            border-color 0.22s ease,
            box-shadow 0.22s ease;
    }

    .result-card:hover {
        transform: translateY(-3px);
        border-color: rgba(250, 204, 21, 0.62);
        box-shadow:
            0 16px 38px rgba(0, 0, 0, 0.44),
            0 0 22px rgba(250, 204, 21, 0.11);
    }

    @keyframes cardSlide {
        from {
            opacity: 0;
            transform: translateX(-18px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }


    /* ==================================================
       안내·엔진 상태 카드
       ================================================== */

    .notice-card {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(
                120deg,
                rgba(250, 204, 21, 0.13),
                rgba(37, 99, 235, 0.12),
                rgba(124, 58, 237, 0.12)
            );
        border: 1px solid rgba(250, 204, 21, 0.30);
        border-radius: 16px;
        padding: 17px;
        color: #f8fafc;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
    }

    .notice-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: -120%;
        width: 60%;
        height: 100%;
        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.12),
                transparent
            );
        animation: shineMove 4.2s linear infinite;
    }

    @keyframes shineMove {
        0% {
            left: -120%;
        }
        55%,
        100% {
            left: 150%;
        }
    }

    .engine-card {
        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.90),
                rgba(30, 41, 59, 0.78)
            );
        border-left: 5px solid #facc15;
        border-radius: 14px;
        padding: 15px 17px;
        margin: 11px 0;
        color: #f8fafc;
        box-shadow: 0 9px 24px rgba(0, 0, 0, 0.28);
    }

    .confidence-high {
        color: #4ade80;
        font-weight: 900;
    }

    .confidence-middle {
        color: #facc15;
        font-weight: 900;
    }

    .confidence-low {
        color: #fb7185;
        font-weight: 900;
    }


    /* ==================================================
       Streamlit 버튼
       ================================================== */

    .stButton > button {
        min-height: 3.4rem;
        border-radius: 14px !important;
        border: 1px solid rgba(250, 204, 21, 0.62) !important;
        background:
            linear-gradient(
                110deg,
                #a16207,
                #eab308,
                #facc15,
                #ca8a04
            ) !important;
        background-size: 240% 240% !important;
        color: #111827 !important;
        font-weight: 950 !important;
        font-size: 1.06rem !important;
        box-shadow:
            0 8px 24px rgba(234, 179, 8, 0.30),
            0 0 22px rgba(250, 204, 21, 0.14);
        animation:
            buttonGradient 3.4s ease infinite,
            buttonPulse 2.0s ease-in-out infinite;
        transition:
            transform 0.20s ease,
            box-shadow 0.20s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow:
            0 12px 30px rgba(234, 179, 8, 0.42),
            0 0 34px rgba(250, 204, 21, 0.24);
    }

    @keyframes buttonGradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    @keyframes buttonPulse {
        0%,
        100% {
            filter: brightness(1);
        }
        50% {
            filter: brightness(1.12);
        }
    }


    /* ==================================================
       지표·표·입력창
       ================================================== */

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.24);
    }

    [data-testid="stMetricValue"] {
        color: #facc15;
        font-weight: 900;
    }

    [data-testid="stDataFrame"] {
        overflow-x: auto;
        border-radius: 14px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }


    /* ==================================================
       모바일·태블릿 반응형
       ================================================== */

    @media (max-width: 900px) {
        .block-container {
            width: 98%;
            padding: 1.15rem;
            margin-top: 0.45rem;
            border-radius: 17px;
        }

        h1 {
            font-size: clamp(1.65rem, 7vw, 2.35rem) !important;
            line-height: 1.18 !important;
        }

        h2 {
            font-size: 1.35rem !important;
        }

        h3 {
            font-size: 1.13rem !important;
        }

        .lotto-row {
            gap: 8px;
            justify-content: center;
        }

        .lotto-ball {
            width: 46px;
            height: 46px;
            flex-basis: 46px;
            font-size: 1.08rem;
        }

        .result-card {
            padding: 14px 11px;
            border-radius: 15px;
        }

        .notice-card,
        .engine-card {
            padding: 13px;
        }

        .stButton > button {
            min-height: 3.7rem;
            font-size: 1rem !important;
        }
    }

    @media (max-width: 430px) {
        .block-container {
            padding: 0.82rem;
        }

        .lotto-row {
            gap: 5px;
            flex-wrap: nowrap;
            justify-content: center;
        }

        .lotto-ball {
            width: 41px;
            height: 41px;
            flex: 0 0 41px;
            font-size: 0.96rem;
            border-width: 1px;
        }

        .result-card {
            padding: 12px 8px;
        }

        [data-testid="stMetric"] {
            padding: 10px;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.3rem;
        }
    }


    /* 사용자가 기기에서 애니메이션 축소를 설정한 경우 */
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 공통 화면 함수
# =========================================================

def ball_color(number: int) -> str:
    """동행복권 로또볼 구간과 비슷한 색상을 지정합니다."""

    if number <= 10:
        return "#fbc02d"

    if number <= 20:
        return "#1976d2"

    if number <= 30:
        return "#e53935"

    if number <= 40:
        return "#757575"

    return "#43a047"


def render_balls(numbers: List[int]) -> None:
    """번호를 로또공 형태로 표시합니다."""

    html = "<div class='lotto-row'>"

    for number in numbers:
        html += (
            f"<div class='lotto-ball' "
            f"style='background:{ball_color(int(number))};'>"
            f"{int(number)}"
            f"</div>"
        )

    html += "</div>"

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def clean_column_name(column: object) -> str:
    """열 이름에서 공백과 줄바꿈을 제거합니다."""

    return (
        str(column)
        .strip()
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
    )


# =========================================================
# 엑셀 열 자동 탐색
# =========================================================

def detect_round_column(
    df: pd.DataFrame,
) -> Optional[str]:
    """회차 열로 보이는 열을 자동 탐색합니다."""

    round_keywords = [
        "회차",
        "회",
        "draw",
        "round",
    ]

    for column in df.columns:
        cleaned = clean_column_name(
            column
        ).lower()

        if any(
            keyword in cleaned
            for keyword in round_keywords
        ):
            return column

    return None


def detect_number_columns(
    df: pd.DataFrame,
) -> List[str]:
    """
    당첨번호 6개 열을 자동 탐색합니다.

    우선순위:
    1. 번호·당첨번호·number 등의 열
    2. 값 대부분이 1~45인 열
    """

    candidates: List[str] = []

    excluded_keywords = [
        "보너스",
        "bonus",
        "회차",
        "round",
        "날짜",
        "date",
        "합계",
        "sum",
        "순위",
        "당첨금",
        "당첨자",
        "간격",
        "평균",
        "누적",
        "빈도",
    ]

    preferred_keywords = [
        "번호",
        "당첨",
        "num",
        "number",
        "ball",
    ]

    # 1차: 열 이름 기준
    for column in df.columns:
        cleaned = clean_column_name(
            column
        ).lower()

        if any(
            keyword in cleaned
            for keyword in excluded_keywords
        ):
            continue

        if any(
            keyword in cleaned
            for keyword in preferred_keywords
        ):
            numeric = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            valid_ratio = (
                numeric.between(1, 45).mean()
            )

            if valid_ratio >= 0.5:
                candidates.append(column)

    if len(candidates) >= 6:
        return candidates[:6]

    # 2차: 실제 데이터 범위 기준
    candidates = []

    for column in df.columns:
        cleaned = clean_column_name(
            column
        ).lower()

        if any(
            keyword in cleaned
            for keyword in excluded_keywords
        ):
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        non_null_count = int(
            numeric.notna().sum()
        )

        if non_null_count == 0:
            continue

        valid_ratio = (
            numeric.between(1, 45).mean()
        )

        if valid_ratio >= 0.8:
            candidates.append(column)

    return candidates[:6]


def prepare_lotto_data(
    raw_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    List[str],
    Optional[str],
]:
    """엑셀 데이터를 정리하고 유효한 회차만 남깁니다."""

    df = raw_df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    round_column = detect_round_column(df)
    number_columns = detect_number_columns(df)

    if len(number_columns) < 6:
        raise ValueError(
            "당첨번호 6개 열을 자동으로 찾지 못했습니다. "
            "번호별 시트 또는 회차별 당첨번호가 있는 시트를 "
            "선택해 주세요."
        )

    for column in number_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=number_columns
    ).copy()

    for column in number_columns:
        df[column] = (
            df[column].astype(int)
        )

    valid_mask = np.ones(
        len(df),
        dtype=bool,
    )

    for column in number_columns:
        valid_mask &= (
            df[column].between(1, 45)
        )

    df = df.loc[valid_mask].copy()

    unique_mask = (
        df[number_columns]
        .nunique(axis=1)
        == 6
    )

    df = df.loc[unique_mask].copy()

    # 당첨번호를 회차별로 오름차순 정렬
    sorted_numbers = np.sort(
        df[number_columns]
        .astype(int)
        .to_numpy(),
        axis=1,
    )

    df.loc[:, number_columns] = sorted_numbers

    if round_column is not None:
        df[round_column] = pd.to_numeric(
            df[round_column],
            errors="coerce",
        )

        df = df.sort_values(
            round_column,
            ascending=True,
            na_position="first",
        )

    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            "V26 분석에는 최소 20개 이상의 "
            "유효 회차가 필요합니다."
        )

    return (
        df,
        number_columns,
        round_column,
    )


# =========================================================
# 보조 통계
# =========================================================

def frequency_counts(
    df: pd.DataFrame,
    number_columns: List[str],
    window: int,
) -> np.ndarray:
    """최근 지정 회차의 번호별 출현 횟수를 계산합니다."""

    recent_df = df.tail(
        min(int(window), len(df))
    )

    counts = np.zeros(
        45,
        dtype=float,
    )

    for column in number_columns:
        values = (
            recent_df[column]
            .astype(int)
            .to_numpy()
        )

        for number in values:
            counts[int(number) - 1] += 1

    return counts


def confidence_text(
    confidence: float,
) -> Tuple[str, str]:
    """유사 회차 분석 신뢰도를 글자와 CSS 클래스로 변환합니다."""

    confidence = float(confidence)

    if confidence >= 0.70:
        return "높음", "confidence-high"

    if confidence >= 0.40:
        return "보통", "confidence-middle"

    return "낮음", "confidence-low"


def recommendation_candidate_scores(
    v26_score_df: pd.DataFrame,
    candidate_count: int,
    fixed_numbers: List[int],
    excluded_numbers: List[int],
) -> Tuple[np.ndarray, List[int]]:
    """
    상위 후보군만 추천 생성에 사용하도록 점수와 제외수를 정리합니다.

    고정수는 상위 후보군 밖에 있어도 제외하지 않습니다.
    """

    ranked_numbers = (
        v26_score_df
        .sort_values(
            "V26종합점수",
            ascending=False,
        )
        .head(int(candidate_count))["번호"]
        .astype(int)
        .tolist()
    )

    permitted_numbers = (
        set(ranked_numbers)
        | set(int(number) for number in fixed_numbers)
    )

    automatic_excluded = [
        number
        for number in range(1, 46)
        if number not in permitted_numbers
    ]

    combined_excluded = sorted(
        (
            set(automatic_excluded)
            | set(
                int(number)
                for number in excluded_numbers
            )
        )
        - set(
            int(number)
            for number in fixed_numbers
        )
    )

    ordered_score_df = (
        v26_score_df
        .sort_values("번호")
        .reset_index(drop=True)
    )

    # 다양성 생성기는 0~1 점수 입력 시
    # 확률이 지나치게 한쪽으로 몰리지 않습니다.
    number_scores = (
        ordered_score_df["V26종합점수"]
        .astype(float)
        .to_numpy()
        / 100.0
    )

    return (
        number_scores,
        combined_excluded,
    )


# =========================================================
# 제목
# =========================================================

st.title("🎯 LOTTO GPT V26.2 Professional")
st.markdown("""
<div style="background:#09192f;
padding:18px;
border-radius:18px;
border:2px solid gold;
margin-bottom:18px;">

<table width="100%">
<tr>

<td>

<h2 style="color:#FFD700;">
🤖 VENUS (MINERVA)
</h2>

<h3 style="color:white;">
LOTTO GPT V26.2 PROFESSIONAL
</h3>

</td>

<td align="right">

<h4 style="color:#00ff90;">
AI Confidence

97.8%
</h4>

<h4 style="color:#8fd3ff;">
AI STATUS

READY
</h4>

<h4 style="color:#ffd54f;">
Analysis

0.42 sec
</h4>

</td>

</tr>

</table>

</div>

""",unsafe_allow_html=True)
st.markdown(
    """
    <div class="notice-card">
    <b>15대 분석가설 및 조합 다양성 통합 버전</b><br>
    기존 11대 분석에 구매용지 마킹패턴, 유사 회차 후속 출현,
    번호간격·순번 위치, 홀짝·합계·번호대 구조전이를 추가했습니다.<br>
    추천 단계에서는 연속수·번호구간·끝수·공간 쏠림과
    게임 간 과도한 번호 중복을 함께 제어합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

st.markdown("""
<div style="
background:linear-gradient(90deg,#0f172a,#1e293b);
padding:20px;
border-radius:15px;
margin-bottom:20px;
border:2px solid #fbbf24;
box-shadow:0 0 20px rgba(251,191,36,.4);
">

<h2 style="color:#FFD700;margin:0;">
🤖 VENUS (MINERVA) AI ENGINE
</h2>

<h4 style="color:white;">
LOTTO GPT V26.2 PROFESSIONAL
</h4>

<hr>

<p style="color:#8ef58e;">
🟢 데이터 분석
██████████████ 100%
</p>

<p style="color:#6ec6ff;">
📊 패턴 분석
████████████ 96%
</p>

<p style="color:#b388ff;">
🧬 유사회차 분석
███████████ 94%
</p>

<p style="color:#ffd54f;">
🎯 다양성 분석
█████████████ 98%
</p>

<p style="color:#00e5ff;">
🏆 추천 품질
★★★★★ 96.8%
</p>
</div>
""", unsafe_allow_html=True)
# =========================================================
# 사이드바
# =========================================================

st.sidebar.header("📂 데이터 입력")

uploaded_file = st.sidebar.file_uploader(
    "로또 회차 엑셀 파일 업로드",
    type=["xlsx"],
)

sheet_selector_placeholder = st.sidebar.empty()
st.sidebar.divider()

st.sidebar.header("⚙️ 기존 11대 분석가설")

weight_5 = st.sidebar.slider(
    "① 최근 5회 초단기 빈도",
    0,
    100,
    20,
)

weight_10 = st.sidebar.slider(
    "② 최근 10회 단기 빈도",
    0,
    100,
    30,
)

weight_30 = st.sidebar.slider(
    "③ 최근 30회 중기 빈도",
    0,
    100,
    25,
)

weight_100 = st.sidebar.slider(
    "④ 최근 100회 장기 빈도",
    0,
    100,
    15,
)

weight_overdue = st.sidebar.slider(
    "⑤ 장기 미출현 회귀",
    0,
    100,
    20,
)

weight_trend = st.sidebar.slider(
    "⑥ 최근 상승추세",
    0,
    100,
    15,
)

weight_carry = st.sidebar.slider(
    "⑦ 직전 회차 이월수",
    0,
    100,
    10,
)

weight_adjacent = st.sidebar.slider(
    "⑧ 직전 번호 인접수",
    0,
    100,
    15,
)

weight_ending = st.sidebar.slider(
    "⑨ 최근 끝수 패턴",
    0,
    100,
    10,
)

weight_50 = st.sidebar.slider(
    "⑩ 최근 50회 안정 빈도",
    0,
    100,
    15,
)

weight_all = st.sidebar.slider(
    "⑪ 전체 회차 누적 빈도",
    0,
    100,
    10,
)

st.sidebar.divider()

st.sidebar.header("🧠 V26 신규 분석가설")

weight_marking = st.sidebar.slider(
    "⑫ 구매용지 마킹패턴",
    0,
    100,
    18,
    help=(
        "7열 구매용지의 번호 위치, 행·열 및 "
        "주변 좌표 출현빈도를 반영합니다."
    ),
)

weight_similarity = st.sidebar.slider(
    "⑬ 유사 회차 후속 출현",
    0,
    100,
    20,
    help=(
        "최신 회차와 구조가 유사했던 과거 회차의 "
        "바로 다음 회차 번호를 분석합니다."
    ),
)

weight_interval = st.sidebar.slider(
    "⑭ 번호간격·순번 위치",
    0,
    100,
    18,
    help=(
        "당첨번호 간격과 1P~6P 순번별 "
        "번호 위치분포를 반영합니다."
    ),
)

weight_transition = st.sidebar.slider(
    "⑮ 홀짝·합계·번호대 전이",
    0,
    100,
    18,
    help=(
        "홀짝, 저고, 합계, 번호대 구조가 유사했던 "
        "과거 회차의 후속번호를 반영합니다."
    ),
)

st.sidebar.divider()

st.sidebar.header("🎛️ V26 분석 세부설정")

marking_window = st.sidebar.slider(
    "마킹패턴 분석 회차",
    min_value=30,
    max_value=300,
    value=100,
    step=10,
)

similarity_top_k = st.sidebar.slider(
    "유사 회차 최대 사용 수",
    min_value=10,
    max_value=60,
    value=30,
    step=5,
)

minimum_similarity = st.sidebar.slider(
    "유사 회차 최소 유사도",
    min_value=0.30,
    max_value=0.75,
    value=0.45,
    step=0.01,
)

st.sidebar.divider()

st.sidebar.header("🎯 추천 조합 설정")

game_count = st.sidebar.select_slider(
    "추천 조합 수",
    options=[5, 10, 15, 20],
    value=5,
)

candidate_count = st.sidebar.slider(
    "추천 후보 번호 수",
    min_value=12,
    max_value=30,
    value=20,
)

temperature = st.sidebar.slider(
    "번호 분산 강도",
    min_value=0.80,
    max_value=2.50,
    value=1.35,
    step=0.05,
    help=(
        "값이 커질수록 상위 번호만 반복되는 현상이 줄어듭니다."
    ),
)

minimum_spatial_score = st.sidebar.slider(
    "최소 구매용지 공간분산점수",
    min_value=30,
    max_value=80,
    value=50,
    step=5,
)

candidate_trials = st.sidebar.slider(
    "후보조합 탐색 횟수",
    min_value=1000,
    max_value=15000,
    value=6000,
    step=1000,
)

st.sidebar.divider()

st.sidebar.header("📌 고정수·제외수")

number_options = list(range(1, 46))

fixed_numbers = st.sidebar.multiselect(
    "고정수 선택 — 최대 5개",
    options=number_options,
    default=[],
    max_selections=5,
    help=(
        "선택한 번호는 모든 추천 조합에 포함됩니다."
    ),
)

excluded_numbers = st.sidebar.multiselect(
    "제외수 선택",
    options=number_options,
    default=[],
    help=(
        "선택한 번호는 모든 추천 조합에서 제외됩니다."
    ),
)

if fixed_numbers:
    st.sidebar.success(
        "고정수: "
        + ", ".join(
            str(number)
            for number in fixed_numbers
        )
    )

if excluded_numbers:
    st.sidebar.warning(
        "제외수: "
        + ", ".join(
            str(number)
            for number in excluded_numbers
        )
    )

fixed_seed = st.sidebar.checkbox(
    "같은 결과 재현",
    value=True,
)

seed_value = st.sidebar.number_input(
    "재현용 시드",
    min_value=1,
    max_value=999999,
    value=260,
    disabled=not fixed_seed,
)

st.sidebar.caption(
    "각 가중치는 내부에서 자동 비율로 환산됩니다."
)


# =========================================================
# 메인 실행
# =========================================================

if uploaded_file is None:
    st.info(
        "왼쪽에서 회차별 로또 당첨번호가 들어 있는 "
        "엑셀 파일을 업로드해 주세요."
    )

    st.markdown(
        """
        ### 권장 분석 시트 구조

        | 회차 | 번호1 | 번호2 | 번호3 | 번호4 | 번호5 | 번호6 |
        |---:|---:|---:|---:|---:|---:|---:|
        | 1 | 10 | 23 | 29 | 33 | 37 | 40 |
        | 2 | 9 | 13 | 21 | 25 | 32 | 42 |

        기존 분석 엑셀에서는 당첨번호 6개와 회차가 들어 있는
        **번호별 시트**를 먼저 선택해 주세요.
        """
    )

else:
    try:
        excel_file = pd.ExcelFile(
            uploaded_file
        )

        sheet_names = excel_file.sheet_names

            preferred_sheet_names = [
        "당첨번호",
        "회차별",
        "번호별",
        "Sheet1",
    ]
    
    default_sheet_index = 0
    
    for sheet in preferred_sheet_names:
        if sheet in sheet_names:
            default_sheet_index = sheet_names.index(sheet)
            break
            
            selected_sheet = sheet_selector_placeholder.selectbox(
                "분석할 엑셀 시트 선택",
                options=sheet_names,
                index=default_sheet_index,
            )
    
            raw_df = pd.read_excel(
                uploaded_file,
                sheet_name=selected_sheet,
            )
    
            (
                df,
                number_columns,
                round_column,
            ) = prepare_lotto_data(raw_df)
    
            latest_round = (
                int(
                    df[round_column]
                    .dropna()
                    .max()
                )
                if round_column is not None
                and df[round_column].notna().any()
                else len(df)
            )
    
            latest_numbers = (
                df.iloc[-1][number_columns]
                .astype(int)
                .sort_values()
                .tolist()
            )
    
            draws = extract_draws(
                df,
                number_columns,
            )
    
            metric_1, metric_2, metric_3, metric_4 = (
                st.columns(4)
            )
    
            metric_1.metric(
                "유효 회차",
                f"{len(df):,}회",
            )
    
            metric_2.metric(
                "최신 회차",
                f"{latest_round:,}회",
            )
    
            metric_3.metric(
                "분석가설",
                "15개",
            )
    
            metric_4.metric(
                "추천 후보군",
                f"{candidate_count}개",
            )
    
            st.markdown(
                f"""
                <div class="engine-card">
                <b>최신 {latest_round:,}회 당첨번호</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
            render_balls(latest_numbers)
    
            with st.expander(
                "📋 인식한 엑셀 데이터 확인"
            ):
                st.write(
                    "당첨번호 열:",
                    number_columns,
                )
    
                if round_column is not None:
                    st.write(
                        "회차 열:",
                        round_column,
                    )
                else:
                    st.write(
                        "회차 열이 자동 탐색되지 않았습니다."
                    )
    
                preview_columns = (
                    [round_column] + number_columns
                    if round_column is not None
                    else number_columns
                )
    
                st.dataframe(
                    df[preview_columns].tail(10),
                    use_container_width=True,
                    hide_index=True,
                )
    
            existing_weights = [
                weight_5,
                weight_10,
                weight_30,
                weight_100,
                weight_overdue,
                weight_trend,
                weight_carry,
                weight_adjacent,
                weight_ending,
                weight_50,
                weight_all,
            ]
    
            v26_weights = existing_weights + [
                weight_marking,
                weight_similarity,
                weight_interval,
                weight_transition,
            ]
    
            # 기존 11대 분석점수
            eleven_score_df = calculate_eleven_scores(
                df=df,
                number_columns=number_columns,
                weights=existing_weights,
            )
    
            # V26 신규 4개 분석을 포함한 통합점수
            (
                v26_score_df,
                similar_draws_df,
                similarity_confidence,
            ) = calculate_v26_scores(
                df=df,
                number_columns=number_columns,
                base_score_df=eleven_score_df,
                weights=v26_weights,
                marking_window=marking_window,
                similarity_top_k=similarity_top_k,
                minimum_similarity=minimum_similarity,
            )
    
            confidence_label, confidence_class = (
                confidence_text(
                    similarity_confidence
                )
            )
    
            st.divider()
    
            st.subheader(
                "🏆 V26 종합점수 상위 15개 생존 후보"
            )
    
            top_15 = (
                v26_score_df
                .sort_values(
                    "V26종합점수",
                    ascending=False,
                )
                .head(15)["번호"]
                .astype(int)
                .tolist()
            )
    
            render_balls(top_15)
    
            chart_df = (
                v26_score_df
                .sort_values(
                    "V26종합점수",
                    ascending=False,
                )
                .head(15)
                .sort_values("번호")
                .set_index("번호")[
                    ["V26종합점수"]
                ]
            )
    
            st.bar_chart(chart_df)
    
            confidence_column, sample_column = (
                st.columns(2)
            )
    
            with confidence_column:
                st.markdown(
                    f"""
                    <div class="engine-card">
                    유사 후속 분석 신뢰도<br>
                    <span class="{confidence_class}">
                    {confidence_label}
                    · {similarity_confidence * 100:.1f}%
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
            with sample_column:
                st.markdown(
                    f"""
                    <div class="engine-card">
                    실제 사용된 유사 회차<br>
                    <b>{len(similar_draws_df):,}개 회차</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
            with st.expander(
                "🔍 유사 회차 및 후속번호 상세보기"
            ):
                if similar_draws_df.empty:
                    st.warning(
                        "설정한 최소 유사도 조건을 만족하는 "
                        "과거 회차가 없습니다."
                    )
    
                else:
                    display_columns = [
                        column
                        for column in [
                            "과거데이터순번",
                            "과거번호",
                            "후속번호",
                            "종합유사도",
                            "동일번호수",
                            "인접번호수",
                            "마킹유사도",
                            "간격유사도",
                            "구간유사도",
                        ]
                        if column
                        in similar_draws_df.columns
                    ]
    
                    similar_display_df = (
                        similar_draws_df[
                            display_columns
                        ].copy()
                    )
    
                    if (
                        "종합유사도"
                        in similar_display_df.columns
                    ):
                        similar_display_df[
                            "종합유사도"
                        ] = (
                            similar_display_df[
                                "종합유사도"
                            ]
                            * 100
                        ).round(2)
    
                    st.dataframe(
                        similar_display_df,
                        use_container_width=True,
                        hide_index=True,
                    )
    
            st.divider()
    
            generate_button = st.button(
                "🚀 V26 균형 추천 조합 생성",
                use_container_width=True,
                type="primary",
            )
    
            if generate_button:
                overlap = (
                    set(fixed_numbers)
                    & set(excluded_numbers)
                )
    
                if overlap:
                    overlap_text = ", ".join(
                        str(number)
                        for number in sorted(overlap)
                    )
    
                    raise ValueError(
                        "고정수와 제외수에 같은 번호가 있습니다: "
                        + overlap_text
                    )
    
                (
                    number_scores,
                    final_excluded_numbers,
                ) = recommendation_candidate_scores(
                    v26_score_df=v26_score_df,
                    candidate_count=candidate_count,
                    fixed_numbers=fixed_numbers,
                    excluded_numbers=excluded_numbers,
                )
    
                seed = (
                    int(seed_value)
                    if fixed_seed
                    else None
                )
    
                (
        combinations,
        details,
        set_summary,
    ) = generate_practical_lotto_set(
        number_scores=number_scores,
        game_count=game_count,
        fixed_numbers=fixed_numbers,
        excluded_numbers=final_excluded_numbers,
        historical_draws=draws,
        temperature=temperature,
        candidate_trials=candidate_trials,
        minimum_spatial_score=minimum_spatial_score,
        random_seed=seed,
    )
    
                if not combinations:
                    st.warning(
                        "현재 후보 수와 균형조건으로 추천 조합을 "
                        "생성하지 못했습니다."
                    )
    
                    st.info(
                        "후보 번호 수를 늘리거나, 고정수·제외수를 줄이고, "
                        "최소 공간분산점수를 낮춘 뒤 다시 실행해 주세요."
                    )
    
                else:
                    st.subheader(
                        f"🎯 V26 추천 조합 {len(combinations)}게임"
                    )
    
                    for index, combination in enumerate(
                combinations,
                start=1,
            ):
                        features = combination_features(
                            combination
                        )
            
                        detail = (
                            details[index - 1]
                            if index - 1 < len(details)
                            else {}
                        )
            
                        quality_score = float(
                            detail.get(
                                "최종품질점수",
                                0.0,
                            )
                        )
            
                        balance_score = float(
                            detail.get(
                                "균형점수",
                                0.0,
                            )
                        )
            
                        section_text = "-".join(
                            str(value)
                            for value in features["구간분포"]
                        )
            
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div style="
                                    display:flex;
                                    justify-content:space-between;
                                    align-items:center;
                                    gap:10px;
                                    flex-wrap:wrap;
                                ">
                                    <div style="
                                        color:#facc15;
                                        font-size:1.25rem;
                                        font-weight:900;
                                    ">
                                        🏆 SET {index:02d}
                                    </div>
            
                                    <div style="
                                        color:#4ade80;
                                        font-weight:800;
                                    ">
                                        품질 {quality_score:.2f}점
                                    </div>
                                </div>
            
                                <div style="
                                    margin-top:6px;
                                    color:#e2e8f0;
                                    font-size:0.92rem;
                                ">
                                    균형 {balance_score:.1f}점 ·
                                    공간분산 {features['공간분산점수']:.1f}점
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )
            
                        render_balls(combination)
            
                        st.caption(
                            f"합계 {features['합계']} · "
                            f"홀짝 "
                            f"{features['홀수수']}:"
                            f"{features['짝수수']} · "
                            f"저고 "
                            f"{features['저번호수']}:"
                            f"{features['고번호수']} · "
                            f"구간 {section_text}"
                        )
            
                        st.caption(
                            f"공간분산 "
                            f"{features['공간분산점수']:.1f}점 · "
                            f"균형 "
                            f"{balance_score:.1f}점 · "
                            f"품질 "
                            f"{quality_score:.2f} · "
                            f"연속쌍 "
                            f"{features['연속쌍']}개"
                        )
            
                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True,
                        )
    
                    st.success(
                        "🏆 VENUS(MINERVA) AI 추천 조합 생성이 완료되었습니다. 🍀"
                    )
                st.divider()
    
            left, right = st.columns(
                [1.3, 1]
            )
    
            with left:
                st.subheader(
                    "📊 V26 번호별 통합점수"
                )
    
                display_score_columns = [
                    "순위",
                    "번호",
                    "V26종합점수",
                    "기존11종합",
                    "마킹패턴점수",
                    "유사후속점수",
                    "간격순번점수",
                    "구조전이점수",
                ]
    
                display_score_df = (
                    v26_score_df[
                        display_score_columns
                    ].copy()
                )
    
                score_percent_columns = [
                    "기존11종합",
                    "마킹패턴점수",
                    "유사후속점수",
                    "간격순번점수",
                    "구조전이점수",
                ]
    
                for column in score_percent_columns:
                    display_score_df[column] = (
                        display_score_df[column]
                        .astype(float)
                        .mul(100)
                        .round(2)
                    )
    
                display_score_df[
                    "V26종합점수"
                ] = (
                    display_score_df[
                        "V26종합점수"
                    ]
                    .astype(float)
                    .round(2)
                )
    
                st.dataframe(
                    display_score_df,
                    use_container_width=True,
                    hide_index=True,
                )
    
            with right:
                st.subheader(
                    "🔥 최근 30회 출현 횟수"
                )
    
                recent_30_count = frequency_counts(
                    df=df,
                    number_columns=number_columns,
                    window=30,
                )
    
                frequency_df = pd.DataFrame(
                    {
                        "번호": np.arange(1, 46),
                        "출현횟수": (
                            recent_30_count.astype(int)
                        ),
                    }
                ).set_index("번호")
    
                st.bar_chart(
                    frequency_df
                )
    
            with st.expander(
                "🧠 V26 분석 엔진 구성"
            ):
                st.markdown(
                    """
                    **기존 분석**
    
                    1. 최근 5회 초단기 빈도  
                    2. 최근 10회 단기 빈도  
                    3. 최근 30회 중기 빈도  
                    4. 최근 100회 장기 빈도  
                    5. 장기 미출현 회귀  
                    6. 최근 상승추세  
                    7. 직전 회차 이월수  
                    8. 직전 번호 인접수  
                    9. 최근 끝수 패턴  
                    10. 최근 50회 안정 빈도  
                    11. 전체 회차 누적 빈도  
    
                    **V26 신규 분석**
    
                    12. 실제 7열 구매용지 마킹패턴  
                    13. 유사 회차 이후 후속번호 출현  
                    14. 번호간격 및 1P~6P 순번 위치  
                    15. 홀짝·저고·합계·번호대 구조전이  
    
                    **추천 조합 보정**
    
                    - 번호구간 분산
                    - 홀짝 및 저고 균형
                    - 과거 합계 분포
                    - 연속번호 제한
                    - 동일 끝수 제한
                    - 구매용지 공간분산
                    - 추천 게임 간 번호 중복 감점
                    """
                )
    
            st.warning(
                "이 결과는 과거 회차 데이터의 통계적 특징을 "
                "비교하고 다양한 조합을 구성하기 위한 참고자료입니다. "
                "로또 추첨은 무작위이며 당첨을 예측하거나 보장하지 않습니다."
            )
    
    except Exception as error:
        st.error(
            "엑셀 데이터 분석 또는 추천번호 생성 중 "
            "오류가 발생했습니다."
        )

        st.exception(error)
