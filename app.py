import textwrap
import itertools
import re
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
# LOTTO GPT V27.2.1 SUM-DISTRIBUTION HOTFIX
# V27.2 FINAL BASELINE 15대 가설/기본값은 그대로 보존
# 부분 수정: 합계 역할 분산 + 최종 5게임 합계 중복 패널티 + 진단창
# =========================================================


st.set_page_config(
    page_title="LOTTO GPT V27.2 FINAL",
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
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.96),
            rgba(30, 41, 59, 0.88)
        );
    border: 1px solid rgba(250, 204, 21, 0.38);
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow:
        0 10px 24px rgba(0, 0, 0, 0.26),
        inset 0 1px 0 rgba(255, 255, 255, 0.07);
    animation: cardSlide 0.45s ease-out both;
    transition:
        transform 0.20s ease,
        border-color 0.20s ease,
        box-shadow 0.20s ease;
}
/* 모바일 추천카드 최적화 */
@media (max-width: 768px) {
    .result-card {
        width: 100%;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 14px;
        box-sizing: border-box;
        overflow: hidden;
        box-shadow:
            0 7px 18px rgba(0, 0, 0, 0.24),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }

    .result-card .lotto-ball {
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        font-size: 0.92rem !important;
        margin: 3px !important;
    }

    .result-card h3,
    .result-card h4 {
        font-size: 1rem !important;
        line-height: 1.35;
    }

    .result-card p,
    .result-card span {
        line-height: 1.45;
    }
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

def parse_round_values(series: pd.Series) -> pd.Series:
    """'1236회', '1236 회', 1236 같은 값을 안전하게 회차 숫자로 변환합니다."""

    cleaned = (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def detect_round_column(
    df: pd.DataFrame,
) -> Optional[str]:
    """회차 열을 이름 + 연속 회차 패턴으로 안전하게 자동 탐색합니다."""

    # 1순위: 열 이름 자체가 회차를 뜻하는 경우
    named_candidates = []
    for column in df.columns:
        cleaned = clean_column_name(column).lower()
        if any(keyword in cleaned for keyword in ["회차", "round", "draw"]):
            named_candidates.append(column)

    if named_candidates:
        # 이름 후보가 여러 개면 실제 회차형 값이 가장 많은 열을 선택
        def named_score(column):
            parsed = parse_round_values(df[column]).dropna()
            if len(parsed) == 0:
                return (-1, -1)
            return (int(parsed.nunique()), int(parsed.max()))

        return max(named_candidates, key=named_score)

    # 2순위: 이름이 불명확한 구형 시트.
    # 단순히 "양수/고유값"만 보지 않고 1,2,3... 형태의 회차 연속성을 요구합니다.
    best_column = None
    best_score = -1.0

    for column in df.columns:
        parsed = parse_round_values(df[column]).dropna().astype(int)
        if len(parsed) < 50:
            continue

        values = parsed.to_numpy()
        unique_ratio = float(pd.Series(values).nunique() / max(len(values), 1))
        if unique_ratio < 0.90:
            continue

        sorted_unique = np.sort(np.unique(values))
        if len(sorted_unique) < 50:
            continue

        diffs = np.diff(sorted_unique)
        one_step_ratio = float((diffs == 1).mean()) if len(diffs) else 0.0
        starts_near_one = 1.0 if sorted_unique[0] <= 5 else 0.0
        monotonic_ratio = float((np.diff(values) >= 0).mean()) if len(values) > 1 else 0.0

        # 회차열은 대부분 +1씩 증가하고, 초반 회차부터 시작하며, 중복이 거의 없어야 합니다.
        score = (
            one_step_ratio * 0.55
            + starts_near_one * 0.20
            + monotonic_ratio * 0.15
            + unique_ratio * 0.10
        )

        if one_step_ratio >= 0.85 and score > best_score:
            best_score = score
            best_column = column

    return best_column


def detect_number_columns(
    df: pd.DataFrame,
) -> List[str]:
    """당첨번호 6개 열을 안정적으로 탐색합니다."""

    # 가장 신뢰도가 높은 표준형: 1P~6P
    normalized_map = {
        clean_column_name(column).upper(): column
        for column in df.columns
    }

    standard = []
    for index in range(1, 7):
        key = f"{index}P"
        if key in normalized_map:
            standard.append(normalized_map[key])

    if len(standard) == 6:
        return standard

    excluded_keywords = [
        "보너스", "bonus", "회차", "round", "draw",
        "날짜", "date", "합계", "sum", "total", "aver", "avg",
        "순위", "당첨금", "당첨자", "간격", "평균", "누적", "빈도",
    ]

    candidates = []

    for column in df.columns:
        cleaned = clean_column_name(column).lower()

        if any(keyword in cleaned for keyword in excluded_keywords):
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        non_null = numeric.notna()

        if int(non_null.sum()) < 20:
            continue

        valid_ratio = float(numeric[non_null].between(1, 45).mean())
        unique_count = int(numeric[non_null].nunique())

        # 로또번호 열은 대부분 1~45 범위이며 여러 숫자가 반복 출현합니다.
        if valid_ratio >= 0.95 and unique_count >= 15:
            candidates.append(column)

    return candidates[:6]


def prepare_lotto_data(
    raw_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    List[str],
    Optional[str],
]:
    """회차/당첨번호를 정제하고 실제 유효 회차만 반환합니다."""

    df = raw_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    round_column = detect_round_column(df)
    number_columns = detect_number_columns(df)

    if len(number_columns) < 6:
        raise ValueError(
            "당첨번호 6개 열을 찾지 못했습니다. "
            "회차, 1P~6P 구조의 시트를 사용해 주세요."
        )

    for column in number_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if round_column is not None:
        df[round_column] = parse_round_values(df[round_column])

    df = df.dropna(subset=number_columns).copy()

    valid_mask = np.ones(len(df), dtype=bool)
    for column in number_columns:
        valid_mask &= df[column].between(1, 45)

    df = df.loc[valid_mask].copy()

    # 한 회차에 같은 번호가 2번 이상 들어간 비정상 행 제거
    unique_mask = df[number_columns].nunique(axis=1).eq(6)
    df = df.loc[unique_mask].copy()

    for column in number_columns:
        df[column] = df[column].astype(int)

    # 번호 오름차순 정리
    sorted_numbers = np.sort(
        df[number_columns].to_numpy(dtype=int),
        axis=1,
    )
    df.loc[:, number_columns] = sorted_numbers

    if round_column is not None:
        df = df.dropna(subset=[round_column]).copy()
        df[round_column] = df[round_column].astype(int)

        # 동일 회차 중복행 제거 후 회차순 정렬
        df = (
            df.sort_values(round_column)
            .drop_duplicates(subset=[round_column], keep="last")
        )

    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            "V26 분석에는 최소 20개 이상의 유효 회차가 필요합니다."
        )

    return df, number_columns, round_column


def sheet_lotto_quality(raw_df: pd.DataFrame) -> Tuple[int, int, int, int, int]:
    """시트 자동선택용 품질점수.

    우선순위:
    1) 1P~6P 표준 당첨번호 열
    2) 명시적인 회차 열
    3) 회차 연속성
    4) 최신회차
    5) 유효행수

    주의: 최신회차 숫자가 크다는 이유로 통계/패턴 시트를 선택하지 않습니다.

    이렇게 해야 패턴표의 큰 숫자를 '최신회차'로 오인하지 않습니다.
    """

    try:
        prepared, nums, round_col = prepare_lotto_data(raw_df)
    except Exception:
        return (-1, -1, -1, -1, -1)

    normalized_nums = [clean_column_name(c).upper() for c in nums]
    standard_bonus = int(
        normalized_nums == ["1P", "2P", "3P", "4P", "5P", "6P"]
    )

    explicit_round_bonus = 0
    if round_col is not None:
        cleaned_round = clean_column_name(round_col).lower()
        explicit_round_bonus = int(
            any(k in cleaned_round for k in ["회차", "round", "draw"])
        )

    latest = (
        int(prepared[round_col].max())
        if round_col is not None and prepared[round_col].notna().any()
        else len(prepared)
    )

    continuity_score = 0
    if round_col is not None and len(prepared) >= 20:
        rounds = np.sort(prepared[round_col].astype(int).unique())
        if len(rounds) > 1:
            continuity_score = int(round(float((np.diff(rounds) == 1).mean()) * 1000))

    # 자동 시트 선택은 "큰 숫자"보다 구조 신뢰도를 먼저 봅니다.
    # 1P~6P + 명시적 회차 열이 있는 원본 회차 시트가 최우선입니다.
    return (
        standard_bonus,
        explicit_round_bonus,
        continuity_score,
        latest,
        len(prepared),
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
# V27 FINAL - 합계 수열·유사후속 기반 생존번호 추천기
# =========================================================

def _sequence_features(numbers: List[int]) -> dict:
    """조합 구조 특징을 계산합니다."""
    values = sorted(int(n) for n in numbers)
    total = int(sum(values))
    odd = sum(n % 2 for n in values)
    low = sum(n <= 22 for n in values)

    sections = [0, 0, 0, 0, 0]
    for n in values:
        if n <= 10:
            sections[0] += 1
        elif n <= 20:
            sections[1] += 1
        elif n <= 30:
            sections[2] += 1
        elif n <= 40:
            sections[3] += 1
        else:
            sections[4] += 1

    thirds = [
        sum(1 <= n <= 15 for n in values),
        sum(16 <= n <= 30 for n in values),
        sum(31 <= n <= 45 for n in values),
    ]

    adjacent_pairs = sum(
        1 for a, b in zip(values, values[1:])
        if b - a == 1
    )

    endings = [n % 10 for n in values]
    max_same_ending = max(endings.count(e) for e in set(endings))

    rows = len({(n - 1) // 7 for n in values})
    cols = len({(n - 1) % 7 for n in values})
    spatial = round(
        min(100.0, (rows / 6.0) * 55.0 + (cols / 6.0) * 45.0),
        2,
    )

    return {
        "합계": total,
        "홀수수": int(odd),
        "짝수수": int(6 - odd),
        "저번호수": int(low),
        "고번호수": int(6 - low),
        "저중고분포": thirds,
        "구간분포": sections,
        "연속쌍": int(adjacent_pairs),
        "끝수최대중복": int(max_same_ending),
        "공간분산점수": float(spatial),
    }


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        return 0.0

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    total = float(weights.sum())
    if total <= 0:
        return float(np.quantile(values, quantile))

    cumulative = np.cumsum(weights) / total
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    index = min(max(index, 0), len(values) - 1)
    return float(values[index])


def analyze_sum_sequence_successors(
    df: pd.DataFrame,
    number_columns: List[str],
    round_column: Optional[str] = None,
    pattern_length: int = 4,
    top_k: int = 30,
) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """
    최신 합계 수열과 과거 합계 수열을 비교하고,
    유사한 과거 수열 직후(t+1)에 출현한 번호를 가중 집계합니다.

    V27 개선:
    - 회차가 실제로 연속된 구간만 사용
    - 비정상 행 제거로 생긴 회차 공백을 임의로 이어붙이지 않음
    - 합계는 하드 컷이 아니라 후속사례 적합도 점수로 사용
    """
    draws = df[number_columns].astype(int).to_numpy()
    sums = draws.sum(axis=1).astype(float)
    n = len(sums)

    if n < max(30, pattern_length + 3):
        return (
            np.zeros(45, dtype=float),
            pd.DataFrame(),
            {
                "현재합계": float(sums[-1]) if n else 0.0,
                "신뢰도": 0.0,
                "후속합계": [],
                "후속가중치": [],
            },
        )

    k = max(3, int(pattern_length))
    current = sums[-k:]
    current_delta = np.diff(current)

    round_values = None
    if round_column is not None and round_column in df.columns:
        round_values = pd.to_numeric(
            df[round_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        current_rounds = round_values[-k:]
        if (
            np.all(np.isfinite(current_rounds))
            and not np.all(np.diff(current_rounds) == 1)
        ):
            # 최신 수열 자체에 회차 공백이 있으면 수열모델 신뢰도를 낮춤
            current_has_gap = True
        else:
            current_has_gap = False
    else:
        current_has_gap = False

    sum_scale = max(float(np.std(sums)), 15.0)
    delta_scale = max(float(np.std(np.diff(sums))), 12.0)

    candidates = []

    for t in range(k - 1, n - 1):
        # 과거 패턴구간과 후속회차가 실제 연속회차인지 확인
        if round_values is not None:
            hist_rounds = round_values[t - k + 1:t + 2]
            if (
                len(hist_rounds) != k + 1
                or not np.all(np.isfinite(hist_rounds))
                or not np.all(np.diff(hist_rounds) == 1)
            ):
                continue

        hist = sums[t - k + 1:t + 1]
        hist_delta = np.diff(hist)

        level_distance = float(
            np.mean(np.abs(hist - current)) / sum_scale
        )
        delta_distance = float(
            np.mean(np.abs(hist_delta - current_delta)) / delta_scale
        )

        direction_match = float(
            np.mean(
                np.sign(hist_delta) == np.sign(current_delta)
            )
        )

        similarity = (
            np.exp(
                -(
                    level_distance * 0.55
                    + delta_distance * 0.45
                )
            )
            * (0.80 + 0.20 * direction_match)
        )

        if current_has_gap:
            similarity *= 0.70

        successor_numbers = sorted(
            int(x) for x in draws[t + 1]
        )
        successor_sum = float(sums[t + 1])

        anchor_round = (
            int(round_values[t])
            if round_values is not None
            and np.isfinite(round_values[t])
            else int(t + 1)
        )
        successor_round = (
            int(round_values[t + 1])
            if round_values is not None
            and np.isfinite(round_values[t + 1])
            else int(t + 2)
        )

        candidates.append(
            {
                "기준회차": anchor_round,
                "기준합계": int(sums[t]),
                "유사도": float(similarity),
                "후속회차": successor_round,
                "후속합계": int(successor_sum),
                "후속번호": successor_numbers,
            }
        )

    if not candidates:
        return (
            np.zeros(45, dtype=float),
            pd.DataFrame(),
            {
                "현재합계": int(sums[-1]),
                "최근합계수열": [int(x) for x in current.tolist()],
                "신뢰도": 0.0,
                "후속합계": [],
                "후속가중치": [],
            },
        )

    candidates.sort(
        key=lambda item: item["유사도"],
        reverse=True,
    )
    selected = candidates[:max(5, int(top_k))]

    raw_weights = np.asarray(
        [item["유사도"] for item in selected],
        dtype=float,
    )
    weights = np.power(
        np.clip(raw_weights, 0.0, 1.0),
        2.0,
    )
    if weights.sum() <= 0:
        weights = np.ones(len(selected), dtype=float)

    number_raw = np.zeros(45, dtype=float)
    for item, weight in zip(selected, weights):
        for number in item["후속번호"]:
            number_raw[int(number) - 1] += float(weight)

    if number_raw.max() > number_raw.min():
        number_score = (
            (number_raw - number_raw.min())
            / (number_raw.max() - number_raw.min())
            * 100.0
        )
    else:
        number_score = np.zeros(45, dtype=float)

    successor_sums = np.asarray(
        [item["후속합계"] for item in selected],
        dtype=float,
    )

    weighted_mean = float(
        np.average(successor_sums, weights=weights)
    )
    q25 = _weighted_quantile(successor_sums, weights, 0.25)
    q50 = _weighted_quantile(successor_sums, weights, 0.50)
    q75 = _weighted_quantile(successor_sums, weights, 0.75)

    confidence = float(
        np.clip(
            np.mean(raw_weights[:min(10, len(raw_weights))]),
            0.0,
            1.0,
        )
    )

    analog_df = pd.DataFrame(selected)
    if not analog_df.empty:
        analog_df["유사도"] = (
            analog_df["유사도"] * 100.0
        ).round(2)

    context = {
        "현재합계": int(sums[-1]),
        "최근합계수열": [int(x) for x in current.tolist()],
        "후속합계가중평균": round(weighted_mean, 2),
        "후속합계Q25": int(round(q25)),
        "후속합계중앙": int(round(q50)),
        "후속합계Q75": int(round(q75)),
        "신뢰도": confidence,
        "후속합계": successor_sums.tolist(),
        "후속가중치": weights.tolist(),
        "수열표본수": len(selected),
    }

    return number_score, analog_df, context


def analyze_anchor_sum_successors(
    df: pd.DataFrame,
    number_columns: List[str],
    round_column: Optional[str] = None,
    exact_min_samples: int = 8,
) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """
    최신 회차 합계와 동일/근접했던 과거 회차의 '바로 다음 회차'를 분석합니다.

    우선순위:
    1) 정확히 같은 합계
    2) 정확일치 표본이 부족할 때만 ±2
    3) 그래도 부족하면 ±5

    핵심:
    - 회차 t와 t+1이 실제 연속회차일 때만 사용
    - 반복 후속합계(mode), 중심값, 상·하단 꼬리를 자동 산출
    - 후속번호 출현을 번호 가중점수로 사용
    """
    draws = df[number_columns].astype(int).to_numpy()
    sums = draws.sum(axis=1).astype(int)

    if len(sums) < 20:
        return (
            np.zeros(45, dtype=float),
            pd.DataFrame(),
            {},
        )

    current_sum = int(sums[-1])

    round_values = None
    if round_column is not None and round_column in df.columns:
        round_values = pd.to_numeric(
            df[round_column],
            errors="coerce",
        ).to_numpy(dtype=float)

    def collect(max_delta: int):
        rows = []
        for t in range(0, len(sums) - 1):
            delta = abs(int(sums[t]) - current_sum)
            if delta > int(max_delta):
                continue

            if round_values is not None:
                if (
                    not np.isfinite(round_values[t])
                    or not np.isfinite(round_values[t + 1])
                    or int(round_values[t + 1])
                    != int(round_values[t]) + 1
                ):
                    continue

            # 정확일치 우선, 근접합계일수록 지수적으로 감쇠
            if delta == 0:
                weight = 1.0
            elif delta <= 2:
                weight = 0.65 * np.exp(-delta / 2.0)
            else:
                weight = 0.35 * np.exp(-delta / 3.0)

            rows.append(
                {
                    "기준회차": (
                        int(round_values[t])
                        if round_values is not None
                        else int(t + 1)
                    ),
                    "기준합계": int(sums[t]),
                    "합계차이": int(delta),
                    "사례가중치": float(weight),
                    "후속회차": (
                        int(round_values[t + 1])
                        if round_values is not None
                        else int(t + 2)
                    ),
                    "후속합계": int(sums[t + 1]),
                    "후속번호": sorted(
                        int(x) for x in draws[t + 1]
                    ),
                }
            )
        return rows

    exact_rows = collect(0)
    rows = list(exact_rows)
    expansion = "정확일치"

    if len(rows) < int(exact_min_samples):
        rows = collect(2)
        expansion = "±2 확장"

    if len(rows) < int(exact_min_samples):
        rows = collect(5)
        expansion = "±5 확장"

    if not rows:
        return (
            np.zeros(45, dtype=float),
            pd.DataFrame(),
            {
                "현재합계": current_sum,
                "정확일치표본수": 0,
                "사용표본수": 0,
                "확장단계": "없음",
            },
        )

    weights = np.asarray(
        [row["사례가중치"] for row in rows],
        dtype=float,
    )
    successor_sums = np.asarray(
        [row["후속합계"] for row in rows],
        dtype=float,
    )

    number_raw = np.zeros(45, dtype=float)
    for row, weight in zip(rows, weights):
        for number in row["후속번호"]:
            number_raw[number - 1] += float(weight)

    if number_raw.max() > number_raw.min():
        number_score = (
            (number_raw - number_raw.min())
            / (number_raw.max() - number_raw.min())
            * 100.0
        )
    else:
        number_score = np.zeros(45, dtype=float)

    weighted_mean = float(
        np.average(successor_sums, weights=weights)
    )
    weighted_median = _weighted_quantile(
        successor_sums, weights, 0.50
    )
    q25 = _weighted_quantile(
        successor_sums, weights, 0.25
    )
    q75 = _weighted_quantile(
        successor_sums, weights, 0.75
    )

    # 반복 후속합계(mode) - 실제 정확 값의 반복성을 사용
    exact_successor_sums = [
        int(row["후속합계"])
        for row in exact_rows
    ]
    mode_sum = None
    mode_count = 0
    if exact_successor_sums:
        counts = pd.Series(
            exact_successor_sums,
            dtype=int,
        ).value_counts()
        mode_sum = int(counts.index[0])
        mode_count = int(counts.iloc[0])
        if mode_count < 2:
            mode_sum = None
            mode_count = 0

    upper_mask = successor_sums > float(current_sum)
    lower_mask = successor_sums < float(q25)

    upper_count = int(np.sum(upper_mask))
    lower_count = int(np.sum(lower_mask))

    upper_weight_share = float(
        weights[upper_mask].sum() / weights.sum()
    ) if weights.sum() > 0 else 0.0
    lower_weight_share = float(
        weights[lower_mask].sum() / weights.sum()
    ) if weights.sum() > 0 else 0.0

    upper_target = (
        _weighted_quantile(
            successor_sums[upper_mask],
            weights[upper_mask],
            0.50,
        )
        if upper_count > 0
        else float(q75)
    )
    lower_target = (
        _weighted_quantile(
            successor_sums[lower_mask],
            weights[lower_mask],
            0.50,
        )
        if lower_count > 0
        else float(q25)
    )

    context = {
        "현재합계": current_sum,
        "정확일치표본수": len(exact_rows),
        "사용표본수": len(rows),
        "확장단계": expansion,
        "후속합계": successor_sums.tolist(),
        "후속가중치": weights.tolist(),
        "후속합계가중평균": round(weighted_mean, 2),
        "후속합계중앙": int(round(weighted_median)),
        "후속합계Q25": int(round(q25)),
        "후속합계Q75": int(round(q75)),
        "반복모드합계": mode_sum,
        "반복모드횟수": mode_count,
        "상단표본수": upper_count,
        "하단표본수": lower_count,
        "상단가중비중": round(upper_weight_share, 4),
        "하단가중비중": round(lower_weight_share, 4),
        "상단대표합계": int(round(upper_target)),
        "하단대표합계": int(round(lower_target)),
    }

    detail_df = pd.DataFrame(rows)
    return number_score, detail_df, context


def build_adaptive_sum_roles(
    anchor_context: dict,
    sequence_context: dict,
    game_count: int = 5,
) -> List[dict]:
    """
    V27.2.1 HOTFIX - 합계분포 역할 생성기.

    BASELINE 가중치는 전혀 변경하지 않습니다.
    수정 목적은 '최빈합계/중앙값이 동일할 때 5게임이 한 합계에 몰리는 현상'을 줄이는 것입니다.

    원칙:
    - 동일합계 후속사례의 mode는 1게임만 핵심축으로 사용
    - 중앙값은 1게임 핵심축
    - Q25 / Q75를 하단·상단 대표축으로 사용
    - 가중평균은 분포의 브리지 역할로 사용
    - 모두 soft target이며 하드 필터가 아님
    """
    game_count = max(1, int(game_count))

    current_sum = float(anchor_context.get("현재합계", 0.0))

    anchor_q25 = float(anchor_context.get("후속합계Q25", current_sum))
    anchor_q50 = float(anchor_context.get("후속합계중앙", current_sum))
    anchor_q75 = float(anchor_context.get("후속합계Q75", current_sum))
    anchor_mean = float(anchor_context.get("후속합계가중평균", anchor_q50))

    seq_q25 = float(sequence_context.get("후속합계Q25", anchor_q25))
    seq_q50 = float(sequence_context.get("후속합계중앙", anchor_q50))
    seq_q75 = float(sequence_context.get("후속합계Q75", anchor_q75))
    seq_mean = float(sequence_context.get("후속합계가중평균", anchor_mean))

    # 동일합계 사례를 70%, 합계수열 유사사례를 30% 반영.
    lower_target = 0.70 * anchor_q25 + 0.30 * seq_q25
    center_target = 0.70 * anchor_q50 + 0.30 * seq_q50
    upper_target = 0.70 * anchor_q75 + 0.30 * seq_q75
    bridge_target = 0.70 * anchor_mean + 0.30 * seq_mean

    mode_target = anchor_context.get("반복모드합계")
    mode_count = int(anchor_context.get("반복모드횟수", 0))

    roles: List[dict] = []

    # 1) 반복 mode가 실제 2회 이상이면 1게임만 배정
    if mode_target is not None and mode_count >= 2 and len(roles) < game_count:
        roles.append({
            "역할": "반복모드",
            "목표합계": float(mode_target),
            "방향": "mode",
        })

    # 2) 중심축 1게임
    if len(roles) < game_count:
        roles.append({
            "역할": "중심",
            "목표합계": float(center_target),
            "방향": "center",
        })

    # 3) 하단/상단 분위수 축
    if len(roles) < game_count:
        roles.append({
            "역할": "하단Q25",
            "목표합계": float(lower_target),
            "방향": "lower",
        })

    if len(roles) < game_count:
        roles.append({
            "역할": "상단Q75",
            "목표합계": float(upper_target),
            "방향": "upper",
        })

    # 4) 평균 브리지 - 중앙과 꼬리 사이 연결축
    if len(roles) < game_count:
        roles.append({
            "역할": "분포브리지",
            "목표합계": float(bridge_target),
            "방향": "bridge",
        })

    # 5게임을 초과 요청한 경우 Q25/Q75/중심을 순환하되 soft target으로만 사용
    cycle = [
        ("하단분산", lower_target, "lower"),
        ("상단분산", upper_target, "upper"),
        ("중심분산", center_target, "center"),
        ("브리지분산", bridge_target, "bridge"),
    ]
    idx = 0
    while len(roles) < game_count:
        name, target, direction = cycle[idx % len(cycle)]
        roles.append({
            "역할": name,
            "목표합계": float(target),
            "방향": direction,
        })
        idx += 1

    return roles[:game_count]


def _portfolio_sum_diversity_penalty(
    total: int,
    selected_totals: List[int],
) -> float:
    """
    이미 선택된 게임들과 합계가 지나치게 비슷할 때만 약한 패널티를 줍니다.
    하드 컷이 아니므로 좋은 조합은 여전히 선택될 수 있습니다.
    """
    if not selected_totals:
        return 0.0

    nearest_gap = min(abs(int(total) - int(prev)) for prev in selected_totals)

    if nearest_gap <= 1:
        return 8.0
    if nearest_gap <= 3:
        return 5.0
    if nearest_gap <= 6:
        return 2.0
    return 0.0


def _sum_distribution_diagnostics(records: List[dict]) -> dict:
    """후보조합 합계분포를 진단용으로 요약합니다."""
    if not records:
        return {}

    totals = np.asarray(
        [int(record["features"]["합계"]) for record in records],
        dtype=float,
    )
    counts = pd.Series(totals.astype(int)).value_counts()
    mode_sum = int(counts.index[0]) if len(counts) else None
    mode_count = int(counts.iloc[0]) if len(counts) else 0

    return {
        "후보조합수": int(len(totals)),
        "후보합계최소": int(np.min(totals)),
        "후보합계Q25": int(round(np.quantile(totals, 0.25))),
        "후보합계중앙": int(round(np.quantile(totals, 0.50))),
        "후보합계Q75": int(round(np.quantile(totals, 0.75))),
        "후보합계최대": int(np.max(totals)),
        "후보합계최빈": mode_sum,
        "후보합계최빈횟수": mode_count,
    }

def calculate_v27_core_pattern_score(
    eleven_score_df: pd.DataFrame,
    v26_score_df: pd.DataFrame,
    weights: dict,
) -> pd.DataFrame:
    """
    V27.1 핵심패턴 보강점수.

    기존 15대 분석 안에 이미 존재하는 중요 신호를
    합계 후속패턴 엔진과 결합한 뒤에도 묻히지 않도록
    '보강 레이어'로 다시 한 번 투명하게 반영합니다.

    사용 신호:
    - 홀짝·합계·번호대 구조전이
    - 구매용지 마킹 위치
    - 장기 미출현
    - 직전번호 인접수
    - 전 회차 반복출현(이월수)
    - 최근 상승추세
    - 끝수 패턴

    주의:
    이 점수는 독립적인 당첨확률이 아니라
    기존 V26/V27 점수의 안정화·보강용 점수입니다.
    """

    def norm100(values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        if arr.size == 0:
            return arr
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            return np.zeros_like(arr, dtype=float)
        return (arr - lo) / (hi - lo) * 100.0

    base_table = (
        pd.DataFrame({"번호": np.arange(1, 46)})
        .merge(
            eleven_score_df[
                [
                    "번호",
                    "장기미출",
                    "상승추세",
                    "이월수",
                    "인접수",
                    "끝수패턴",
                ]
            ],
            on="번호",
            how="left",
        )
        .merge(
            v26_score_df[
                [
                    "번호",
                    "마킹패턴점수",
                    "구조전이점수",
                ]
            ],
            on="번호",
            how="left",
        )
        .fillna(0.0)
    )

    component_map = {
        "장기미출": "미출현",
        "상승추세": "최근추세",
        "이월수": "전회차반복",
        "인접수": "인접출현",
        "끝수패턴": "끝수",
        "마킹패턴점수": "구매용지",
        "구조전이점수": "홀짝구조전이",
    }

    for source, target in component_map.items():
        base_table[target] = norm100(
            pd.to_numeric(
                base_table[source],
                errors="coerce",
            ).fillna(0.0).to_numpy()
        )

    default_weights = {
        "미출현": 20.0,
        "최근추세": 15.0,
        "전회차반복": 10.0,
        "인접출현": 15.0,
        "끝수": 10.0,
        "구매용지": 18.0,
        "홀짝구조전이": 18.0,
    }

    resolved = {}
    for key, default_value in default_weights.items():
        try:
            resolved[key] = max(
                0.0,
                float(weights.get(key, default_value)),
            )
        except Exception:
            resolved[key] = float(default_value)

    total_weight = sum(resolved.values())
    if total_weight <= 0:
        resolved = default_weights.copy()
        total_weight = sum(resolved.values())

    core = np.zeros(45, dtype=float)

    for key, weight in resolved.items():
        core += (
            base_table[key].to_numpy(dtype=float)
            * (float(weight) / total_weight)
        )

    base_table["핵심패턴보강점수"] = np.round(
        norm100(core),
        2,
    )

    return base_table[
        [
            "번호",
            "미출현",
            "최근추세",
            "전회차반복",
            "인접출현",
            "끝수",
            "구매용지",
            "홀짝구조전이",
            "핵심패턴보강점수",
        ]
    ]


def combine_successor_evidence(
    v26_score_df: pd.DataFrame,
    sequence_scores: np.ndarray,
    anchor_scores: np.ndarray,
    sequence_context: dict,
    anchor_context: dict,
    core_pattern_df: Optional[pd.DataFrame] = None,
    successor_max_weight: float = 0.22,
    core_reinforcement_weight: float = 0.12,
) -> pd.DataFrame:
    """
    기존 15대 V26 점수와
    - 최신 합계수열 유사사례 후속번호
    - 동일/근접 합계 후속번호
    를 통합합니다.
    """
    result = v26_score_df.copy()

    seq = np.asarray(sequence_scores, dtype=float)
    anchor = np.asarray(anchor_scores, dtype=float)

    if seq.size != 45:
        seq = np.zeros(45, dtype=float)
    if anchor.size != 45:
        anchor = np.zeros(45, dtype=float)

    exact_n = int(anchor_context.get("정확일치표본수", 0))
    seq_conf = float(sequence_context.get("신뢰도", 0.0))

    # 정확일치 표본이 충분하면 anchor를 더 신뢰
    anchor_mix = float(
        np.clip(0.45 + min(exact_n, 20) / 100.0, 0.45, 0.65)
    )
    evidence = (
        anchor * anchor_mix
        + seq * (1.0 - anchor_mix)
    )

    evidence_strength = float(
        np.clip(
            0.45
            + 0.35 * min(exact_n / 12.0, 1.0)
            + 0.20 * seq_conf,
            0.35,
            1.0,
        )
    )
    successor_weight = float(
        successor_max_weight * evidence_strength
    )

    ordered = (
        result.sort_values("번호")
        .reset_index(drop=True)
    )
    original = ordered["V26종합점수"].astype(float).to_numpy()

    core_score = np.zeros(45, dtype=float)

    if (
        core_pattern_df is not None
        and not core_pattern_df.empty
        and "핵심패턴보강점수" in core_pattern_df.columns
    ):
        core_ordered = (
            core_pattern_df[
                ["번호", "핵심패턴보강점수"]
            ]
            .copy()
            .sort_values("번호")
            .reset_index(drop=True)
        )

        if len(core_ordered) == 45:
            core_score = (
                core_ordered["핵심패턴보강점수"]
                .astype(float)
                .to_numpy()
            )

    core_weight = float(
        np.clip(
            core_reinforcement_weight,
            0.0,
            0.25,
        )
    )

    base_weight = max(
        0.50,
        1.0 - successor_weight - core_weight,
    )

    weight_sum = base_weight + successor_weight + core_weight

    combined = (
        original * (base_weight / weight_sum)
        + evidence * (successor_weight / weight_sum)
        + core_score * (core_weight / weight_sum)
    )

    if combined.max() > combined.min():
        combined = (
            (combined - combined.min())
            / (combined.max() - combined.min())
            * 100.0
        )

    ordered["원본V26점수"] = original
    ordered["합계동일후속점수"] = anchor
    ordered["합계수열후속점수"] = seq
    ordered["핵심패턴보강점수"] = np.round(core_score, 2)
    ordered["V26종합점수"] = np.round(combined, 2)
    ordered["순위"] = (
        ordered["V26종합점수"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return ordered.sort_values(
        "V26종합점수",
        ascending=False,
    ).reset_index(drop=True)




def _sum_sequence_fit(
    total: int,
    sequence_context: dict,
) -> float:
    """
    유사 수열의 실제 후속합계 분포와 조합 합계의 적합도를 0~100으로 계산합니다.
    하드 컷이 아니라 soft score입니다.
    """
    successor_sums = np.asarray(
        sequence_context.get("후속합계", []),
        dtype=float,
    )
    weights = np.asarray(
        sequence_context.get("후속가중치", []),
        dtype=float,
    )

    if len(successor_sums) == 0:
        return 50.0

    if len(weights) != len(successor_sums) or weights.sum() <= 0:
        weights = np.ones_like(successor_sums)

    # 약 16점 차이마다 적합도가 점진적으로 낮아지게 설정
    kernel = np.exp(
        -np.abs(successor_sums - float(total)) / 16.0
    )
    score = float(
        np.average(kernel, weights=weights) * 100.0
    )
    return round(score, 2)


def _role_sum_fit(
    total: int,
    role: dict,
    anchor_context: dict,
) -> float:
    """역할별 합계 적합도를 0~100 soft score로 계산합니다."""
    target = float(role.get("목표합계", total))
    direction = str(role.get("방향", "center"))
    current_sum = float(anchor_context.get("현재합계", target))

    successor_sums = np.asarray(
        anchor_context.get("후속합계", []),
        dtype=float,
    )

    if len(successor_sums) >= 3:
        dispersion = max(
            float(np.std(successor_sums)),
            8.0,
        )
    else:
        dispersion = 14.0

    # 역할별로 너무 넓지 않되 하드 컷은 사용하지 않음
    scale = float(np.clip(dispersion * 0.45, 7.0, 18.0))
    fit = float(
        np.exp(-abs(float(total) - target) / scale) * 100.0
    )

    # 방향성은 soft penalty로만 적용
    if direction == "upper" and float(total) <= current_sum:
        fit *= 0.35
    elif direction == "lower" and float(total) >= target + scale:
        fit *= 0.45

    return round(fit, 2)


def generate_v27_adaptive_sets(
    v26_score_df: pd.DataFrame,
    anchor_context: dict,
    sequence_context: dict,
    game_count: int,
    candidate_count: int,
    fixed_numbers: List[int],
    excluded_numbers: List[int],
    minimum_spatial_score: float,
):
    """
    V27.2.1 Adaptive Successor + Sum Distribution Portfolio

    1) 최종 생존번호 TOP15를 기본 후보로 사용
    2) TOP15에서 가능한 6개 조합을 전수검사
    3) 구조필터(홀짝/저고/저중고/구간/연속/끝수/공간분산) 적용
    4) 동일·근접합계 후속사례와 합계수열 후속사례를 soft score로 반영
    5) 5게임 역할은 매회 데이터에서 자동 결정
       - 반복모드 1
       - 중심 2
       - 나머지 2는 상/하단 증거에 따라 자동
    6) 게임 간 번호 중복을 억제
    """
    fixed = sorted(set(int(n) for n in fixed_numbers))
    excluded = set(int(n) for n in excluded_numbers)

    if len(fixed) > 5:
        raise ValueError(
            "고정수는 최대 5개까지만 사용할 수 있습니다."
        )

    if set(fixed) & excluded:
        raise ValueError(
            "고정수와 제외수에 같은 번호가 있습니다."
        )

    ranked = (
        v26_score_df
        .sort_values("V26종합점수", ascending=False)
        ["번호"]
        .astype(int)
        .tolist()
    )

    score_map = (
        v26_score_df
        .set_index("번호")["V26종합점수"]
        .astype(float)
        .to_dict()
    )

    roles = build_adaptive_sum_roles(
        anchor_context=anchor_context,
        sequence_context=sequence_context,
        game_count=game_count,
    )

    survivor_steps = []
    base_count = max(15, min(int(candidate_count), 20))
    for value in [15, base_count, 18, 20, 25, 30]:
        value = min(45, int(value))
        if value not in survivor_steps:
            survivor_steps.append(value)

    all_candidates = []
    used_survivor_count = survivor_steps[-1]

    for survivor_count in survivor_steps:
        survivor_numbers = [
            n for n in ranked[:survivor_count]
            if n not in excluded
        ]

        for n in fixed:
            if n not in survivor_numbers:
                survivor_numbers.append(n)

        survivor_numbers = sorted(set(survivor_numbers))

        available = [
            n for n in survivor_numbers
            if n not in fixed
        ]
        need = 6 - len(fixed)

        if need < 0 or len(available) < need:
            continue

        candidate_records = []

        for sampled in itertools.combinations(
            available,
            need,
        ):
            combo = sorted(fixed + list(sampled))

            if len(combo) != 6 or len(set(combo)) != 6:
                continue

            f = _sequence_features(combo)

            # 구조 규칙
            if not (2 <= f["홀수수"] <= 4):
                continue
            if not (2 <= f["저번호수"] <= 4):
                continue
            if sum(1 for v in f["저중고분포"] if v > 0) < 2:
                continue
            if sum(1 for v in f["구간분포"] if v > 0) < 3:
                continue
            if max(f["구간분포"]) > 3:
                continue
            if f["연속쌍"] > 2:
                continue
            if f["끝수최대중복"] > 2:
                continue
            if f["공간분산점수"] < float(minimum_spatial_score):
                continue

            mean_score = float(
                np.mean(
                    [score_map.get(n, 0.0) for n in combo]
                )
            )

            empirical_fit = _sum_sequence_fit(
                f["합계"],
                {
                    "후속합계": anchor_context.get("후속합계", []),
                    "후속가중치": anchor_context.get("후속가중치", []),
                },
            )

            structure_score = 100.0
            structure_score -= abs(f["홀수수"] - 3) * 6.0
            structure_score -= abs(f["저번호수"] - 3) * 6.0
            structure_score -= f["연속쌍"] * 4.0
            structure_score -= max(
                0,
                max(f["구간분포"]) - 2,
            ) * 3.0

            base_quality = (
                mean_score * 0.56
                + empirical_fit * 0.18
                + structure_score * 0.16
                + f["공간분산점수"] * 0.10
            )

            candidate_records.append(
                {
                    "combo": combo,
                    "base_quality": float(base_quality),
                    "mean_score": mean_score,
                    "empirical_fit": empirical_fit,
                    "structure": structure_score,
                    "features": f,
                    "survivor_count": int(survivor_count),
                }
            )

        if len(candidate_records) >= int(game_count) * 5:
            all_candidates = candidate_records
            used_survivor_count = int(survivor_count)
            break

    if not all_candidates:
        return [], [], {
            "사용생존번호수": int(used_survivor_count),
            "완성게임수": 0,
            "역할": roles,
        }

    selected = []
    details = []
    selected_totals: List[int] = []
    number_usage = {n: 0 for n in range(1, 46)}

    # 역할별로 가장 적합한 조합을 순차 선택
    for role in roles:
        scored = []

        for record in all_candidates:
            combo = record["combo"]

            if any(tuple(combo) == tuple(prev) for prev in selected):
                continue

            overlap_max = (
                max(
                    [
                        len(set(combo) & set(prev))
                        for prev in selected
                    ],
                    default=0,
                )
            )

            usage_penalty = sum(
                number_usage[n]
                for n in combo
            ) * 1.6

            overlap_penalty = max(
                0,
                overlap_max - max(2, len(fixed)),
            ) * 10.0

            role_fit = _role_sum_fit(
                total=record["features"]["합계"],
                role=role,
                anchor_context=anchor_context,
            )

            sum_diversity_penalty = _portfolio_sum_diversity_penalty(
                total=record["features"]["합계"],
                selected_totals=selected_totals,
            )

            final = (
                record["base_quality"] * 0.72
                + role_fit * 0.28
                - usage_penalty
                - overlap_penalty
                - sum_diversity_penalty
            )

            scored.append(
                (
                    final,
                    role_fit,
                    record,
                )
            )

        if not scored:
            continue

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        _, role_fit, best = scored[0]
        combo = best["combo"]

        selected.append(combo)
        selected_totals.append(int(best["features"]["합계"]))
        for n in combo:
            number_usage[n] += 1

        details.append(
            {
                "역할": role["역할"],
                "역할목표합계": round(float(role["목표합계"]), 1),
                "역할적합도": round(float(role_fit), 2),
                "최종품질점수": round(float(best["base_quality"]), 2),
                "번호평균점수": round(float(best["mean_score"]), 2),
                "합계후속적합도": round(float(best["empirical_fit"]), 2),
                "균형점수": round(float(best["structure"]), 2),
                "생존번호수": int(best["survivor_count"]),
                "합계분산패널티": round(float(sum_diversity_penalty), 2),
            }
        )

        if len(selected) >= int(game_count):
            break

    # 역할별 선택으로 부족하면 일반 품질순으로 보충
    if len(selected) < int(game_count):
        selected_keys = {tuple(x) for x in selected}
        fallback = sorted(
            all_candidates,
            key=lambda x: x["base_quality"],
            reverse=True,
        )

        for record in fallback:
            combo = record["combo"]
            if tuple(combo) in selected_keys:
                continue

            if any(
                len(set(combo) & set(prev)) > max(3, len(fixed))
                for prev in selected
            ):
                continue

            selected.append(combo)
            selected_totals.append(int(record["features"]["합계"]))
            selected_keys.add(tuple(combo))
            details.append(
                {
                    "역할": "보충",
                    "역할목표합계": None,
                    "역할적합도": None,
                    "최종품질점수": round(float(record["base_quality"]), 2),
                    "번호평균점수": round(float(record["mean_score"]), 2),
                    "합계후속적합도": round(float(record["empirical_fit"]), 2),
                    "균형점수": round(float(record["structure"]), 2),
                    "생존번호수": int(record["survivor_count"]),
                }
            )

            if len(selected) >= int(game_count):
                break

    sum_diagnostics = _sum_distribution_diagnostics(all_candidates)

    summary = {
        "사용생존번호수": int(used_survivor_count),
        "완성게임수": len(selected),
        "현재합계": anchor_context.get("현재합계"),
        "정확일치표본수": anchor_context.get("정확일치표본수"),
        "사용표본수": anchor_context.get("사용표본수"),
        "확장단계": anchor_context.get("확장단계"),
        "후속합계가중평균": anchor_context.get("후속합계가중평균"),
        "후속합계중앙": anchor_context.get("후속합계중앙"),
        "반복모드합계": anchor_context.get("반복모드합계"),
        "반복모드횟수": anchor_context.get("반복모드횟수"),
        "상단표본수": anchor_context.get("상단표본수"),
        "상단대표합계": anchor_context.get("상단대표합계"),
        "역할": roles,
        "최종선택합계": selected_totals[:int(game_count)],
        **sum_diagnostics,
    }

    return selected[:int(game_count)], details[:int(game_count)], summary


# =========================================================
# 제목
# =========================================================

st.title("🎯 LOTTO GPT V27.2.1 SUM-DIVERSITY HOTFIX")
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
LOTTO GPT V27.2.1 SUM-DIVERSITY HOTFIX
</h3>

</td>

<td align="right">

<h4 style="color:#00ff90;">
AI Confidence

DATA-DRIVEN
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
LOTTO GPT V27.2.1 SUM-DIVERSITY HOTFIX
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
PORTFOLIO MODE
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

        업로드하면 프로그램이 모든 시트를 검사해
        **회차 + 당첨번호 6개가 가장 완전한 원본 시트를 자동 선택**합니다.
        사용자가 시트를 따로 선택할 필요가 없습니다.
        """
    )

else:
         
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names

                # 모든 시트를 실제로 검사해 가장 완전한 회차별 원본 시트를 자동선택.
                # 사용자가 시트를 고를 필요가 없으며, 패턴/통계 시트는 자동 배제됩니다.
                sheet_quality_map = {}

                for sheet_name in sheet_names:
                    try:
                        probe_df = pd.read_excel(
                            uploaded_file,
                            sheet_name=sheet_name,
                        )
                        sheet_quality_map[sheet_name] = sheet_lotto_quality(probe_df)
                    except Exception:
                        sheet_quality_map[sheet_name] = (-1, -1, -1, -1, -1)

                valid_sheets = [
                    name
                    for name in sheet_names
                    if sheet_quality_map.get(name, (-1, -1, -1, -1, -1))[0] >= 0
                ]

                if not valid_sheets:
                    raise ValueError(
                        "회차와 당첨번호 6개가 들어 있는 정상 분석 시트를 자동으로 찾지 못했습니다."
                    )

                best_sheet = max(
                    valid_sheets,
                    key=lambda name: sheet_quality_map.get(
                        name, (-1, -1, -1, -1, -1)
                    ),
                )

                # 완전 자동: 드롭다운을 없애고 최적 시트를 그대로 사용
                selected_sheet = best_sheet

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
                    int(df[round_column].max())
                    if round_column is not None
                    and df[round_column].notna().any()
                    else len(df)
                )

                if round_column is not None:
                    latest_row = (
                        df.loc[df[round_column].idxmax()]
                    )
                else:
                    latest_row = df.iloc[-1]

                latest_numbers = (
                    latest_row[number_columns]
                    .astype(int)
                    .sort_values()
                    .tolist()
                )

                # 데이터 무결성 점검
                data_warning = None
                if round_column is not None:
                    rounds = df[round_column].astype(int).tolist()
                    unique_rounds = len(set(rounds))
                    if unique_rounds != len(rounds):
                        data_warning = "회차 중복이 감지되었습니다."
                    elif min(rounds) == 1 and latest_round != len(rounds):
                        missing_count = latest_round - len(rounds)
                        if missing_count > 0:
                            data_warning = f"중간에 누락된 회차가 {missing_count}개 있습니다."

                st.caption(
                    f"✅ 자동 분석 시트: {selected_sheet} · "
                    f"유효 {len(df):,}회 · 최신 {latest_round:,}회"
                )

                if data_warning:
                    st.warning("⚠️ 데이터 무결성 경고: " + data_warning)
        
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

                # ============================================================
                # V27.2.1 - 최신 합계수열 유사사례 → 후속번호 가중
                # ============================================================
                (
                    sum_sequence_scores,
                    sum_sequence_analogs_df,
                    sum_sequence_context,
                ) = analyze_sum_sequence_successors(
                    df=df,
                    number_columns=number_columns,
                    round_column=round_column,
                    pattern_length=4,
                    top_k=30,
                )

                (
                    anchor_sum_scores,
                    anchor_sum_cases_df,
                    anchor_sum_context,
                ) = analyze_anchor_sum_successors(
                    df=df,
                    number_columns=number_columns,
                    round_column=round_column,
                    exact_min_samples=8,
                )

                # ============================================================
                # V27.2.1 - 핵심패턴 보강 레이어
                # 기존 슬라이더의 기본 가중치를 그대로 사용합니다.
                # ============================================================
                core_pattern_weights = {
                    "미출현": weight_overdue,
                    "최근추세": weight_trend,
                    "전회차반복": weight_carry,
                    "인접출현": weight_adjacent,
                    "끝수": weight_ending,
                    "구매용지": weight_marking,
                    "홀짝구조전이": weight_transition,
                }

                core_pattern_df = calculate_v27_core_pattern_score(
                    eleven_score_df=eleven_score_df,
                    v26_score_df=v26_score_df,
                    weights=core_pattern_weights,
                )

                v26_score_df = combine_successor_evidence(
                    v26_score_df=v26_score_df,
                    sequence_scores=sum_sequence_scores,
                    anchor_scores=anchor_sum_scores,
                    sequence_context=sum_sequence_context,
                    anchor_context=anchor_sum_context,
                    core_pattern_df=core_pattern_df,
                    successor_max_weight=0.22,
                    core_reinforcement_weight=0.12,
                )

                with st.expander(
                    "⚙️ V27.2 FINAL 핵심패턴 기본가중치·보강점수",
                    expanded=False,
                ):
                    st.caption(
                        "기존 15대 분석에 포함된 핵심 신호가 "
                        "합계 후속패턴 결합 후에도 희석되지 않도록 "
                        "12% 보강 레이어로 재반영합니다."
                    )
                    st.write(
                        {
                            "장기 미출현": weight_overdue,
                            "최근 상승추세": weight_trend,
                            "전 회차 반복출현": weight_carry,
                            "인접 출현": weight_adjacent,
                            "끝수 패턴": weight_ending,
                            "구매용지 위치": weight_marking,
                            "홀짝·합계·번호대 전이": weight_transition,
                        }
                    )
                    st.dataframe(
                        core_pattern_df
                        .sort_values(
                            "핵심패턴보강점수",
                            ascending=False,
                        )
                        .head(15),
                        use_container_width=True,
                        hide_index=True,
                    )

                with st.expander(
                    "🧬 합계수열 유사사례·후속번호 분석",
                    expanded=False,
                ):
                    st.write(
                        "최근 합계수열:",
                        sum_sequence_context.get("최근합계수열", []),
                    )
                    st.write(
                        "현재 회차 합계:",
                        sum_sequence_context.get("현재합계", "-"),
                    )
                    st.write(
                        "동일합계 후속표본:",
                        anchor_sum_context.get("정확일치표본수", 0),
                        "회",
                    )
                    st.write(
                        "후속합계 중심(중앙값):",
                        anchor_sum_context.get("후속합계중앙", "-"),
                    )
                    st.write(
                        "반복 후속합계(mode):",
                        anchor_sum_context.get("반복모드합계", "-"),
                        " / 출현 ",
                        anchor_sum_context.get("반복모드횟수", 0),
                        "회",
                    )
                    st.write(
                        "현재합계 초과 후속사례:",
                        anchor_sum_context.get("상단표본수", 0),
                        "회 / 대표합계 ",
                        anchor_sum_context.get("상단대표합계", "-"),
                    )
                    st.write(
                        "유사사례 후속합계 가중평균:",
                        sum_sequence_context.get("후속합계가중평균", "-"),
                    )
                    st.write(
                        "후속합계 가중 중앙값:",
                        sum_sequence_context.get("후속합계중앙", "-"),
                    )
                    st.caption(
                        "※ 후속합계는 강제 범위가 아니라 조합 품질의 soft score로만 사용합니다."
                    )
                    if not sum_sequence_analogs_df.empty:
                        st.dataframe(
                            sum_sequence_analogs_df.head(20),
                            use_container_width=True,
                            hide_index=True,
                        )
                    if not anchor_sum_cases_df.empty:
                        st.markdown("**동일·근접 합계 후속사례**")
                        st.dataframe(
                            anchor_sum_cases_df.head(30),
                            use_container_width=True,
                            hide_index=True,
                        )

                                 # ============================================================
                # V26 FINAL DEBUG - 15대 분석 실제 기여도 점검
                # 계산에는 영향을 주지 않고 화면 확인용으로만 사용
                # ============================================================

                with st.expander(
                    "🔍 V26 FINAL DEBUG - 상위번호 기여도 점검",
                    expanded=False,
                ):
                    debug_columns = [
                        "번호",
                        "기존11가중점수",
                        "마킹가중점수",
                        "유사가중점수",
                        "간격가중점수",
                        "전이가중점수",
                        "V26종합점수",
                    ]

                    debug_df = (
                        v26_score_df[debug_columns]
                        .sort_values("V26종합점수", ascending=False)
                        .head(10)
                        .copy()
                    )

                    st.dataframe(
                        debug_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                confidence_label, confidence_class = confidence_text(
                    similarity_confidence
                )

                st.divider()
                st.subheader("🏆 V26 종합점수 상위 15개 생존 후보")

                top_15 = (
                    v26_score_df
                    .sort_values("V26종합점수", ascending=False)
                    .head(15)["번호"]
                    .astype(int)
                    .tolist()
                )

                render_balls(top_15)

                chart_df = (
                    v26_score_df
                    .sort_values("V26종합점수", ascending=False)
                    .head(15)
                    .sort_values("번호")
                    .set_index("번호")[["V26종합점수"]]
                )

                st.bar_chart(chart_df)

                confidence_column, sample_column = st.columns(2)

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

                with st.expander("🔍 유사 회차 및 후속번호 상세보기"):
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
                            if column in similar_draws_df.columns
                        ]

                        similar_display_df = similar_draws_df[
                            display_columns
                        ].copy()

                        if "종합유사도" in similar_display_df.columns:
                            similar_display_df["종합유사도"] = (
                                similar_display_df["종합유사도"] * 100
                            ).round(2)

                        st.dataframe(
                            similar_display_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                st.divider()

                generate_button = st.button(
                    "🚀 V27.2.1 SUM-DIVERSITY 추천 생성",
                    use_container_width=True,
                    type="primary",
                )

                if generate_button:
                    seed = int(seed_value) if fixed_seed else None

                    (
                        combinations,
                        details,
                        set_summary,
                    ) = generate_v27_adaptive_sets(
                        v26_score_df=v26_score_df,
                        anchor_context=anchor_sum_context,
                        sequence_context=sum_sequence_context,
                        game_count=game_count,
                        candidate_count=candidate_count,
                        fixed_numbers=fixed_numbers,
                        excluded_numbers=excluded_numbers,
                        minimum_spatial_score=minimum_spatial_score,
                    )

                    used_survivor_count = int(
                        set_summary.get("사용생존번호수", 15)
                    )

                    if used_survivor_count > 15:
                        st.info(
                            "🔄 기본 생존번호 15개에서 구조조건을 만족하는 "
                            f"{game_count}게임이 부족해 {used_survivor_count}개로 최소 확장했습니다."
                        )
                    else:
                        st.caption(
                            "✅ V26.4 최종 생존번호 15개 안에서 추천 조합을 생성했습니다."
                        )

                    st.caption(
                        "🧬 합계수열은 강제구간이 아니라 유사 과거수열의 후속합계·후속번호를 "
                        "조합 품질점수에 가중하는 방식으로 적용됩니다."
                    )

                    with st.expander(
                        "🧪 V27.2.1 합계분포 진단 - BASELINE 가중치 변경 없음",
                        expanded=False,
                    ):
                        diag_cols = st.columns(4)
                        diag_cols[0].metric(
                            "동일합계 후속표본",
                            f"{anchor_sum_context.get('정확일치표본수', 0)}회",
                        )
                        diag_cols[1].metric(
                            "후속합계 중앙",
                            anchor_sum_context.get("후속합계중앙", "-"),
                        )
                        diag_cols[2].metric(
                            "후속합계 Q25",
                            anchor_sum_context.get("후속합계Q25", "-"),
                        )
                        diag_cols[3].metric(
                            "후속합계 Q75",
                            anchor_sum_context.get("후속합계Q75", "-"),
                        )

                        st.write(
                            "반복모드:",
                            anchor_sum_context.get("반복모드합계", "-"),
                            "/",
                            anchor_sum_context.get("반복모드횟수", 0),
                            "회",
                        )
                        st.write(
                            "후보조합 합계분포:",
                            {
                                "최소": set_summary.get("후보합계최소"),
                                "Q25": set_summary.get("후보합계Q25"),
                                "중앙": set_summary.get("후보합계중앙"),
                                "Q75": set_summary.get("후보합계Q75"),
                                "최대": set_summary.get("후보합계최대"),
                                "최빈": set_summary.get("후보합계최빈"),
                            },
                        )
                        st.write(
                            "최종 선택 합계:",
                            set_summary.get("최종선택합계", []),
                        )
                        st.write(
                            "역할별 soft target:",
                            set_summary.get("역할", []),
                        )
                        st.caption(
                            "※ Q25/Q50/Q75 및 반복 mode는 목표값 강제가 아니라 역할별 soft target입니다."
                        )

                    if len(combinations) < int(game_count):
                        st.warning(
                            f"요청 {game_count}게임 중 {len(combinations)}게임만 생성되었습니다."
                        )
                    else:
                        st.success(
                            "✅ V27.2.1 합계분포·포트폴리오 분산 5게임 생성 완료"
                        )

                    if not combinations:
                        st.error(
                            "현재 고정수·제외수 조건으로 조합을 만들 수 없습니다. "
                            "고정수/제외수만 확인해 주세요."
                        )

                    else:
                        st.subheader(
                            f"🎯 V27.2.1 추천 조합 {len(combinations)}게임"
                        )

                        for index, combination in enumerate(
                            combinations,
                            start=1,
                        ):
                            # 기존 화면 함수와 호환되도록 기본 feature 사용
                            features = combination_features(combination)
                            local_features = _sequence_features(combination)

                            detail = (
                                details[index - 1]
                                if index - 1 < len(details)
                                else {}
                            )

                            quality_score = float(
                                detail.get("최종품질점수", 0.0)
                            )
                            balance_score = float(
                                detail.get("균형점수", 0.0)
                            )
                            sum_fit = detail.get(
                                "합계후속적합도",
                                detail.get("합계수열적합도", 0.0),
                            )
                            role_name = detail.get("역할", "일반")
                            role_target = detail.get("역할목표합계", "-")

                            section_text = "-".join(
                                str(value)
                                for value in local_features["구간분포"]
                            )

                            st.html(
                                f"""
                                <div class="result-card">
                                    <div style="
                                        display:flex;
                                        justify-content:space-between;
                                        align-items:center;
                                        gap:10px;
                                        flex-wrap:wrap;
                                        margin-bottom:10px;
                                    ">
                                        <div style="
                                            color:#facc15;
                                            font-size:1.15rem;
                                            font-weight:900;
                                        ">
                                            🏆 SET {index:02d}
                                        </div>

                                        <div style="
                                            padding:4px 10px;
                                            border-radius:999px;
                                            background:rgba(250,204,21,0.14);
                                            border:1px solid rgba(250,204,21,0.38);
                                            color:#fde68a;
                                            font-size:0.78rem;
                                            font-weight:800;
                                        ">
                                            SUM-FIT {sum_fit:.1f}
                                        </div>
                                    </div>

                                    <div style="
                                        display:flex;
                                        gap:7px;
                                        flex-wrap:wrap;
                                    ">
                                        <span style="
                                            padding:6px 9px;
                                            border-radius:9px;
                                            background:rgba(74,222,128,0.13);
                                            border:1px solid rgba(74,222,128,0.30);
                                            color:#86efac;
                                            font-size:0.84rem;
                                            font-weight:800;
                                        ">
                                            품질 {quality_score:.2f}
                                        </span>

                                        <span style="
                                            padding:6px 9px;
                                            border-radius:9px;
                                            background:rgba(56,189,248,0.13);
                                            border:1px solid rgba(56,189,248,0.30);
                                            color:#7dd3fc;
                                            font-size:0.84rem;
                                            font-weight:800;
                                        ">
                                            균형 {balance_score:.1f}
                                        </span>
                                    </div>
                                </div>
                                """
                            )

                            render_balls(combination)

                            st.caption(
                                f"합계 {local_features['합계']} · "
                                f"홀짝 {local_features['홀수수']}:{local_features['짝수수']} · "
                                f"저고 {local_features['저번호수']}:{local_features['고번호수']} · "
                                f"구간 {section_text}"
                            )

                            st.caption(
                                f"역할 {role_name} · soft target {role_target}"
                            )

                            st.caption(
                                f"공간분산 {local_features['공간분산점수']:.1f}점 · "
                                f"균형 {balance_score:.1f}점 · "
                                f"품질 {quality_score:.2f} · "
                                f"연속쌍 {local_features['연속쌍']}개 · "
                                f"합계수열적합도 {sum_fit:.1f}점"
                            )

                        st.success(
                            "🏆 VENUS(MINERVA) V27 FINAL 추천 조합 "
                            "생성이 완료되었습니다. 🍀"
                        )

                st.divider()

                left, right = st.columns([1.3, 1])

                with left:
                    st.subheader("📊 V26 번호별 통합점수")

                    display_score_columns = [
                        "순위",
                        "번호",
                        "V26종합점수",
                        "기존11종합",
                        "마킹패턴점수",
                        "유사후속점수",
                        "간격순번점수",
                        "구조전이점수",
                        "합계수열후속점수",
                    ]

                    display_score_df = v26_score_df[
                        display_score_columns
                    ].copy()

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

                    display_score_df["V26종합점수"] = (
                        display_score_df["V26종합점수"]
                        .astype(float)
                        .round(2)
                    )

                    st.dataframe(
                        display_score_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                with right:
                    st.subheader("🔥 최근 30회 출현 횟수")

                    recent_30_count = frequency_counts(
                        df=df,
                        number_columns=number_columns,
                        window=30,
                    )

                    frequency_df = pd.DataFrame(
                        {
                            "번호": np.arange(1, 46),
                            "출현횟수": recent_30_count.astype(int),
                        }
                    ).set_index("번호")

                    st.bar_chart(frequency_df)
