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
    .stApp {
        background:
            linear-gradient(
                rgba(15, 23, 42, 0.92),
                rgba(15, 23, 42, 0.97)
            ),
            url("https://images.unsplash.com/photo-1566041510394-cf7c8d049f17?q=80&w=1600&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 215, 0, 0.20);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label, div {
        word-break: keep-all;
    }

    .lotto-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin: 10px 0 18px 0;
    }

    .lotto-ball {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 52px;
        height: 52px;
        border-radius: 50%;
        color: white;
        font-weight: 900;
        font-size: 1.25rem;
        box-shadow:
            inset -4px -4px 7px rgba(0,0,0,0.35),
            2px 4px 8px rgba(0,0,0,0.35);
    }

    .result-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255, 215, 0, 0.28);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .notice-card {
        background: rgba(255, 215, 0, 0.09);
        border: 1px solid rgba(255, 215, 0, 0.25);
        border-radius: 12px;
        padding: 15px;
        color: #f8fafc;
    }

    .engine-card {
        background: rgba(30, 41, 59, 0.76);
        border-left: 5px solid #fbc02d;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0;
        color: #f8fafc;
    }

    .confidence-high {
        color: #4ade80;
        font-weight: 800;
    }

    .confidence-middle {
        color: #facc15;
        font-weight: 800;
    }

    .confidence-low {
        color: #fb7185;
        font-weight: 800;
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

st.title("🎯 LOTTO GPT V26.0")

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

        default_sheet_index = (
            sheet_names.index("번호별")
            if "번호별" in sheet_names
            else 0
        )

        selected_sheet = st.selectbox(
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
                        for value in features[
                            "구간분포"
                        ]
                    )

                    st.markdown(
                        (
                            "<div class='result-card'>"
                            f"<b>SET {index:02d}</b>"
                        ),
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
                    "V26 균형 추천 조합 생성이 완료되었습니다."
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
