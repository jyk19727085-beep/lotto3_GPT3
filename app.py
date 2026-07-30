import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from engine.scoring import calculate_eleven_scores

# =========================================================
# LOTTO GPT V25.4
# Excel 실제 데이터 기반 기초 분석 대시보드
# =========================================================

st.set_page_config(
    page_title="LOTTO GPT V25.4",
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
        margin: 10px 0 22px 0;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 공통 함수
# =========================================================

def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """배열을 0~1 범위로 변환합니다."""
    values = np.asarray(values, dtype=float)

    min_value = np.nanmin(values)
    max_value = np.nanmax(values)

    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return np.zeros_like(values, dtype=float)

    if max_value == min_value:
        return np.zeros_like(values, dtype=float)

    return (values - min_value) / (max_value - min_value)


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
    """번호를 로또공 형태로 화면에 표시합니다."""
    html = "<div class='lotto-row'>"

    for number in numbers:
        html += (
            f"<div class='lotto-ball' "
            f"style='background:{ball_color(number)};'>"
            f"{number}"
            f"</div>"
        )

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


def clean_column_name(column: object) -> str:
    """열 이름에서 공백과 줄바꿈을 제거합니다."""
    return (
        str(column)
        .strip()
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
    )


def detect_round_column(df: pd.DataFrame) -> Optional[str]:
    """회차 열로 보이는 열을 자동 탐색합니다."""
    round_keywords = ["회차", "회", "draw", "round"]

    for column in df.columns:
        cleaned = clean_column_name(column).lower()

        if any(keyword in cleaned for keyword in round_keywords):
            return column

    return None


def detect_number_columns(df: pd.DataFrame) -> List[str]:
    """
    당첨번호 6개 열을 자동 탐색합니다.

    우선순위:
    1. 번호1~번호6, 당첨번호1~6 형식
    2. 숫자 데이터가 대부분 1~45인 열
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
    ]

    preferred_keywords = [
        "번호",
        "당첨",
        "num",
        "number",
        "ball",
    ]

    # 1차: 열 이름으로 탐색
    for column in df.columns:
        cleaned = clean_column_name(column).lower()

        if any(keyword in cleaned for keyword in excluded_keywords):
            continue

        if any(keyword in cleaned for keyword in preferred_keywords):
            numeric = pd.to_numeric(df[column], errors="coerce")
            valid_ratio = numeric.between(1, 45).mean()

            if valid_ratio >= 0.5:
                candidates.append(column)

    if len(candidates) >= 6:
        return candidates[:6]

    # 2차: 데이터 범위로 탐색
    candidates = []

    for column in df.columns:
        cleaned = clean_column_name(column).lower()

        if any(keyword in cleaned for keyword in excluded_keywords):
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        non_null_count = int(numeric.notna().sum())

        if non_null_count == 0:
            continue

        valid_ratio = numeric.between(1, 45).mean()

        if valid_ratio >= 0.8:
            candidates.append(column)

    return candidates[:6]


def prepare_lotto_data(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    """엑셀 데이터를 정리하고 유효한 회차만 남깁니다."""

    df = raw_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    round_column = detect_round_column(df)
    number_columns = detect_number_columns(df)

    if len(number_columns) < 6:
        raise ValueError(
            "당첨번호 6개 열을 자동으로 찾지 못했습니다. "
            "엑셀 열 이름을 번호1, 번호2, 번호3, 번호4, 번호5, 번호6처럼 작성해 주세요."
        )

    for column in number_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=number_columns).copy()

    for column in number_columns:
        df[column] = df[column].astype(int)

    valid_mask = np.ones(len(df), dtype=bool)

    for column in number_columns:
        valid_mask &= df[column].between(1, 45)

    df = df.loc[valid_mask].copy()

    # 한 회차 안에 중복 번호가 있는 행 제거
    unique_mask = df[number_columns].nunique(axis=1) == 6
    df = df.loc[unique_mask].copy()

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

    if len(df) < 10:
        raise ValueError(
            "분석 가능한 유효 회차가 10개 미만입니다. "
            "엑셀 데이터 형식을 확인해 주세요."
        )

    return df, number_columns, round_column


def frequency_counts(
    df: pd.DataFrame,
    number_columns: List[str],
    window: int,
) -> np.ndarray:
    """최근 지정 회차 동안 번호별 출현 횟수를 계산합니다."""

    recent_df = df.tail(min(window, len(df)))
    counts = np.zeros(45, dtype=float)

    for column in number_columns:
        values = recent_df[column].astype(int).to_numpy()

        for number in values:
            counts[number - 1] += 1

    return counts


def overdue_scores(
    df: pd.DataFrame,
    number_columns: List[str],
) -> np.ndarray:
    """각 번호가 마지막으로 출현한 이후의 미출현 회차 수를 계산합니다."""

    draws = df[number_columns].astype(int).to_numpy()
    scores = np.zeros(45, dtype=float)

    for number in range(1, 46):
        gap = len(draws)

        for reverse_index, draw in enumerate(draws[::-1]):
            if number in draw:
                gap = reverse_index
                break

        scores[number - 1] = gap

    return scores


def trend_scores(
    df: pd.DataFrame,
    number_columns: List[str],
) -> np.ndarray:
    """
    최근 10회 빈도와 이전 40회 빈도의 차이를 이용해
    최근 상승 추세를 계산합니다.
    """

    recent_10 = frequency_counts(df, number_columns, 10)
    recent_50 = frequency_counts(df, number_columns, 50)

    previous_40 = np.maximum(recent_50 - recent_10, 0)

    recent_rate = recent_10 / max(10, min(10, len(df)))
    previous_rate = previous_40 / max(1, min(40, max(len(df) - 10, 1)))

    return recent_rate - previous_rate


def calculate_scores(
    df: pd.DataFrame,
    number_columns: List[str],
    weight_recent: int,
    weight_medium: int,
    weight_long: int,
    weight_overdue: int,
    weight_trend: int,
) -> pd.DataFrame:
    """5개 기초 가설을 합산해 1~45번의 종합점수를 계산합니다."""

    recent_10 = minmax_normalize(
        frequency_counts(df, number_columns, 10)
    )

    recent_30 = minmax_normalize(
        frequency_counts(df, number_columns, 30)
    )

    recent_100 = minmax_normalize(
        frequency_counts(df, number_columns, 100)
    )

    overdue = minmax_normalize(
        overdue_scores(df, number_columns)
    )

    trend = minmax_normalize(
        trend_scores(df, number_columns)
    )

    weights = np.array(
        [
            weight_recent,
            weight_medium,
            weight_long,
            weight_overdue,
            weight_trend,
        ],
        dtype=float,
    )

    if weights.sum() <= 0:
        weights = np.ones(5, dtype=float)

    weights = weights / weights.sum()

    final_score = (
        weights[0] * recent_10
        + weights[1] * recent_30
        + weights[2] * recent_100
        + weights[3] * overdue
        + weights[4] * trend
    )

    final_score = minmax_normalize(final_score) * 100

    result = pd.DataFrame(
        {
            "번호": np.arange(1, 46),
            "최근10회": np.round(recent_10 * 100, 2),
            "최근30회": np.round(recent_30 * 100, 2),
            "최근100회": np.round(recent_100 * 100, 2),
            "미출현": np.round(overdue * 100, 2),
            "상승추세": np.round(trend * 100, 2),
            "종합점수": np.round(final_score, 2),
        }
    )

    return result.sort_values(
        "종합점수",
        ascending=False,
    ).reset_index(drop=True)


def valid_combination(numbers: List[int]) -> bool:
    """지나치게 한쪽으로 쏠린 조합을 걸러냅니다."""

    numbers = sorted(numbers)

    odd_count = sum(number % 2 == 1 for number in numbers)
    total_sum = sum(numbers)
    low_count = sum(number <= 22 for number in numbers)

    consecutive_count = sum(
        numbers[index + 1] - numbers[index] == 1
        for index in range(5)
    )

    if odd_count not in [2, 3, 4]:
        return False

    if not 100 <= total_sum <= 180:
        return False

    if low_count not in [2, 3, 4]:
        return False

    if consecutive_count > 2:
        return False

    return True


def generate_combinations(
    score_df: pd.DataFrame,
    game_count: int,
    candidate_count: int,
    fixed_numbers: Optional[List[int]] = None,
    excluded_numbers: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    상위 후보군에서 점수 비례 방식으로 추천 조합을 생성합니다.

    fixed_numbers:
        모든 조합에 반드시 포함할 번호

    excluded_numbers:
        모든 조합에서 제외할 번호
    """

    fixed_numbers = sorted(set(fixed_numbers or []))
    excluded_numbers = sorted(set(excluded_numbers or []))

    if len(fixed_numbers) > 5:
        raise ValueError("고정수는 최대 5개까지 선택할 수 있습니다.")

    overlap = set(fixed_numbers) & set(excluded_numbers)

    if overlap:
        overlap_text = ", ".join(
            str(number)
            for number in sorted(overlap)
        )

        raise ValueError(
            f"고정수와 제외수에 같은 번호가 있습니다: {overlap_text}"
        )

    candidate_df = score_df.head(candidate_count).copy()

    candidate_df = candidate_df[
        ~candidate_df["번호"].isin(excluded_numbers)
    ].copy()

    # 고정수가 후보군 밖에 있더라도 조합에 사용할 수 있도록 추가
    missing_fixed = [
        number
        for number in fixed_numbers
        if number not in candidate_df["번호"].astype(int).tolist()
    ]

    if missing_fixed:
        fixed_rows = score_df[
            score_df["번호"].isin(missing_fixed)
        ]

        candidate_df = pd.concat(
            [candidate_df, fixed_rows],
            ignore_index=True,
        ).drop_duplicates(
            subset=["번호"],
            keep="first",
        )

    available_df = candidate_df[
        ~candidate_df["번호"].isin(fixed_numbers)
    ].copy()

    needed_count = 6 - len(fixed_numbers)

    if len(available_df) < needed_count:
        raise ValueError(
            "고정수와 제외수 설정 때문에 조합을 만들 번호가 부족합니다. "
            "제외수를 줄이거나 후보 번호 수를 늘려주세요."
        )

    candidate_numbers = (
        available_df["번호"]
        .astype(int)
        .to_numpy()
    )

    probabilities = (
        available_df["종합점수"]
        .astype(float)
        .to_numpy()
    )

    probabilities = np.clip(
        probabilities,
        0.01,
        None,
    )

    probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(seed)

    combinations: List[List[int]] = []
    attempts = 0
    max_attempts = 10000

    while (
        len(combinations) < game_count
        and attempts < max_attempts
    ):
        attempts += 1

        selected_extra = (
            rng.choice(
                candidate_numbers,
                size=needed_count,
                replace=False,
                p=probabilities,
            )
            .astype(int)
            .tolist()
        )

        selected = sorted(
            fixed_numbers + selected_extra
        )

        if not valid_combination(selected):
            continue

        if selected not in combinations:
            combinations.append(selected)

    # 조건이 너무 엄격해 부족하면 균형 조건만 완화해 추가
    while (
        len(combinations) < game_count
        and attempts < max_attempts * 2
    ):
        attempts += 1

        selected_extra = (
            rng.choice(
                candidate_numbers,
                size=needed_count,
                replace=False,
                p=probabilities,
            )
            .astype(int)
            .tolist()
        )

        selected = sorted(
            fixed_numbers + selected_extra
        )

        if selected not in combinations:
            combinations.append(selected)

    if len(combinations) < game_count:
        raise ValueError(
            "현재 조건으로 필요한 조합 수를 만들지 못했습니다. "
            "고정수 또는 제외수를 줄여주세요."
        )

    return combinations


# =========================================================
# 제목
# =========================================================

st.title("🎯 LOTTO GPT V25.4")
st.markdown(
    """
    <div class="notice-card">
    <b>실제 엑셀 회차 데이터를 이용한 기초 통계 분석 버전</b><br>
    최근 출현빈도, 장기 미출현, 최근 상승추세를 종합해 번호별 점수를 계산합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# =========================================================
# 사이드바 설정
# =========================================================

st.sidebar.header("📂 데이터 입력")

uploaded_file = st.sidebar.file_uploader(
    "로또 회차 엑셀 파일 업로드",
    type=["xlsx"],
)

st.sidebar.divider()

st.sidebar.header("⚙️ 11대 분석 가설")

weight_5 = st.sidebar.slider(
    "① 최근 5회 초단기 빈도",
    min_value=0,
    max_value=100,
    value=20,
)

weight_10 = st.sidebar.slider(
    "② 최근 10회 단기 빈도",
    min_value=0,
    max_value=100,
    value=30,
)

weight_30 = st.sidebar.slider(
    "③ 최근 30회 중기 빈도",
    min_value=0,
    max_value=100,
    value=25,
)

weight_100 = st.sidebar.slider(
    "④ 최근 100회 장기 빈도",
    min_value=0,
    max_value=100,
    value=15,
)

weight_overdue = st.sidebar.slider(
    "⑤ 장기 미출현 회귀",
    min_value=0,
    max_value=100,
    value=20,
)

weight_trend = st.sidebar.slider(
    "⑥ 최근 상승추세",
    min_value=0,
    max_value=100,
    value=15,
)

weight_carry = st.sidebar.slider(
    "⑦ 직전 회차 이월수",
    min_value=0,
    max_value=100,
    value=10,
)

weight_adjacent = st.sidebar.slider(
    "⑧ 직전 번호 인접수",
    min_value=0,
    max_value=100,
    value=15,
)

weight_ending = st.sidebar.slider(
    "⑨ 최근 끝수 패턴",
    min_value=0,
    max_value=100,
    value=10,
)

weight_50 = st.sidebar.slider(
    "⑩ 최근 50회 안정 빈도",
    min_value=0,
    max_value=100,
    value=15,
)

weight_all = st.sidebar.slider(
    "⑪ 전체 회차 누적 빈도",
    min_value=0,
    max_value=100,
    value=10,
)

st.sidebar.divider()

game_count = st.sidebar.select_slider(
    "추천 조합 수",
    options=[5, 10, 15, 20],
    value=5,
)

candidate_count = st.sidebar.slider(
    "후보 번호 수",
    min_value=12,
    max_value=25,
    value=18,
)
st.sidebar.divider()

st.sidebar.header("🎯 조합 세부 설정")

number_options = list(range(1, 46))

fixed_numbers = st.sidebar.multiselect(
    "고정수 선택 — 최대 5개",
    options=number_options,
    default=[],
    max_selections=5,
    help="선택한 번호는 생성되는 모든 조합에 포함됩니다.",
)

excluded_numbers = st.sidebar.multiselect(
    "제외수 선택",
    options=number_options,
    default=[],
    help="선택한 번호는 모든 추천 조합에서 제외됩니다.",
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
    value=252,
    disabled=not fixed_seed,
)

st.sidebar.caption(
    "가중치 합계는 자동으로 100% 비율로 환산됩니다."
)


# =========================================================
# 메인 실행
# =========================================================

if uploaded_file is None:
    st.info(
        "왼쪽의 Upload 버튼을 눌러 로또 회차별 엑셀 파일을 선택해 주세요."
    )

    st.markdown(
        """
        ### 권장 엑셀 구조

        | 회차 | 번호1 | 번호2 | 번호3 | 번호4 | 번호5 | 번호6 |
        |---:|---:|---:|---:|---:|---:|---:|
        | 1 | 10 | 23 | 29 | 33 | 37 | 40 |
        | 2 | 9 | 13 | 21 | 25 | 32 | 42 |

        열 이름이 달라도 숫자 범위를 검사하여 자동 탐색을 시도합니다.
        """
    )

else:
    try:
        excel_file = pd.ExcelFile(uploaded_file)

        selected_sheet = st.selectbox(
            "분석할 엑셀 시트 선택",
            options=excel_file.sheet_names,
        )

        raw_df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet,
        )

        df, number_columns, round_column = prepare_lotto_data(raw_df)

        latest_round = (
            int(df[round_column].dropna().max())
            if round_column is not None
            and df[round_column].notna().any()
            else len(df)
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        metric_1.metric("유효 회차", f"{len(df):,}회")
        metric_2.metric("최신 회차", f"{latest_round:,}회")
        metric_3.metric("인식 번호 열", "6개")
        metric_4.metric("추천 후보군", f"{candidate_count}개")

        with st.expander("📋 인식한 데이터 확인"):
            st.write("당첨번호 열:", number_columns)

            if round_column is not None:
                st.write("회차 열:", round_column)
            else:
                st.write("회차 열: 자동 탐색되지 않음")

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

        weights = [
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

        score_df = calculate_eleven_scores(
            df=df,
            number_columns=number_columns,
            weights=weights,
        )
        st.divider()

        st.subheader("🏆 종합점수 상위 15개 생존 후보")

        top_15 = (
            score_df.head(15)["번호"]
            .astype(int)
            .tolist()
        )

        render_balls(top_15)

        chart_df = (
            score_df.head(15)
            .sort_values("번호")
            .set_index("번호")[["종합점수"]]
        )

        st.bar_chart(chart_df)

        st.divider()

        if st.button(
            "🚀 데이터 기반 추천 조합 생성",
            use_container_width=True,
            type="primary",
        ):
            seed = int(seed_value) if fixed_seed else None

            combinations = generate_combinations(
                score_df=score_df,
                game_count=game_count,
                candidate_count=candidate_count,
                fixed_numbers=fixed_numbers,
                excluded_numbers=excluded_numbers,
                seed=seed,
            )

            st.subheader(f"🎯 추천 조합 {game_count}게임")

            for index, combination in enumerate(combinations, start=1):
                st.markdown(
                    f"<div class='result-card'><b>SET {index:02d}</b>",
                    unsafe_allow_html=True,
                )

                render_balls(combination)

                odd_count = sum(
                    number % 2 == 1
                    for number in combination
                )

                st.caption(
                    f"합계 {sum(combination)} · "
                    f"홀짝 {odd_count}:{6 - odd_count} · "
                    f"저고 {sum(number <= 22 for number in combination)}:"
                    f"{sum(number >= 23 for number in combination)}"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            st.success("추천 조합 생성이 완료되었습니다.")

        st.divider()

        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("📊 번호별 종합점수")

            st.dataframe(
                score_df,
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.subheader("🔥 최근 30회 출현 횟수")

            recent_30_count = frequency_counts(
                df,
                number_columns,
                30,
            )

            frequency_df = pd.DataFrame(
                {
                    "번호": np.arange(1, 46),
                    "출현횟수": recent_30_count.astype(int),
                }
            ).set_index("번호")

            st.bar_chart(frequency_df)

        st.warning(
            "이 결과는 과거 데이터의 통계적 탐색과 조합 생성을 위한 것입니다. "
            "무작위 추첨의 당첨을 예측하거나 보장하지 않습니다."
        )

    except Exception as error:
        st.error("엑셀 데이터를 분석하지 못했습니다.")
        st.exception(error)
