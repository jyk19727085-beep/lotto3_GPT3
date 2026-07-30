from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.patterns import (
    extract_draws,
    marking_frequency_scores,
    normalize,
)

from engine.similarity import (
    similarity_followup_scores,
)


# =========================================================
# LOTTO GPT V26.0
# 15대 분석가설 통합점수 엔진
#
# 기존 11개 분석점수
# + 12. 구매용지 마킹패턴
# + 13. 유사 회차 후속 출현
# + 14. 번호간격·순번 위치
# + 15. 홀짝·합계·번호대 전이
# =========================================================


NUMBER_MIN = 1
NUMBER_MAX = 45


V26_SCORE_NAMES = [
    "최근5회",
    "최근10회",
    "최근30회",
    "최근100회",
    "장기미출",
    "최근추세",
    "이월수",
    "인접수",
    "끝수",
    "최근50회",
    "전체빈도",
    "마킹패턴",
    "유사후속",
    "간격순번",
    "구조전이",
]


DEFAULT_V26_WEIGHTS = [
    1.00,  # 최근 5회
    1.10,  # 최근 10회
    1.10,  # 최근 30회
    0.90,  # 최근 100회
    1.00,  # 장기 미출
    1.10,  # 최근 추세
    0.90,  # 이월수
    1.00,  # 인접수
    0.90,  # 끝수
    0.90,  # 최근 50회
    0.80,  # 전체 빈도
    1.05,  # 구매용지 패턴
    1.10,  # 유사 회차 후속
    1.00,  # 번호 간격·순번
    1.00,  # 홀짝·합계·구간 전이
]


def safe_normalize(values: Sequence[float]) -> np.ndarray:
    """결측값과 무한값을 처리한 뒤 0~1로 정규화합니다."""

    array = np.asarray(values, dtype=float)

    array = np.where(
        np.isfinite(array),
        array,
        0.0,
    )

    return normalize(array)


def prepare_base_score_table(
    base_score_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    기존 11대 분석결과를 V26 통합형식으로 변환합니다.

    지원 형식:
    - 번호 열 + 종합점수 열
    - 번호 열 + 점수 열
    - 1~45 순서의 데이터프레임
    """

    if base_score_df is None or base_score_df.empty:
        raise ValueError(
            "기존 11대 분석 점수표가 비어 있습니다."
        )

    result = base_score_df.copy()

    number_candidates = [
        "번호",
        "number",
        "Number",
        "NUM",
        "num",
    ]

    number_column = next(
        (
            column
            for column in number_candidates
            if column in result.columns
        ),
        None,
    )

    if number_column is None:
        if len(result) != 45:
            raise ValueError(
                "기존 점수표에서 번호 열을 찾을 수 없습니다."
            )

        result.insert(
            0,
            "번호",
            np.arange(1, 46),
        )

        number_column = "번호"

    result[number_column] = pd.to_numeric(
        result[number_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[number_column]
    )

    result[number_column] = (
        result[number_column].astype(int)
    )

    result = result[
        result[number_column].between(1, 45)
    ]

    result = (
        result.drop_duplicates(
            subset=[number_column],
            keep="last",
        )
        .set_index(number_column)
        .reindex(range(1, 46))
        .reset_index()
    )

    result = result.rename(
        columns={number_column: "번호"}
    )

    score_candidates = [
        "종합점수",
        "최종점수",
        "총점",
        "점수",
        "score",
        "Score",
    ]

    score_column = next(
        (
            column
            for column in score_candidates
            if column in result.columns
        ),
        None,
    )

    if score_column is None:
        numeric_columns = [
            column
            for column in result.columns
            if column != "번호"
            and pd.api.types.is_numeric_dtype(
                result[column]
            )
        ]

        if not numeric_columns:
            raise ValueError(
                "기존 점수표에서 점수 열을 찾을 수 없습니다."
            )

        result["기존11종합"] = (
            result[numeric_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(0)
            .mean(axis=1)
        )

    else:
        result["기존11종합"] = pd.to_numeric(
            result[score_column],
            errors="coerce",
        ).fillna(0)

    result["기존11종합"] = safe_normalize(
        result["기존11종합"].to_numpy()
    )

    return result


def position_frequency_scores(
    draws: np.ndarray,
    recent_window: int = 150,
) -> np.ndarray:
    """
    당첨번호 1P~6P의 순번별 위치분포를 분석합니다.

    최신 회차의 각 순번 주변에서 과거에 자주 등장한 번호에
    높은 점수를 부여합니다.
    """

    draws = np.asarray(draws, dtype=int)

    if (
        draws.ndim != 2
        or draws.shape[1] != 6
        or len(draws) < 10
    ):
        return np.zeros(45, dtype=float)

    recent = draws[
        -min(len(draws), int(recent_window)):
    ]

    latest = recent[-1]

    scores = np.zeros(45, dtype=float)

    for position in range(6):
        position_values = recent[:, position]

        center = int(latest[position])

        for number in range(1, 46):
            historical_frequency = np.sum(
                position_values == number
            )

            distance = abs(number - center)

            proximity = max(
                0.0,
                1.0 - distance / 12.0,
            )

            scores[number - 1] += (
                0.65 * historical_frequency
                + 0.35 * proximity
            )

    return safe_normalize(scores)


def gap_transition_scores(
    draws: np.ndarray,
    recent_window: int = 300,
) -> np.ndarray:
    """
    최신 회차의 번호 간격과 유사한 과거 회차를 찾고,
    그 다음 회차에서 등장한 번호를 집계합니다.
    """

    draws = np.asarray(draws, dtype=int)

    if (
        draws.ndim != 2
        or draws.shape[1] != 6
        or len(draws) < 20
    ):
        return np.zeros(45, dtype=float)

    recent = draws[
        -min(len(draws), int(recent_window)):
    ]

    latest = recent[-1]
    latest_gaps = np.diff(latest)

    raw_scores = np.zeros(45, dtype=float)
    total_weight = 0.0

    for index in range(len(recent) - 2):
        historical = recent[index]
        following = recent[index + 1]

        historical_gaps = np.diff(historical)

        mean_gap_difference = float(
            np.mean(
                np.abs(
                    latest_gaps
                    - historical_gaps
                )
            )
        )

        range_difference = abs(
            (latest[-1] - latest[0])
            - (
                historical[-1]
                - historical[0]
            )
        )

        gap_similarity = max(
            0.0,
            1.0 - mean_gap_difference / 12.0,
        )

        range_similarity = max(
            0.0,
            1.0 - range_difference / 35.0,
        )

        similarity = (
            0.75 * gap_similarity
            + 0.25 * range_similarity
        )

        if similarity < 0.45:
            continue

        weight = similarity ** 2
        total_weight += weight

        for number in following:
            raw_scores[int(number) - 1] += weight

    if total_weight <= 0:
        return np.zeros(45, dtype=float)

    return safe_normalize(raw_scores)


def interval_position_scores(
    draws: np.ndarray,
) -> np.ndarray:
    """
    번호간격 모델과 1P~6P 순번 위치 모델을 합칩니다.
    """

    gap_scores = gap_transition_scores(draws)
    position_scores = position_frequency_scores(draws)

    combined = (
        0.58 * gap_scores
        + 0.42 * position_scores
    )

    return safe_normalize(combined)


def draw_structure(
    numbers: Sequence[int],
) -> Dict[str, object]:
    """회차의 홀짝·합계·번호구간 구조를 계산합니다."""

    clean_numbers = sorted(
        int(number)
        for number in numbers
    )

    odd_count = sum(
        number % 2 == 1
        for number in clean_numbers
    )

    low_count = sum(
        number <= 22
        for number in clean_numbers
    )

    section_counts = np.asarray(
        [
            sum(
                1 <= number <= 10
                for number in clean_numbers
            ),
            sum(
                11 <= number <= 20
                for number in clean_numbers
            ),
            sum(
                21 <= number <= 30
                for number in clean_numbers
            ),
            sum(
                31 <= number <= 40
                for number in clean_numbers
            ),
            sum(
                41 <= number <= 45
                for number in clean_numbers
            ),
        ],
        dtype=float,
    )

    return {
        "홀수수": odd_count,
        "저번호수": low_count,
        "합계": sum(clean_numbers),
        "구간분포": section_counts,
    }


def structure_similarity(
    first: Dict[str, object],
    second: Dict[str, object],
) -> float:
    """두 회차의 홀짝·합계·번호대 구조 유사도를 계산합니다."""

    odd_similarity = max(
        0.0,
        1.0
        - abs(
            int(first["홀수수"])
            - int(second["홀수수"])
        ) / 6.0,
    )

    low_similarity = max(
        0.0,
        1.0
        - abs(
            int(first["저번호수"])
            - int(second["저번호수"])
        ) / 6.0,
    )

    sum_similarity = max(
        0.0,
        1.0
        - abs(
            int(first["합계"])
            - int(second["합계"])
        ) / 170.0,
    )

    first_sections = np.asarray(
        first["구간분포"],
        dtype=float,
    )

    second_sections = np.asarray(
        second["구간분포"],
        dtype=float,
    )

    section_difference = float(
        np.mean(
            np.abs(
                first_sections
                - second_sections
            )
        )
    )

    section_similarity = max(
        0.0,
        1.0 - section_difference / 2.0,
    )

    return (
        0.26 * odd_similarity
        + 0.20 * low_similarity
        + 0.24 * sum_similarity
        + 0.30 * section_similarity
    )


def structure_transition_scores(
    draws: np.ndarray,
    recent_window: int = 400,
) -> np.ndarray:
    """
    최신 홀짝·합계·번호대 구조와 유사한 과거 회차의
    다음 회차 번호를 가중 집계합니다.
    """

    draws = np.asarray(draws, dtype=int)

    if (
        draws.ndim != 2
        or draws.shape[1] != 6
        or len(draws) < 20
    ):
        return np.zeros(45, dtype=float)

    recent = draws[
        -min(len(draws), int(recent_window)):
    ]

    latest_structure = draw_structure(
        recent[-1]
    )

    raw_scores = np.zeros(45, dtype=float)
    accepted_count = 0

    for index in range(len(recent) - 2):
        historical_structure = draw_structure(
            recent[index]
        )

        similarity = structure_similarity(
            latest_structure,
            historical_structure,
        )

        if similarity < 0.58:
            continue

        following_draw = recent[index + 1]

        weight = similarity ** 2
        accepted_count += 1

        for number in following_draw:
            raw_scores[int(number) - 1] += weight

    if accepted_count == 0:
        return np.zeros(45, dtype=float)

    normalized = safe_normalize(raw_scores)

    reliability = min(
        1.0,
        accepted_count / 25.0,
    )

    return normalized * reliability


def calculate_v26_scores(
    df: pd.DataFrame,
    number_columns: Sequence[str],
    base_score_df: pd.DataFrame,
    weights: Optional[Sequence[float]] = None,
    marking_window: int = 100,
    similarity_top_k: int = 30,
    minimum_similarity: float = 0.45,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    float,
]:
    """
    기존 11대 점수와 V26 신규 4개 점수를 통합합니다.

    반환:
    1. 번호별 V26 통합 점수표
    2. 유사 과거회차 상세표
    3. 유사후속 분석 신뢰도
    """

    if weights is None:
        weights = DEFAULT_V26_WEIGHTS

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if weights.size != 15:
        raise ValueError(
            "V26 가중치는 정확히 15개여야 합니다."
        )

    base_table = prepare_base_score_table(
        base_score_df
    )

    draws = extract_draws(
        df,
        number_columns,
    )

    if len(draws) == 0:
        raise ValueError(
            "유효한 당첨번호 데이터를 찾지 못했습니다."
        )

    marking_scores = marking_frequency_scores(
        df=df,
        number_columns=number_columns,
        window=marking_window,
    )

    (
        similar_scores,
        similar_draws_df,
        similarity_confidence,
    ) = similarity_followup_scores(
        df=df,
        number_columns=number_columns,
        top_k=similarity_top_k,
        minimum_similarity=minimum_similarity,
    )

    interval_scores = interval_position_scores(
        draws
    )

    transition_scores = (
        structure_transition_scores(draws)
    )

    result = pd.DataFrame(
        {
            "번호": np.arange(1, 46),
            "기존11종합": safe_normalize(
                base_table["기존11종합"]
            ),
            "마킹패턴점수": safe_normalize(
                marking_scores
            ),
            "유사후속점수": safe_normalize(
                similar_scores
            ),
            "간격순번점수": safe_normalize(
                interval_scores
            ),
            "구조전이점수": safe_normalize(
                transition_scores
            ),
        }
    )

    existing_weight_total = float(
        np.sum(weights[:11])
    )

    additional_weight_total = float(
        np.sum(weights[11:])
    )

    total_weight = (
        existing_weight_total
        + additional_weight_total
    )

    if total_weight <= 0:
        raise ValueError(
            "가중치 합계는 0보다 커야 합니다."
        )

    result["기존11가중점수"] = (
        result["기존11종합"]
        * existing_weight_total
    )

    result["마킹가중점수"] = (
        result["마킹패턴점수"]
        * weights[11]
    )

    result["유사가중점수"] = (
        result["유사후속점수"]
        * weights[12]
    )

    result["간격가중점수"] = (
        result["간격순번점수"]
        * weights[13]
    )

    result["전이가중점수"] = (
        result["구조전이점수"]
        * weights[14]
    )

    result["V26종합점수"] = (
        result[
            [
                "기존11가중점수",
                "마킹가중점수",
                "유사가중점수",
                "간격가중점수",
                "전이가중점수",
            ]
        ].sum(axis=1)
        / total_weight
    )

    result["V26종합점수"] = (
        safe_normalize(
            result["V26종합점수"]
        )
        * 100.0
    )

    result["순위"] = (
        result["V26종합점수"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    result = result.sort_values(
        ["순위", "번호"]
    ).reset_index(drop=True)

    return (
        result,
        similar_draws_df,
        similarity_confidence,
    )
