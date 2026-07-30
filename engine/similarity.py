from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.patterns import (
    draw_pattern_features,
    extract_draws,
    normalize,
)


# =========================================================
# LOTTO GPT V26.0
# 유사 당첨번호 이후 후속 출현 분석 엔진
# =========================================================


NUMBER_MIN = 1
NUMBER_MAX = 45


def draw_basic_features(
    numbers: Sequence[int],
) -> Dict[str, object]:
    """
    한 회차의 번호·간격·홀짝·구간·끝수·공간 특징을 계산합니다.
    """

    clean_numbers = sorted(
        set(
            int(number)
            for number in numbers
            if NUMBER_MIN <= int(number) <= NUMBER_MAX
        )
    )

    if len(clean_numbers) != 6:
        raise ValueError(
            "유사도 계산에는 유효한 번호 6개가 필요합니다."
        )

    gaps = [
        clean_numbers[index + 1] - clean_numbers[index]
        for index in range(5)
    ]

    odd_count = sum(
        number % 2 == 1
        for number in clean_numbers
    )

    low_count = sum(
        number <= 22
        for number in clean_numbers
    )

    section_counts = [
        sum(1 <= number <= 10 for number in clean_numbers),
        sum(11 <= number <= 20 for number in clean_numbers),
        sum(21 <= number <= 30 for number in clean_numbers),
        sum(31 <= number <= 40 for number in clean_numbers),
        sum(41 <= number <= 45 for number in clean_numbers),
    ]

    ending_counts = [0] * 10

    for number in clean_numbers:
        ending_counts[number % 10] += 1

    spatial = draw_pattern_features(clean_numbers)

    return {
        "번호": clean_numbers,
        "번호집합": set(clean_numbers),
        "합계": sum(clean_numbers),
        "홀수수": odd_count,
        "저번호수": low_count,
        "구간분포": np.asarray(section_counts, dtype=float),
        "끝수분포": np.asarray(ending_counts, dtype=float),
        "간격": np.asarray(gaps, dtype=float),
        "첫끝범위": clean_numbers[-1] - clean_numbers[0],
        "공간특징": spatial,
    }


def scaled_difference(
    first: float,
    second: float,
    scale: float,
) -> float:
    """
    두 값의 차이를 0~1 유사도로 변환합니다.
    """

    if scale <= 0:
        return 0.0

    difference = abs(float(first) - float(second))

    return max(
        0.0,
        1.0 - (difference / scale),
    )


def vector_similarity(
    first: np.ndarray,
    second: np.ndarray,
    scale: float,
) -> float:
    """
    두 숫자 벡터의 평균 절대차를 이용해
    0~1 유사도를 계산합니다.
    """

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    if first.shape != second.shape:
        return 0.0

    if first.size == 0:
        return 0.0

    mean_difference = float(
        np.mean(np.abs(first - second))
    )

    return max(
        0.0,
        1.0 - (mean_difference / scale),
    )


def exact_and_adjacent_similarity(
    target_numbers: Sequence[int],
    historical_numbers: Sequence[int],
) -> Tuple[float, int, int]:
    """
    동일번호와 ±1 인접번호의 일치 정도를 계산합니다.

    반환:
    - 유사도
    - 동일번호 개수
    - 인접번호 개수
    """

    target_set = set(int(number) for number in target_numbers)
    historical_set = set(
        int(number)
        for number in historical_numbers
    )

    exact_count = len(
        target_set & historical_set
    )

    adjacent_count = 0

    for target in target_set:
        if (
            target - 1 in historical_set
            or target + 1 in historical_set
        ):
            adjacent_count += 1

    exact_score = exact_count / 6.0
    adjacent_score = adjacent_count / 6.0

    combined_score = (
        0.72 * exact_score
        + 0.28 * adjacent_score
    )

    return (
        min(1.0, combined_score),
        exact_count,
        adjacent_count,
    )


def spatial_similarity(
    first_features: Dict[str, float],
    second_features: Dict[str, float],
) -> float:
    """
    구매용지 공간패턴 특징의 유사도를 계산합니다.
    """

    comparisons = [
        scaled_difference(
            first_features["사용행수"],
            second_features["사용행수"],
            5,
        ),
        scaled_difference(
            first_features["사용열수"],
            second_features["사용열수"],
            5,
        ),
        scaled_difference(
            first_features["최대동일행"],
            second_features["최대동일행"],
            4,
        ),
        scaled_difference(
            first_features["최대동일열"],
            second_features["최대동일열"],
            4,
        ),
        scaled_difference(
            first_features["마킹면적"],
            second_features["마킹면적"],
            35,
        ),
        scaled_difference(
            first_features["전체인접"],
            second_features["전체인접"],
            6,
        ),
        scaled_difference(
            first_features["평균거리"],
            second_features["평균거리"],
            5,
        ),
        scaled_difference(
            first_features["상단수"],
            second_features["상단수"],
            6,
        ),
        scaled_difference(
            first_features["중단수"],
            second_features["중단수"],
            6,
        ),
        scaled_difference(
            first_features["하단수"],
            second_features["하단수"],
            6,
        ),
        scaled_difference(
            first_features["좌측수"],
            second_features["좌측수"],
            6,
        ),
        scaled_difference(
            first_features["중앙수"],
            second_features["중앙수"],
            6,
        ),
        scaled_difference(
            first_features["우측수"],
            second_features["우측수"],
            6,
        ),
    ]

    return float(np.mean(comparisons))


def calculate_draw_similarity(
    target_numbers: Sequence[int],
    historical_numbers: Sequence[int],
) -> Dict[str, float]:
    """
    최신 회차와 한 과거 회차의 종합 유사도를 계산합니다.

    구성:
    - 동일·인접 번호
    - 홀짝
    - 합계
    - 저고
    - 번호구간
    - 끝수
    - 번호간격
    - 첫수~끝수 범위
    - 마킹용지 공간패턴
    """

    target = draw_basic_features(target_numbers)
    historical = draw_basic_features(historical_numbers)

    number_score, exact_count, adjacent_count = (
        exact_and_adjacent_similarity(
            target["번호"],
            historical["번호"],
        )
    )

    odd_score = scaled_difference(
        target["홀수수"],
        historical["홀수수"],
        6,
    )

    sum_score = scaled_difference(
        target["합계"],
        historical["합계"],
        170,
    )

    low_high_score = scaled_difference(
        target["저번호수"],
        historical["저번호수"],
        6,
    )

    section_score = vector_similarity(
        target["구간분포"],
        historical["구간분포"],
        2.0,
    )

    ending_score = vector_similarity(
        target["끝수분포"],
        historical["끝수분포"],
        1.2,
    )

    gap_score = vector_similarity(
        target["간격"],
        historical["간격"],
        12.0,
    )

    range_score = scaled_difference(
        target["첫끝범위"],
        historical["첫끝범위"],
        40,
    )

    marking_score = spatial_similarity(
        target["공간특징"],
        historical["공간특징"],
    )

    total_similarity = (
        0.24 * number_score
        + 0.09 * odd_score
        + 0.10 * sum_score
        + 0.07 * low_high_score
        + 0.11 * section_score
        + 0.08 * ending_score
        + 0.11 * gap_score
        + 0.06 * range_score
        + 0.14 * marking_score
    )

    return {
        "종합유사도": float(
            max(0.0, min(1.0, total_similarity))
        ),
        "동일번호수": exact_count,
        "인접번호수": adjacent_count,
        "번호유사도": number_score,
        "홀짝유사도": odd_score,
        "합계유사도": sum_score,
        "저고유사도": low_high_score,
        "구간유사도": section_score,
        "끝수유사도": ending_score,
        "간격유사도": gap_score,
        "범위유사도": range_score,
        "마킹유사도": marking_score,
    }


def find_similar_draws(
    df: pd.DataFrame,
    number_columns: Sequence[str],
    top_k: int = 30,
    minimum_similarity: float = 0.45,
) -> pd.DataFrame:
    """
    가장 최근 회차와 유사한 과거 회차를 검색합니다.

    마지막 회차는 비교 대상에서 제외하고,
    후속 회차가 존재하는 과거 회차만 사용합니다.
    """

    draws = extract_draws(df, number_columns)

    if len(draws) < 3:
        return pd.DataFrame()

    target_draw = draws[-1]
    records: List[Dict[str, object]] = []

    for index in range(len(draws) - 2):
        historical_draw = draws[index]
        following_draw = draws[index + 1]

        similarity = calculate_draw_similarity(
            target_draw,
            historical_draw,
        )

        if (
            similarity["종합유사도"]
            < float(minimum_similarity)
        ):
            continue

        record: Dict[str, object] = {
            "과거데이터순번": index + 1,
            "과거번호": ", ".join(
                str(int(number))
                for number in historical_draw
            ),
            "후속번호": ", ".join(
                str(int(number))
                for number in following_draw
            ),
        }

        record.update(similarity)
        records.append(record)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    return (
        result.sort_values(
            "종합유사도",
            ascending=False,
        )
        .head(max(1, int(top_k)))
        .reset_index(drop=True)
    )


def similarity_followup_scores(
    df: pd.DataFrame,
    number_columns: Sequence[str],
    top_k: int = 30,
    minimum_similarity: float = 0.45,
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """
    유사한 과거 회차의 바로 다음 회차에서
    등장한 번호를 가중 집계합니다.

    반환:
    - 1~45번 유사후속 점수
    - 유사 회차 상세표
    - 분석 신뢰도 0~1

    표본이 적으면 신뢰도를 낮춰
    과적합을 완화합니다.
    """

    similar_df = find_similar_draws(
        df=df,
        number_columns=number_columns,
        top_k=top_k,
        minimum_similarity=minimum_similarity,
    )

    raw_scores = np.zeros(45, dtype=float)

    if similar_df.empty:
        return raw_scores, similar_df, 0.0

    for _, row in similar_df.iterrows():
        similarity = float(row["종합유사도"])

        if similarity >= 0.75:
            similarity_weight = 1.00
        elif similarity >= 0.62:
            similarity_weight = 0.70
        else:
            similarity_weight = 0.40

        following_numbers = [
            int(value.strip())
            for value in str(row["후속번호"]).split(",")
            if value.strip()
        ]

        for number in following_numbers:
            if NUMBER_MIN <= number <= NUMBER_MAX:
                raw_scores[number - 1] += (
                    similarity
                    * similarity_weight
                )

    normalized_scores = normalize(raw_scores)

    sample_count = len(similar_df)
    average_similarity = float(
        similar_df["종합유사도"].mean()
    )

    sample_reliability = min(
        1.0,
        sample_count / 20.0,
    )

    similarity_reliability = max(
        0.0,
        min(
            1.0,
            (average_similarity - minimum_similarity)
            / max(0.01, 1.0 - minimum_similarity),
        ),
    )

    confidence = (
        0.60 * sample_reliability
        + 0.40 * similarity_reliability
    )

    adjusted_scores = (
        normalized_scores
        * confidence
    )

    return (
        adjusted_scores,
        similar_df,
        round(float(confidence), 4),
    )


def top_followup_numbers(
    scores: np.ndarray,
    count: int = 15,
) -> List[Tuple[int, float]]:
    """
    유사 회차 후속점수가 높은 번호를 반환합니다.
    """

    scores = np.asarray(scores, dtype=float)

    if scores.size != 45:
        raise ValueError(
            "유사 후속점수 배열은 45개여야 합니다."
        )

    top_indices = np.argsort(scores)[::-1][:count]

    return [
        (
            int(index) + 1,
            round(float(scores[index]) * 100, 2),
        )
        for index in top_indices
    ]
