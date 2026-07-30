from typing import List

import numpy as np
import pandas as pd


def normalize(values: np.ndarray) -> np.ndarray:
    """
    숫자 배열을 0~1 범위로 변환합니다.
    모든 값이 같으면 0으로 반환합니다.
    """

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return np.zeros_like(values, dtype=float)

    if maximum == minimum:
        return np.zeros_like(values, dtype=float)

    return (values - minimum) / (maximum - minimum)


def frequency_counts(
    df: pd.DataFrame,
    number_columns: List[str],
    window: int,
) -> np.ndarray:
    """
    최근 window회 동안 1~45번의 출현 횟수를 계산합니다.
    """

    recent_df = df.tail(min(window, len(df)))
    counts = np.zeros(45, dtype=float)

    for column in number_columns:
        values = pd.to_numeric(
            recent_df[column],
            errors="coerce",
        ).dropna()

        for value in values:
            number = int(value)

            if 1 <= number <= 45:
                counts[number - 1] += 1

    return counts


def overdue_counts(
    df: pd.DataFrame,
    number_columns: List[str],
) -> np.ndarray:
    """
    각 번호가 마지막으로 출현한 뒤
    몇 회 동안 나오지 않았는지 계산합니다.
    """

    draws = df[number_columns].astype(int).to_numpy()
    overdue = np.zeros(45, dtype=float)

    for number in range(1, 46):
        gap = len(draws)

        for reverse_index, draw in enumerate(draws[::-1]):
            if number in draw:
                gap = reverse_index
                break

        overdue[number - 1] = gap

    return overdue


def carryover_scores(
    df: pd.DataFrame,
    number_columns: List[str],
) -> np.ndarray:
    """
    가장 최근 회차 번호에 이월수 점수를 부여합니다.
    직전 당첨번호는 1점, 나머지는 0점입니다.
    """

    scores = np.zeros(45, dtype=float)

    if df.empty:
        return scores

    latest_numbers = (
        df.iloc[-1][number_columns]
        .astype(int)
        .tolist()
    )

    for number in latest_numbers:
        if 1 <= number <= 45:
            scores[number - 1] = 1.0

    return scores


def adjacent_scores(
    df: pd.DataFrame,
    number_columns: List[str],
) -> np.ndarray:
    """
    직전 회차 번호의 앞수와 뒷수에 점수를 부여합니다.

    예:
    직전 번호가 20이면 19와 21에 점수를 부여합니다.
    """

    scores = np.zeros(45, dtype=float)

    if df.empty:
        return scores

    latest_numbers = (
        df.iloc[-1][number_columns]
        .astype(int)
        .tolist()
    )

    for number in latest_numbers:
        previous_number = number - 1
        next_number = number + 1

        if 1 <= previous_number <= 45:
            scores[previous_number - 1] += 1

        if 1 <= next_number <= 45:
            scores[next_number - 1] += 1

    return normalize(scores)


def ending_digit_scores(
    df: pd.DataFrame,
    number_columns: List[str],
    window: int = 30,
) -> np.ndarray:
    """
    최근 회차에서 자주 나온 끝수를 계산하고,
    같은 끝수를 가진 번호에 점수를 부여합니다.
    """

    recent_df = df.tail(min(window, len(df)))
    ending_counts = np.zeros(10, dtype=float)

    for column in number_columns:
        values = pd.to_numeric(
            recent_df[column],
            errors="coerce",
        ).dropna()

        for value in values:
            number = int(value)

            if 1 <= number <= 45:
                ending_counts[number % 10] += 1

    ending_scores = normalize(ending_counts)

    number_scores = np.zeros(45, dtype=float)

    for number in range(1, 46):
        number_scores[number - 1] = ending_scores[number % 10]

    return number_scores


def trend_scores(
    df: pd.DataFrame,
    number_columns: List[str],
) -> np.ndarray:
    """
    최근 10회 빈도와 이전 구간 빈도를 비교해
    최근 상승 추세를 계산합니다.
    """

    recent_10 = frequency_counts(
        df,
        number_columns,
        10,
    )

    recent_50 = frequency_counts(
        df,
        number_columns,
        50,
    )

    previous_40 = np.maximum(
        recent_50 - recent_10,
        0,
    )

    recent_window = max(1, min(10, len(df)))
    previous_window = max(
        1,
        min(40, max(len(df) - 10, 1)),
    )

    recent_rate = recent_10 / recent_window
    previous_rate = previous_40 / previous_window

    return normalize(recent_rate - previous_rate)


def calculate_eleven_scores(
    df: pd.DataFrame,
    number_columns: List[str],
    weights: List[int],
) -> pd.DataFrame:
    """
    11개 분석 가설을 가중합하여
    1~45번의 종합점수를 계산합니다.
    """

    if len(weights) != 11:
        raise ValueError("가중치는 정확히 11개가 필요합니다.")

    model_scores = [
        normalize(frequency_counts(df, number_columns, 5)),
        normalize(frequency_counts(df, number_columns, 10)),
        normalize(frequency_counts(df, number_columns, 30)),
        normalize(frequency_counts(df, number_columns, 100)),
        normalize(overdue_counts(df, number_columns)),
        trend_scores(df, number_columns),
        carryover_scores(df, number_columns),
        adjacent_scores(df, number_columns),
        ending_digit_scores(df, number_columns, 30),
        normalize(frequency_counts(df, number_columns, 50)),
        normalize(frequency_counts(df, number_columns, len(df))),
    ]

    weight_array = np.asarray(weights, dtype=float)

    if weight_array.sum() <= 0:
        weight_array = np.ones(11, dtype=float)

    weight_array = weight_array / weight_array.sum()

    final_score = np.zeros(45, dtype=float)

    for weight, score in zip(weight_array, model_scores):
        final_score += weight * score

    final_score = normalize(final_score) * 100

    result = pd.DataFrame(
        {
            "번호": np.arange(1, 46),
            "최근5회": np.round(model_scores[0] * 100, 2),
            "최근10회": np.round(model_scores[1] * 100, 2),
            "최근30회": np.round(model_scores[2] * 100, 2),
            "최근100회": np.round(model_scores[3] * 100, 2),
            "장기미출": np.round(model_scores[4] * 100, 2),
            "상승추세": np.round(model_scores[5] * 100, 2),
            "이월수": np.round(model_scores[6] * 100, 2),
            "인접수": np.round(model_scores[7] * 100, 2),
            "끝수패턴": np.round(model_scores[8] * 100, 2),
            "최근50회": np.round(model_scores[9] * 100, 2),
            "전체빈도": np.round(model_scores[10] * 100, 2),
            "종합점수": np.round(final_score, 2),
        }
    )

    return result.sort_values(
        "종합점수",
        ascending=False,
    ).reset_index(drop=True)
