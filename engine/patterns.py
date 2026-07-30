from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# LOTTO GPT V26.0
# 구매 마킹용지 공간패턴 분석 엔진
#
# 실제 로또 구매용지 배열:
#
#  1  2  3  4  5  6  7
#  8  9 10 11 12 13 14
# 15 16 17 18 19 20 21
# 22 23 24 25 26 27 28
# 29 30 31 32 33 34 35
# 36 37 38 39 40 41 42
# 43 44 45
# =========================================================


MARKING_COLUMNS = 7
NUMBER_MIN = 1
NUMBER_MAX = 45


def normalize(values: np.ndarray) -> np.ndarray:
    """숫자 배열을 0~1 범위로 정규화합니다."""

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


def number_to_position(number: int) -> Tuple[int, int]:
    """
    로또 번호를 구매용지의 행·열 좌표로 변환합니다.

    행과 열은 내부 계산을 위해 0부터 시작합니다.

    예:
    1  -> (0, 0)
    7  -> (0, 6)
    8  -> (1, 0)
    14 -> (1, 6)
    45 -> (6, 2)
    """

    number = int(number)

    if not NUMBER_MIN <= number <= NUMBER_MAX:
        raise ValueError(
            f"로또 번호는 1~45 사이여야 합니다: {number}"
        )

    zero_based = number - 1
    row = zero_based // MARKING_COLUMNS
    column = zero_based % MARKING_COLUMNS

    return row, column


def position_to_number(row: int, column: int) -> int:
    """
    구매용지 좌표를 로또 번호로 변환합니다.
    존재하지 않는 46~49 위치는 오류 처리합니다.
    """

    number = (
        int(row) * MARKING_COLUMNS
        + int(column)
        + 1
    )

    if not NUMBER_MIN <= number <= NUMBER_MAX:
        raise ValueError(
            f"해당 좌표에는 유효한 로또 번호가 없습니다: "
            f"행={row}, 열={column}"
        )

    return number


def extract_draws(
    df: pd.DataFrame,
    number_columns: Sequence[str],
) -> np.ndarray:
    """데이터프레임에서 유효한 당첨번호 배열을 추출합니다."""

    if len(number_columns) != 6:
        raise ValueError("당첨번호 열은 정확히 6개여야 합니다.")

    draw_df = df[list(number_columns)].copy()

    for column in number_columns:
        draw_df[column] = pd.to_numeric(
            draw_df[column],
            errors="coerce",
        )

    draw_df = draw_df.dropna().astype(int)

    valid_rows = []

    for row in draw_df.to_numpy():
        numbers = sorted(int(number) for number in row)

        if len(set(numbers)) != 6:
            continue

        if not all(
            NUMBER_MIN <= number <= NUMBER_MAX
            for number in numbers
        ):
            continue

        valid_rows.append(numbers)

    if not valid_rows:
        return np.empty((0, 6), dtype=int)

    return np.asarray(valid_rows, dtype=int)


def draw_positions(
    numbers: Sequence[int],
) -> List[Tuple[int, int]]:
    """한 회차 번호를 구매용지 좌표 목록으로 바꿉니다."""

    unique_numbers = sorted(
        set(int(number) for number in numbers)
    )

    return [
        number_to_position(number)
        for number in unique_numbers
        if NUMBER_MIN <= number <= NUMBER_MAX
    ]


def count_adjacencies(
    positions: Sequence[Tuple[int, int]],
) -> Dict[str, int]:
    """
    한 회차의 마킹 위치에서 인접 관계를 계산합니다.

    horizontal : 같은 행에서 좌우 한 칸
    vertical   : 같은 열에서 상하 한 칸
    diagonal   : 대각선 한 칸
    """

    position_set = set(positions)

    horizontal = 0
    vertical = 0
    diagonal = 0

    for row, column in position_set:
        if (row, column + 1) in position_set:
            horizontal += 1

        if (row + 1, column) in position_set:
            vertical += 1

        if (row + 1, column + 1) in position_set:
            diagonal += 1

        if (row + 1, column - 1) in position_set:
            diagonal += 1

    return {
        "가로인접": horizontal,
        "세로인접": vertical,
        "대각인접": diagonal,
        "전체인접": horizontal + vertical + diagonal,
    }


def draw_pattern_features(
    numbers: Sequence[int],
) -> Dict[str, float]:
    """
    한 회차의 구매용지 공간패턴 특징을 계산합니다.
    """

    clean_numbers = sorted(
        set(
            int(number)
            for number in numbers
            if NUMBER_MIN <= int(number) <= NUMBER_MAX
        )
    )

    if not clean_numbers:
        return {
            "사용행수": 0,
            "사용열수": 0,
            "최대동일행": 0,
            "최대동일열": 0,
            "마킹면적": 0,
            "행범위": 0,
            "열범위": 0,
            "가로인접": 0,
            "세로인접": 0,
            "대각인접": 0,
            "전체인접": 0,
            "평균거리": 0.0,
            "상단수": 0,
            "중단수": 0,
            "하단수": 0,
            "좌측수": 0,
            "중앙수": 0,
            "우측수": 0,
        }

    positions = draw_positions(clean_numbers)

    rows = [row for row, _ in positions]
    columns = [column for _, column in positions]

    row_counts = Counter(rows)
    column_counts = Counter(columns)

    minimum_row = min(rows)
    maximum_row = max(rows)
    minimum_column = min(columns)
    maximum_column = max(columns)

    row_span = maximum_row - minimum_row + 1
    column_span = maximum_column - minimum_column + 1
    marking_area = row_span * column_span

    adjacency = count_adjacencies(positions)

    pair_distances = []

    for first_index in range(len(positions)):
        for second_index in range(
            first_index + 1,
            len(positions),
        ):
            row_1, column_1 = positions[first_index]
            row_2, column_2 = positions[second_index]

            distance = (
                abs(row_1 - row_2)
                + abs(column_1 - column_2)
            )

            pair_distances.append(distance)

    average_distance = (
        float(np.mean(pair_distances))
        if pair_distances
        else 0.0
    )

    top_count = sum(row <= 1 for row in rows)
    middle_count = sum(2 <= row <= 4 for row in rows)
    bottom_count = sum(row >= 5 for row in rows)

    left_count = sum(column <= 1 for column in columns)
    center_count = sum(2 <= column <= 4 for column in columns)
    right_count = sum(column >= 5 for column in columns)

    return {
        "사용행수": len(row_counts),
        "사용열수": len(column_counts),
        "최대동일행": max(row_counts.values()),
        "최대동일열": max(column_counts.values()),
        "마킹면적": marking_area,
        "행범위": row_span,
        "열범위": column_span,
        "가로인접": adjacency["가로인접"],
        "세로인접": adjacency["세로인접"],
        "대각인접": adjacency["대각인접"],
        "전체인접": adjacency["전체인접"],
        "평균거리": round(average_distance, 4),
        "상단수": top_count,
        "중단수": middle_count,
        "하단수": bottom_count,
        "좌측수": left_count,
        "중앙수": center_count,
        "우측수": right_count,
    }


def build_pattern_table(
    df: pd.DataFrame,
    number_columns: Sequence[str],
) -> pd.DataFrame:
    """
    모든 회차의 구매용지 공간패턴을 표로 만듭니다.
    """

    draws = extract_draws(df, number_columns)

    records = []

    for index, draw in enumerate(draws):
        features = draw_pattern_features(draw)

        record = {
            "데이터순번": index + 1,
            "번호": ", ".join(
                str(int(number))
                for number in draw
            ),
        }

        record.update(features)
        records.append(record)

    return pd.DataFrame(records)


def marking_frequency_scores(
    df: pd.DataFrame,
    number_columns: Sequence[str],
    window: int = 100,
) -> np.ndarray:
    """
    최근 지정 회차의 구매용지 마킹 빈도를 이용하여
    1~45번의 공간패턴 점수를 계산합니다.

    구성:
    - 개별 번호 마킹 빈도
    - 해당 행 출현 빈도
    - 해당 열 출현 빈도
    - 주변 좌표 출현 빈도
    """

    draws = extract_draws(df, number_columns)

    if len(draws) == 0:
        return np.zeros(45, dtype=float)

    recent_draws = draws[-min(window, len(draws)):]

    cell_counts = np.zeros(45, dtype=float)
    row_counts = np.zeros(7, dtype=float)
    column_counts = np.zeros(7, dtype=float)
    neighborhood_counts = np.zeros(45, dtype=float)

    for draw in recent_draws:
        positions = draw_positions(draw)

        for number in draw:
            cell_counts[int(number) - 1] += 1

        for row, column in positions:
            row_counts[row] += 1
            column_counts[column] += 1

            neighbors = [
                (row, column - 1),
                (row, column + 1),
                (row - 1, column),
                (row + 1, column),
                (row - 1, column - 1),
                (row - 1, column + 1),
                (row + 1, column - 1),
                (row + 1, column + 1),
            ]

            for neighbor_row, neighbor_column in neighbors:
                if not (
                    0 <= neighbor_row <= 6
                    and 0 <= neighbor_column <= 6
                ):
                    continue

                neighbor_number = (
                    neighbor_row * MARKING_COLUMNS
                    + neighbor_column
                    + 1
                )

                if NUMBER_MIN <= neighbor_number <= NUMBER_MAX:
                    neighborhood_counts[
                        neighbor_number - 1
                    ] += 1

    normalized_cells = normalize(cell_counts)
    normalized_rows = normalize(row_counts)
    normalized_columns = normalize(column_counts)
    normalized_neighbors = normalize(neighborhood_counts)

    scores = np.zeros(45, dtype=float)

    for number in range(1, 46):
        row, column = number_to_position(number)

        scores[number - 1] = (
            0.45 * normalized_cells[number - 1]
            + 0.20 * normalized_rows[row]
            + 0.20 * normalized_columns[column]
            + 0.15 * normalized_neighbors[number - 1]
        )

    return normalize(scores)


def spatial_balance_score(
    numbers: Sequence[int],
) -> float:
    """
    하나의 추천 조합이 구매용지에 얼마나 적절히
    분산돼 있는지 0~100점으로 평가합니다.

    지나친 밀집과 지나친 인접은 감점됩니다.
    """

    clean_numbers = sorted(
        set(int(number) for number in numbers)
    )

    if len(clean_numbers) != 6:
        return 0.0

    features = draw_pattern_features(clean_numbers)

    score = 100.0

    if features["사용행수"] < 3:
        score -= 25

    if features["사용열수"] < 3:
        score -= 20

    if features["최대동일행"] >= 4:
        score -= 30
    elif features["최대동일행"] == 3:
        score -= 10

    if features["최대동일열"] >= 4:
        score -= 30
    elif features["최대동일열"] == 3:
        score -= 10

    if features["전체인접"] >= 5:
        score -= 30
    elif features["전체인접"] == 4:
        score -= 15

    if features["마킹면적"] < 12:
        score -= 25
    elif features["마킹면적"] < 18:
        score -= 10

    if (
        features["상단수"] >= 5
        or features["중단수"] >= 5
        or features["하단수"] >= 5
    ):
        score -= 20

    if (
        features["좌측수"] >= 5
        or features["중앙수"] >= 5
        or features["우측수"] >= 5
    ):
        score -= 20

    return round(
        max(0.0, min(100.0, score)),
        2,
    )


def valid_spatial_combination(
    numbers: Sequence[int],
    minimum_score: float = 55.0,
) -> bool:
    """
    추천번호 조합이 최소 공간분산 점수를
    만족하는지 검사합니다.
    """

    return (
        spatial_balance_score(numbers)
        >= float(minimum_score)
    )
