from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.patterns import (
    draw_pattern_features,
    spatial_balance_score,
)


# =========================================================
# LOTTO GPT V26.0
# 추천 조합 균형·다양성·숫자 쏠림 방지 엔진
# =========================================================


NUMBER_MIN = 1
NUMBER_MAX = 45
DRAW_SIZE = 6


def clean_combination(
    numbers: Sequence[int],
) -> List[int]:
    """추천번호를 검증하고 오름차순으로 정리합니다."""

    cleaned = sorted(
        set(
            int(number)
            for number in numbers
            if NUMBER_MIN <= int(number) <= NUMBER_MAX
        )
    )

    return cleaned


def section_index(number: int) -> int:
    """
    번호를 5개 구간으로 구분합니다.

    0: 1~10
    1: 11~20
    2: 21~30
    3: 31~40
    4: 41~45
    """

    number = int(number)

    if 1 <= number <= 10:
        return 0
    if 11 <= number <= 20:
        return 1
    if 21 <= number <= 30:
        return 2
    if 31 <= number <= 40:
        return 3
    if 41 <= number <= 45:
        return 4

    raise ValueError(f"유효하지 않은 번호입니다: {number}")


def section_counts(
    numbers: Sequence[int],
) -> List[int]:
    """조합의 번호구간 분포를 계산합니다."""

    counts = [0, 0, 0, 0, 0]

    for number in clean_combination(numbers):
        counts[section_index(number)] += 1

    return counts


def consecutive_pairs(
    numbers: Sequence[int],
) -> int:
    """서로 연속된 번호쌍 개수를 계산합니다."""

    cleaned = clean_combination(numbers)

    return sum(
        cleaned[index + 1] - cleaned[index] == 1
        for index in range(len(cleaned) - 1)
    )


def longest_consecutive_run(
    numbers: Sequence[int],
) -> int:
    """가장 긴 연속번호 길이를 계산합니다."""

    cleaned = clean_combination(numbers)

    if not cleaned:
        return 0

    longest = 1
    current = 1

    for index in range(1, len(cleaned)):
        if cleaned[index] == cleaned[index - 1] + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


def ending_digit_counts(
    numbers: Sequence[int],
) -> Counter:
    """끝수별 개수를 계산합니다."""

    return Counter(
        int(number) % 10
        for number in clean_combination(numbers)
    )


def low_high_counts(
    numbers: Sequence[int],
    low_maximum: int = 22,
) -> Tuple[int, int]:
    """저번호와 고번호 개수를 계산합니다."""

    cleaned = clean_combination(numbers)

    low_count = sum(
        number <= int(low_maximum)
        for number in cleaned
    )

    return low_count, len(cleaned) - low_count


def pairwise_distance_summary(
    numbers: Sequence[int],
) -> Dict[str, float]:
    """번호 간 거리의 기본 특징을 계산합니다."""

    cleaned = clean_combination(numbers)

    if len(cleaned) < 2:
        return {
            "최소간격": 0,
            "평균간격": 0.0,
            "최대간격": 0,
            "첫끝범위": 0,
        }

    gaps = [
        cleaned[index + 1] - cleaned[index]
        for index in range(len(cleaned) - 1)
    ]

    return {
        "최소간격": min(gaps),
        "평균간격": round(float(np.mean(gaps)), 4),
        "최대간격": max(gaps),
        "첫끝범위": cleaned[-1] - cleaned[0],
    }


def combination_features(
    numbers: Sequence[int],
) -> Dict[str, object]:
    """추천번호 한 조합의 전체 균형 특징을 계산합니다."""

    cleaned = clean_combination(numbers)

    odd_count = sum(
        number % 2 == 1
        for number in cleaned
    )

    even_count = len(cleaned) - odd_count
    low_count, high_count = low_high_counts(cleaned)
    sections = section_counts(cleaned)
    endings = ending_digit_counts(cleaned)
    distances = pairwise_distance_summary(cleaned)
    marking = draw_pattern_features(cleaned)

    return {
        "번호": cleaned,
        "합계": sum(cleaned),
        "홀수수": odd_count,
        "짝수수": even_count,
        "저번호수": low_count,
        "고번호수": high_count,
        "구간분포": sections,
        "사용구간수": sum(count > 0 for count in sections),
        "최대구간수": max(sections) if sections else 0,
        "연속쌍": consecutive_pairs(cleaned),
        "최장연속길이": longest_consecutive_run(cleaned),
        "최대동일끝수": max(endings.values()) if endings else 0,
        "끝수분포": dict(endings),
        "공간분산점수": spatial_balance_score(cleaned),
        **distances,
        **marking,
    }


def historical_sum_range(
    draws: np.ndarray,
    lower_quantile: float = 0.12,
    upper_quantile: float = 0.88,
    recent_window: int = 100,
) -> Tuple[int, int]:
    """
    과거 당첨번호 합계의 분위수를 이용해
    허용 합계 범위를 자동 계산합니다.

    데이터가 부족하면 100~175를 사용합니다.
    """

    draws = np.asarray(draws)

    if draws.ndim != 2 or draws.shape[1] != 6:
        return 100, 175

    if len(draws) < 20:
        return 100, 175

    recent = draws[-min(len(draws), int(recent_window)):]

    sums = np.sum(recent, axis=1)

    minimum_sum = int(
        np.floor(
            np.quantile(sums, lower_quantile)
        )
    )

    maximum_sum = int(
        np.ceil(
            np.quantile(sums, upper_quantile)
        )
    )

    minimum_sum = max(80, minimum_sum)
    maximum_sum = min(200, maximum_sum)

    if minimum_sum >= maximum_sum:
        return 100, 175

    return minimum_sum, maximum_sum


def hard_filter_combination(
    numbers: Sequence[int],
    minimum_sum: int = 100,
    maximum_sum: int = 175,
    minimum_used_sections: int = 3,
    maximum_per_section: int = 3,
    maximum_consecutive_pairs: int = 2,
    maximum_consecutive_run: int = 3,
    maximum_same_ending: int = 2,
    minimum_spatial_score: float = 50.0,
    allowed_odd_counts: Sequence[int] = (2, 3, 4),
    allowed_low_counts: Sequence[int] = (2, 3, 4),
) -> Tuple[bool, List[str]]:
    """
    조합이 필수 균형조건을 통과하는지 검사합니다.

    반환:
    - 통과 여부
    - 탈락 사유 목록
    """

    features = combination_features(numbers)
    reasons: List[str] = []

    if len(features["번호"]) != DRAW_SIZE:
        reasons.append("유효한 번호가 6개가 아님")
        return False, reasons

    if not (
        int(minimum_sum)
        <= features["합계"]
        <= int(maximum_sum)
    ):
        reasons.append("합계 범위 벗어남")

    if (
        features["홀수수"]
        not in set(int(value) for value in allowed_odd_counts)
    ):
        reasons.append("홀짝 균형 불량")

    if (
        features["저번호수"]
        not in set(int(value) for value in allowed_low_counts)
    ):
        reasons.append("저고 균형 불량")

    if (
        features["사용구간수"]
        < int(minimum_used_sections)
    ):
        reasons.append("번호구간 사용 부족")

    if (
        features["최대구간수"]
        > int(maximum_per_section)
    ):
        reasons.append("한 번호구간에 과도하게 집중")

    if (
        features["연속쌍"]
        > int(maximum_consecutive_pairs)
    ):
        reasons.append("연속번호쌍 과다")

    if (
        features["최장연속길이"]
        > int(maximum_consecutive_run)
    ):
        reasons.append("연속번호 길이 과다")

    if (
        features["최대동일끝수"]
        > int(maximum_same_ending)
    ):
        reasons.append("동일 끝수 과다")

    if (
        features["공간분산점수"]
        < float(minimum_spatial_score)
    ):
        reasons.append("구매용지 공간 쏠림")

    return len(reasons) == 0, reasons


def soft_balance_score(
    numbers: Sequence[int],
    minimum_sum: int = 100,
    maximum_sum: int = 175,
) -> float:
    """
    조합의 전체 균형을 0~100점으로 평가합니다.

    엄격 탈락조건과 별도로 추천 조합 간
    우선순위를 결정할 때 사용합니다.
    """

    features = combination_features(numbers)

    if len(features["번호"]) != DRAW_SIZE:
        return 0.0

    score = 100.0

    target_sum = (
        float(minimum_sum)
        + float(maximum_sum)
    ) / 2.0

    sum_half_range = max(
        1.0,
        (
            float(maximum_sum)
            - float(minimum_sum)
        ) / 2.0,
    )

    sum_distance = abs(
        float(features["합계"])
        - target_sum
    )

    score -= min(
        20.0,
        20.0 * sum_distance / sum_half_range,
    )

    if features["홀수수"] == 3:
        score += 5
    elif features["홀수수"] in (2, 4):
        score += 2
    else:
        score -= 18

    if features["저번호수"] == 3:
        score += 5
    elif features["저번호수"] in (2, 4):
        score += 2
    else:
        score -= 15

    if features["사용구간수"] >= 4:
        score += 6
    elif features["사용구간수"] == 3:
        score += 2
    else:
        score -= 20

    if features["최대구간수"] >= 4:
        score -= 25
    elif features["최대구간수"] == 3:
        score -= 8

    if features["연속쌍"] == 0:
        score += 3
    elif features["연속쌍"] == 1:
        score += 1
    elif features["연속쌍"] == 2:
        score -= 4
    else:
        score -= 20

    if features["최장연속길이"] >= 4:
        score -= 25
    elif features["최장연속길이"] == 3:
        score -= 8

    if features["최대동일끝수"] >= 3:
        score -= 18

    if features["첫끝범위"] < 25:
        score -= 15
    elif features["첫끝범위"] >= 35:
        score += 3

    score += (
        float(features["공간분산점수"])
        - 50.0
    ) * 0.30

    return round(
        max(0.0, min(100.0, score)),
        2,
    )


def overlap_count(
    first: Sequence[int],
    second: Sequence[int],
) -> int:
    """두 조합의 동일번호 개수를 계산합니다."""

    return len(
        set(clean_combination(first))
        & set(clean_combination(second))
    )


def combination_diversity_penalty(
    candidate: Sequence[int],
    selected_combinations: Sequence[Sequence[int]],
    overlap_penalties: Optional[Dict[int, float]] = None,
) -> float:
    """
    이미 선택된 게임들과의 번호 중복에 따라 감점합니다.

    기본:
    0~2개 중복: 감점 없음
    3개 중복: 8점
    4개 중복: 22점
    5개 중복: 45점
    6개 중복: 100점
    """

    if overlap_penalties is None:
        overlap_penalties = {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 8.0,
            4: 22.0,
            5: 45.0,
            6: 100.0,
        }

    if not selected_combinations:
        return 0.0

    penalties = []

    for selected in selected_combinations:
        overlap = overlap_count(candidate, selected)

        penalties.append(
            float(
                overlap_penalties.get(
                    overlap,
                    100.0,
                )
            )
        )

    maximum_penalty = max(penalties)
    average_penalty = float(np.mean(penalties))

    return round(
        0.70 * maximum_penalty
        + 0.30 * average_penalty,
        4,
    )


def candidate_quality_score(
    numbers: Sequence[int],
    number_scores: Sequence[float],
    selected_combinations: Sequence[Sequence[int]],
    minimum_sum: int = 100,
    maximum_sum: int = 175,
) -> float:
    """
    번호별 분석점수와 조합 균형·게임 간 다양성을
    합쳐 최종 후보조합 품질점수를 계산합니다.
    """

    cleaned = clean_combination(numbers)
    score_array = np.asarray(number_scores, dtype=float)

    if (
        len(cleaned) != DRAW_SIZE
        or score_array.size != 45
    ):
        return -9999.0

    individual_score = float(
        np.mean(
            [
                score_array[number - 1]
                for number in cleaned
            ]
        )
    )

    if np.nanmax(score_array) <= 1.5:
        individual_score *= 100.0

    balance_score = soft_balance_score(
        cleaned,
        minimum_sum=minimum_sum,
        maximum_sum=maximum_sum,
    )

    diversity_penalty = combination_diversity_penalty(
        cleaned,
        selected_combinations,
    )

    final_score = (
        0.68 * individual_score
        + 0.32 * balance_score
        - diversity_penalty
    )

    return round(float(final_score), 4)


def probability_with_temperature(
    number_scores: Sequence[float],
    temperature: float = 1.35,
    excluded_numbers: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    번호점수를 추첨확률로 변환합니다.

    temperature가 1보다 크면 상위 번호에
    확률이 지나치게 몰리는 현상이 완화됩니다.
    """

    scores = np.asarray(number_scores, dtype=float)

    if scores.size != 45:
        raise ValueError(
            "번호별 점수 배열은 45개여야 합니다."
        )

    temperature = max(0.20, float(temperature))

    finite_scores = np.where(
        np.isfinite(scores),
        scores,
        0.0,
    )

    shifted = (
        finite_scores
        - np.max(finite_scores)
    ) / temperature

    shifted = np.clip(shifted, -50, 50)

    probabilities = np.exp(shifted)

    if excluded_numbers:
        for number in excluded_numbers:
            number = int(number)

            if NUMBER_MIN <= number <= NUMBER_MAX:
                probabilities[number - 1] = 0.0

    probability_sum = probabilities.sum()

    if probability_sum <= 0:
        probabilities = np.ones(45, dtype=float)

        if excluded_numbers:
            for number in excluded_numbers:
                number = int(number)

                if NUMBER_MIN <= number <= NUMBER_MAX:
                    probabilities[number - 1] = 0.0

        probability_sum = probabilities.sum()

    return probabilities / probability_sum


def generate_balanced_combinations(
    number_scores: Sequence[float],
    game_count: int = 5,
    fixed_numbers: Optional[Sequence[int]] = None,
    excluded_numbers: Optional[Sequence[int]] = None,
    historical_draws: Optional[np.ndarray] = None,
    temperature: float = 1.35,
    candidate_trials: int = 6000,
    minimum_spatial_score: float = 50.0,
    random_seed: Optional[int] = None,
) -> Tuple[List[List[int]], List[Dict[str, object]]]:
    """
    점수 기반 추출과 균형필터를 결합해
    숫자 쏠림이 완화된 추천조합을 생성합니다.

    반환:
    - 추천조합 목록
    - 각 조합의 평가정보
    """

    rng = np.random.default_rng(random_seed)

    fixed = clean_combination(fixed_numbers or [])
    excluded = set(
        clean_combination(excluded_numbers or [])
    )

    if len(fixed) > DRAW_SIZE:
        raise ValueError(
            "고정수는 최대 6개까지 가능합니다."
        )

    if set(fixed) & excluded:
        raise ValueError(
            "고정수와 제외수에 같은 번호가 있습니다."
        )

    selectable_numbers = [
        number
        for number in range(NUMBER_MIN, NUMBER_MAX + 1)
        if number not in excluded
        and number not in set(fixed)
    ]

    required_count = DRAW_SIZE - len(fixed)

    if len(selectable_numbers) < required_count:
        raise ValueError(
            "제외수가 너무 많아 조합을 만들 수 없습니다."
        )

    if historical_draws is not None:
        minimum_sum, maximum_sum = historical_sum_range(
            historical_draws
        )
    else:
        minimum_sum, maximum_sum = 100, 175

    all_probabilities = probability_with_temperature(
        number_scores=number_scores,
        temperature=temperature,
        excluded_numbers=list(excluded),
    )

    selectable_probabilities = np.asarray(
        [
            all_probabilities[number - 1]
            for number in selectable_numbers
        ],
        dtype=float,
    )

    selectable_probabilities /= (
        selectable_probabilities.sum()
    )

    candidate_map: Dict[
        Tuple[int, ...],
        Dict[str, object],
    ] = {}

    for _ in range(max(500, int(candidate_trials))):
        if required_count == 0:
            candidate = sorted(fixed)
        else:
            sampled = rng.choice(
                selectable_numbers,
                size=required_count,
                replace=False,
                p=selectable_probabilities,
            )

            candidate = sorted(
                fixed
                + [int(number) for number in sampled]
            )

        candidate_key = tuple(candidate)

        if candidate_key in candidate_map:
            continue

        passed, reasons = hard_filter_combination(
            candidate,
            minimum_sum=minimum_sum,
            maximum_sum=maximum_sum,
            minimum_spatial_score=minimum_spatial_score,
        )

        if not passed:
            continue

        candidate_map[candidate_key] = {
            "번호": candidate,
            "탈락사유": reasons,
            "균형점수": soft_balance_score(
                candidate,
                minimum_sum=minimum_sum,
                maximum_sum=maximum_sum,
            ),
        }

    if not candidate_map:
        return [], []

    remaining = list(candidate_map.values())
    selected: List[List[int]] = []
    selected_details: List[Dict[str, object]] = []

    while (
        remaining
        and len(selected) < int(game_count)
    ):
        for item in remaining:
            item["최종품질점수"] = candidate_quality_score(
                numbers=item["번호"],
                number_scores=number_scores,
                selected_combinations=selected,
                minimum_sum=minimum_sum,
                maximum_sum=maximum_sum,
            )

        best_item = max(
            remaining,
            key=lambda item: item["최종품질점수"],
        )

        best_numbers = list(best_item["번호"])
        selected.append(best_numbers)

        details = dict(best_item)
        details.update(
            combination_features(best_numbers)
        )
        details["합계허용최소"] = minimum_sum
        details["합계허용최대"] = maximum_sum

        selected_details.append(details)

        remaining = [
            item
            for item in remaining
            if tuple(item["번호"]) != tuple(best_numbers)
        ]

    return selected, selected_details
