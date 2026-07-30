from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.diversity import generate_balanced_combinations


# =========================================================
# LOTTO GPT V26.1
# 5게임 세트 다양성 최적화 엔진
#
# 주요 기능
# 1. 다수의 후보조합 생성
# 2. 동일 조합 제거
# 3. 게임 간 중복번호 감점
# 4. 동일 번호 반복 횟수 제한
# 5. 세트 전체 번호 사용범위 확대
# =========================================================


def clean_combination(
    combination: Sequence[int],
) -> List[int]:
    """조합을 정수형 오름차순 6개 번호로 정리합니다."""

    cleaned = sorted(
        set(
            int(number)
            for number in combination
            if 1 <= int(number) <= 45
        )
    )

    if len(cleaned) != 6:
        return []

    return cleaned


def combination_key(
    combination: Sequence[int],
) -> Tuple[int, ...]:
    """중복 조합 확인용 키를 만듭니다."""

    return tuple(
        clean_combination(combination)
    )


def overlap_count(
    first: Sequence[int],
    second: Sequence[int],
) -> int:
    """두 조합의 공통 번호 개수를 계산합니다."""

    return len(
        set(first) & set(second)
    )


def number_usage_counts(
    combinations: Sequence[Sequence[int]],
) -> Counter:
    """현재 세트에서 번호별 사용 횟수를 계산합니다."""

    counter: Counter = Counter()

    for combination in combinations:
        counter.update(
            int(number)
            for number in combination
        )

    return counter


def detail_quality(
    detail: Optional[Dict],
) -> float:
    """
    다양성 엔진이 반환한 품질점수를 안전하게 읽습니다.

    다양한 열 이름을 지원합니다.
    """

    if not detail:
        return 0.0

    candidate_names = [
        "최종품질점수",
        "품질점수",
        "균형점수",
        "quality_score",
        "score",
    ]

    for name in candidate_names:
        if name not in detail:
            continue

        try:
            value = float(detail[name])

            if np.isfinite(value):
                return value

        except (TypeError, ValueError):
            continue

    return 0.0


def candidate_set_score(
    combination: Sequence[int],
    selected_combinations: Sequence[Sequence[int]],
    selected_usage: Counter,
    detail: Optional[Dict] = None,
    max_number_repeat: int = 2,
    max_pair_overlap: int = 2,
) -> float:
    """
    새 조합을 현재 세트에 추가할 가치가 있는지 계산합니다.

    높은 점수:
    - 기존 게임에서 적게 사용된 번호
    - 새로운 번호가 많은 조합
    - 기존 조합과 겹침이 적은 조합
    - 자체 품질점수가 높은 조합

    강한 감점:
    - 동일 번호가 제한 횟수를 초과
    - 기존 한 게임과 3개 이상 겹침
    """

    numbers = clean_combination(combination)

    if len(numbers) != 6:
        return -1_000_000.0

    repeated_over_limit = sum(
        selected_usage[number] >= max_number_repeat
        for number in numbers
    )

    if repeated_over_limit > 0:
        return -100_000.0 - repeated_over_limit * 10_000.0

    overlaps = [
        overlap_count(
            numbers,
            selected,
        )
        for selected in selected_combinations
    ]

    if overlaps and max(overlaps) > max_pair_overlap:
        return -50_000.0 - max(overlaps) * 5_000.0

    new_number_count = sum(
        selected_usage[number] == 0
        for number in numbers
    )

    lightly_used_count = sum(
        selected_usage[number] == 1
        for number in numbers
    )

    usage_penalty = sum(
        selected_usage[number] ** 2
        for number in numbers
    )

    total_overlap = sum(overlaps)
    maximum_overlap = max(overlaps) if overlaps else 0

    quality = detail_quality(detail)

    return (
        quality * 0.30
        + new_number_count * 24.0
        + lightly_used_count * 5.0
        - usage_penalty * 13.0
        - total_overlap * 11.0
        - maximum_overlap * 18.0
    )


def collect_candidate_pool(
    number_scores: Sequence[float],
    fixed_numbers: Optional[Sequence[int]],
    excluded_numbers: Optional[Sequence[int]],
    historical_draws: Optional[np.ndarray],
    temperature: float,
    candidate_trials: int,
    minimum_spatial_score: float,
    random_seed: Optional[int],
    pool_rounds: int = 18,
    combinations_per_round: int = 12,
) -> Tuple[List[List[int]], List[Dict]]:
    """
    시드를 바꾸면서 다수의 후보조합을 수집합니다.
    동일한 조합은 한 번만 저장합니다.
    """

    pool: List[List[int]] = []
    pool_details: List[Dict] = []
    seen = set()

    base_seed = (
        int(random_seed)
        if random_seed is not None
        else None
    )

    for round_index in range(
        max(1, int(pool_rounds))
    ):
        round_seed = (
            base_seed + round_index * 137
            if base_seed is not None
            else None
        )

        try:
            combinations, details = (
                generate_balanced_combinations(
                    number_scores=number_scores,
                    game_count=int(combinations_per_round),
                    fixed_numbers=list(
                        fixed_numbers or []
                    ),
                    excluded_numbers=list(
                        excluded_numbers or []
                    ),
                    historical_draws=historical_draws,
                    temperature=float(temperature),
                    candidate_trials=int(candidate_trials),
                    minimum_spatial_score=float(
                        minimum_spatial_score
                    ),
                    random_seed=round_seed,
                )
            )

        except Exception:
            continue

        for index, combination in enumerate(
            combinations
        ):
            cleaned = clean_combination(
                combination
            )

            if len(cleaned) != 6:
                continue

            key = tuple(cleaned)

            if key in seen:
                continue

            seen.add(key)
            pool.append(cleaned)

            if index < len(details):
                pool_details.append(
                    dict(details[index])
                )
            else:
                pool_details.append({})

    return pool, pool_details


def select_diverse_set(
    candidate_pool: Sequence[Sequence[int]],
    candidate_details: Sequence[Dict],
    target_game_count: int = 5,
    max_number_repeat: int = 2,
    max_pair_overlap: int = 2,
) -> Tuple[List[List[int]], List[Dict]]:
    """
    후보조합 중에서 세트 전체 다양성이 높은 조합을 선택합니다.
    """

    target_game_count = max(
        1,
        int(target_game_count),
    )

    remaining = [
        clean_combination(combination)
        for combination in candidate_pool
    ]

    details = [
        (
            dict(candidate_details[index])
            if index < len(candidate_details)
            else {}
        )
        for index in range(len(remaining))
    ]

    valid_items = [
        (combination, detail)
        for combination, detail in zip(
            remaining,
            details,
        )
        if len(combination) == 6
    ]

    selected: List[List[int]] = []
    selected_details: List[Dict] = []
    usage: Counter = Counter()

    while (
        len(selected) < target_game_count
        and valid_items
    ):
        scored_items = []

        for item_index, (
            combination,
            detail,
        ) in enumerate(valid_items):
            score = candidate_set_score(
                combination=combination,
                selected_combinations=selected,
                selected_usage=usage,
                detail=detail,
                max_number_repeat=max_number_repeat,
                max_pair_overlap=max_pair_overlap,
            )

            scored_items.append(
                (
                    score,
                    item_index,
                    combination,
                    detail,
                )
            )

        scored_items.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        (
            best_score,
            best_index,
            best_combination,
            best_detail,
        ) = scored_items[0]

        if best_score <= -50_000:
            break

        selected.append(
            best_combination
        )

        enriched_detail = dict(
            best_detail
        )

        enriched_detail[
            "세트선택점수"
        ] = round(
            float(best_score),
            2,
        )

        selected_details.append(
            enriched_detail
        )

        usage.update(
            best_combination
        )

        valid_items.pop(
            best_index
        )

    return (
        selected,
        selected_details,
    )


def set_summary(
    combinations: Sequence[Sequence[int]],
) -> Dict[str, object]:
    """완성된 추천 세트의 다양성 통계를 계산합니다."""

    cleaned = [
        clean_combination(combination)
        for combination in combinations
        if len(clean_combination(combination)) == 6
    ]

    usage = number_usage_counts(
        cleaned
    )

    used_numbers = sorted(
        usage.keys()
    )

    repeated_numbers = {
        int(number): int(count)
        for number, count in usage.items()
        if count >= 2
    }

    pair_overlaps: List[int] = []

    for first_index in range(len(cleaned)):
        for second_index in range(
            first_index + 1,
            len(cleaned),
        ):
            pair_overlaps.append(
                overlap_count(
                    cleaned[first_index],
                    cleaned[second_index],
                )
            )

    total_slots = len(cleaned) * 6
    unique_count = len(used_numbers)

    diversity_ratio = (
        unique_count / total_slots
        if total_slots > 0
        else 0.0
    )

    return {
        "게임수": len(cleaned),
        "전체번호칸": total_slots,
        "고유번호수": unique_count,
        "고유번호비율": diversity_ratio,
        "최대번호반복": (
            max(usage.values())
            if usage
            else 0
        ),
        "평균게임중복": (
            float(np.mean(pair_overlaps))
            if pair_overlaps
            else 0.0
        ),
        "최대게임중복": (
            max(pair_overlaps)
            if pair_overlaps
            else 0
        ),
        "반복번호": repeated_numbers,
        "사용번호": used_numbers,
    }


def generate_practical_lotto_set(
    number_scores: Sequence[float],
    game_count: int,
    fixed_numbers: Optional[Sequence[int]] = None,
    excluded_numbers: Optional[Sequence[int]] = None,
    historical_draws: Optional[np.ndarray] = None,
    temperature: float = 1.35,
    candidate_trials: int = 15000,
    minimum_spatial_score: float = 30.0,
    random_seed: Optional[int] = None,
) -> Tuple[
    List[List[int]],
    List[Dict],
    Dict[str, object],
]:
    """
    V26.1 실전용 추천 세트를 생성합니다.

    1차:
    - 번호별 최대 2회
    - 게임 간 최대 중복 2개

    부족할 경우:
    - 번호별 최대 3회까지 완화
    - 게임 간 최대 중복 3개까지 완화
    """

    target_count = max(
        1,
        int(game_count),
    )

    candidate_pool, candidate_details = (
        collect_candidate_pool(
            number_scores=number_scores,
            fixed_numbers=fixed_numbers,
            excluded_numbers=excluded_numbers,
            historical_draws=historical_draws,
            temperature=temperature,
            candidate_trials=max(
                int(candidate_trials),
                8000,
            ),
            minimum_spatial_score=minimum_spatial_score,
            random_seed=random_seed,
            pool_rounds=20,
            combinations_per_round=max(
                10,
                target_count * 2,
            ),
        )
    )

    combinations, details = select_diverse_set(
        candidate_pool=candidate_pool,
        candidate_details=candidate_details,
        target_game_count=target_count,
        max_number_repeat=2,
        max_pair_overlap=2,
    )

    # 엄격한 조건에서 게임 수가 부족할 때만 완화
    if len(combinations) < target_count:
        combinations, details = select_diverse_set(
            candidate_pool=candidate_pool,
            candidate_details=candidate_details,
            target_game_count=target_count,
            max_number_repeat=3,
            max_pair_overlap=3,
        )

    summary = set_summary(
        combinations
    )

    return (
        combinations,
        details,
        summary,
    )
