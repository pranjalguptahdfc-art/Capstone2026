#!/usr/bin/env python3
"""
test_bunch_logic.py
-------------------
Standalone test for the bunch analysis logic using mock classification results.
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from processing.bunch_analyzer import BunchAnalyzer, PROBLEMATIC_THRESHOLD

def print_test_case(name, bananas):
    print(f"\n{'='*60}")
    print(f"TEST CASE: {name}")
    print(f"{'='*60}")

    analyzer = BunchAnalyzer()
    result = analyzer.analyze(bananas)

    print(f"Total bananas: {result['total_bananas']}")
    print(f"Unripe count: {result['counts']['unripe']} ({result['percentages']['unripe']}%)")
    print(f"Ripe count: {result['counts']['ripe']} ({result['percentages']['ripe']}%)")
    print(f"Overripe count: {result['counts']['overripe']} ({result['percentages']['overripe']}%)")
    print(f"Problematic count: {result['problematic_count']} ({result['problematic_percentage']}%)")
    print(f"Bunch status: {result['bunch_status']}")
    print(f"Dominant problem: {result['dominant_problem']}")
    print(f"Majority result: {result['majority_result']}")
    print(f"Average ripeness score: {result['avg_ripeness_score']}")
    print(f"Score result: {result['score_result']}")

    return result

def main():
    print("Testing Bunch Analysis Logic")
    print(f"Problematic threshold: {PROBLEMATIC_THRESHOLD*100}%")

    # TEST CASE 1: 8 Ripe, 2 Unripe, 0 Over Ripe
    bananas1 = []
    for i in range(1, 9):  # 8 ripe
        bananas1.append({
            "banana_id": i,
            "bbox": [0, 0, 50, 50],
            "confidence": 0.9,
            "canonical": "ripe",
            "ripeness_class": "Ripe",
            "ripe_confidence": 0.9
        })
    for i in range(9, 11):  # 2 unripe
        bananas1.append({
            "banana_id": i,
            "bbox": [0, 0, 50, 50],
            "confidence": 0.9,
            "canonical": "unripe",
            "ripeness_class": "Unripe",
            "ripe_confidence": 0.9
        })

    result1 = print_test_case("8 Ripe, 2 Unripe, 0 Over Ripe", bananas1)

    # TEST CASE 2: 4 Ripe, 1 Unripe, 5 Over Ripe
    bananas2 = []
    for i in range(1, 5):  # 4 ripe
        bananas2.append({
            "banana_id": i,
            "bbox": [0, 0, 50, 50],
            "confidence": 0.9,
            "canonical": "ripe",
            "ripeness_class": "Ripe",
            "ripe_confidence": 0.9
        })
    bananas2.append({
        "banana_id": 5,
        "bbox": [0, 0, 50, 50],
        "confidence": 0.9,
        "canonical": "unripe",
        "ripeness_class": "Unripe",
        "ripe_confidence": 0.9
    })  # 1 unripe
    for i in range(6, 11):  # 5 overripe
        bananas2.append({
            "banana_id": i,
            "bbox": [0, 0, 50, 50],
            "confidence": 0.9,
            "canonical": "overripe",
            "ripeness_class": "Over Ripe",
            "ripe_confidence": 0.9
        })

    result2 = print_test_case("4 Ripe, 1 Unripe, 5 Over Ripe", bananas2)

    # TEST CASE 3: 10 Ripe
    bananas3 = []
    for i in range(1, 11):  # 10 ripe
        bananas3.append({
            "banana_id": i,
            "bbox": [0, 0, 50, 50],
            "confidence": 0.9,
            "canonical": "ripe",
            "ripeness_class": "Ripe",
            "ripe_confidence": 0.9
        })

    result3 = print_test_case("10 Ripe", bananas3)

    # TEST CASE 4: 10 Unripe
    bananas4 = []
    for i in range(1, 11):  # 10 unripe
        bananas4.append({
            "banana_id": i,
            "bbox": [0, 0, 50, 50],
            "confidence": 0.9,
            "canonical": "unripe",
            "ripeness_class": "Unripe",
            "ripe_confidence": 0.9
        })

    result4 = print_test_case("10 Unripe", bananas4)

    # TEST CASE 5: 0 bananas
    result5 = print_test_case("0 bananas", [])

    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()