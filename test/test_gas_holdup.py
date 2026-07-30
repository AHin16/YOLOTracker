#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import unittest

import numpy as np

from bubble_analyzer.gas_holdup import EdgeBubblePolicy, ROI, analyze_masks


class GasHoldupAnalysisTest(unittest.TestCase):
    def test_uses_union_of_segmentation_masks_for_area_and_ratio(self):
        first = np.zeros((10, 10), dtype=np.uint8)
        second = np.zeros((10, 10), dtype=np.uint8)
        first[1:5, 1:5] = 1
        second[3:7, 3:7] = 1

        result = analyze_masks(
            [first, second],
            frame_shape=(10, 10),
            min_bubble_area=1,
            edge_policy=EdgeBubblePolicy.INCLUDE,
        )

        self.assertEqual(result.bubble_count, 2)
        self.assertEqual(result.bubble_area, 28)
        self.assertEqual(result.roi_area, 100)
        self.assertAlmostEqual(result.bubble_area_ratio, 28.0)
        self.assertAlmostEqual(result.gas_holdup, 28.0)
        expected_diameter = 2.0 * math.sqrt(16.0 / math.pi)
        self.assertAlmostEqual(result.average_diameter, expected_diameter)

    def test_applies_minimum_area_without_removing_valid_small_bubbles(self):
        noise = np.zeros((8, 8), dtype=np.uint8)
        bubble = np.zeros((8, 8), dtype=np.uint8)
        noise[1, 1] = 1
        bubble[2:4, 2:5] = 1

        result = analyze_masks(
            [noise, bubble],
            frame_shape=(8, 8),
            min_bubble_area=2,
            edge_policy=EdgeBubblePolicy.INCLUDE,
        )

        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.bubble_area, 6)

    def test_clips_statistics_to_roi(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1

        result = analyze_masks(
            [mask],
            frame_shape=(10, 10),
            roi=ROI(x=5, y=5, width=5, height=5),
            min_bubble_area=1,
            edge_policy=EdgeBubblePolicy.INCLUDE,
        )

        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.bubble_area, 9)
        self.assertEqual(result.roi_area, 25)
        self.assertAlmostEqual(result.gas_holdup, 36.0)

    def test_exclude_ignores_masks_touching_the_image_border(self):
        edge_bubble = np.zeros((10, 10), dtype=np.uint8)
        inner_bubble = np.zeros((10, 10), dtype=np.uint8)
        edge_bubble[0:3, 2:5] = 1
        inner_bubble[4:7, 4:7] = 1

        result = analyze_masks(
            [edge_bubble, inner_bubble],
            frame_shape=(10, 10),
            min_bubble_area=1,
            edge_policy=EdgeBubblePolicy.EXCLUDE,
        )

        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.bubble_area, 9)

    def test_weighted_counts_only_visible_segmented_area(self):
        edge_bubble = np.zeros((6, 6), dtype=np.uint8)
        edge_bubble[0:2, 0:3] = 1

        result = analyze_masks(
            [edge_bubble],
            frame_shape=(6, 6),
            min_bubble_area=1,
            edge_policy=EdgeBubblePolicy.WEIGHTED,
        )

        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.bubble_area, 6)
        self.assertAlmostEqual(result.gas_holdup, 100.0 * 6.0 / 36.0)

    def test_rejects_masks_with_unexpected_dimensions(self):
        bad_mask = np.zeros((5, 5), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "mask shape"):
            analyze_masks(
                [bad_mask],
                frame_shape=(10, 10),
                min_bubble_area=1,
                edge_policy=EdgeBubblePolicy.INCLUDE,
            )


if __name__ == "__main__":
    unittest.main()
