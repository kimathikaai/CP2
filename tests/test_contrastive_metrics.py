import unittest
from typing import List, Tuple

import cv2
import numpy as np
import torch
from parameterized import parameterized

from loader import rescale_ids
from tools.correlation_mapping import visualize_correlation_map

OUTPUT_DIR = "./artifacts"


class TestContrastiveMetrics(unittest.TestCase):
    def test_median_calculation(self):
        scores = torch.Tensor(
            [
                [
                    [1, 2, 3],  # 2
                    [4, 5, 6],  # 5
                ],
                [
                    [1, 2, 3],  # 2
                    [7, 8, 9],  # 8
                ],
            ]
        )
        per_pixel_average = torch.Tensor(
            [
                [2, 5],
                [2, 8],
            ]
        )

        per_sample_average = torch.Tensor([3.5, 5])

        per_sample_median = torch.Tensor([3.5, 5])

        # print(f"{scores.mean(2) = },\n{scores.mean(2).shape = }")
        # print(f"{scores.mean(1) = },\n{scores.mean(1).shape = }")
        # print(f"{scores.mean((1,2)) = },\n{scores.mean((1,2)).shape = }")
        # print(f"{per_pixel_average = },\n{per_pixel_average.shape = }")
        self.assertTrue(torch.all(scores.mean(2) == per_pixel_average))
        self.assertTrue(torch.all(scores.mean((1, 2)) == per_sample_average))

        # quantiles
        flatten_scores = scores.flatten(1)
        print(f"{flatten_scores = },\n{flatten_scores.shape = }")
        quartiles = torch.quantile(
            flatten_scores, q=torch.Tensor([0.25, 0.5, 0.75]), dim=1
        )
        per_sample_quartiles = torch.Tensor(
            [[2.2500, 2.2500], [3.5000, 5.0000], [4.7500, 7.7500]]
        )
        print(f"{quartiles = }, {quartiles.shape = }")
        self.assertTrue(torch.all(quartiles == per_sample_quartiles))


if __name__ == "__main__":
    unittest.main()
