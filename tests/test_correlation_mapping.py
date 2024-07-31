import unittest
from typing import List, Tuple

import numpy as np
import torch
from parameterized import parameterized

from tools.correlation_mapping import visualize_correlation_map

OUTPUT_DIR = "./artifacts"


def get_params() -> List[Tuple[str, dict]]:
    """
    Returns
        name,
        {
            "map_a": torch.Tensor,
            "map_b": torch.Tensor,
            "mask_a": torch.Tensor,
            "mask_b": torch.Tensor,
            "iou": iou,
            "iou_masked": iou_masked,
        }
    """
    params = []
    batch_size = 4
    height, width = 10, 10
    crop_height, crop_width = height // 2, width // 2

    #
    # Variant: Unique id per image pixel
    #

    # Create the pixel ids
    base_map = torch.arange(1, batch_size * height * width + 1)
    # Shuffle ids
    shuffle_ids = list(range(0, len(base_map)))
    np.random.shuffle(shuffle_ids)
    base_map = base_map[shuffle_ids]
    # Reshape to image dimensions
    base_map = base_map.reshape(batch_size, height, width)

    offset_h = 1
    offset_w = 2

    map_a = base_map[:, :crop_height, :crop_width]
    map_b = base_map[
        :,
        offset_h : offset_h + crop_height,
        offset_w : offset_w + crop_width,
    ]
    # map_a = torch.randint(0, 2, (batch_size, height, width))
    # map_b = torch.randint(0, 2, (batch_size, height, width))

    # Create masks
    mask_a = torch.zeros((batch_size, crop_height, crop_width))
    mask_a[:, 2:4, 1:3] = 1
    mask_b = torch.zeros((batch_size, crop_height, crop_width))
    mask_b[:, 1:3, 0:2] = 1

    # Update Params
    params.append(
        (
            "uniqueIds",
            {
                "map_a": map_a.clone(),
                "map_b": map_b.clone(),
                "mask_a": mask_a.clone(),
                "mask_b": mask_b.clone(),
                "iou": torch.ones(batch_size) * (12 / (12 + 25 - 12 + 25 - 12)),
                "iou_masked": torch.ones(batch_size) * (1 / 3),
            },
        )
    )

    #
    # Variant: Shared ids
    #
    base_map = torch.Tensor(
        [
            [
                [1, 2, 2, 3, 4, 5],
                [6, 2, 2, 3, 3, 3],
                [7, 8, 9, 10, 11, 12],
                [13, 8, 8, 8, 14, 15],
            ]
        ]
    )

    map_a = base_map[:, 0:3, 1:4]
    map_b = base_map[:, 0:3, 2:5]
    print(f"{base_map = }")
    print(f"{map_a = }")
    print(f"{map_b = }")
    mask_a = torch.Tensor(
        [
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
            ]
        ]
    )
    mask_b = torch.Tensor(
        [
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        ]
    )

    # Update Params
    params.append(
        (
            "sharedIds",
            {
                "map_a": map_a.clone(),
                "map_b": map_b.clone(),
                "mask_a": mask_a.clone(),
                "mask_b": mask_b.clone(),
                "iou": torch.Tensor([[4 / 7]]),
                "iou_masked": torch.Tensor([[2 / 3]]),
            },
        )
    )

    return params


class TestCorrelationMapping(unittest.TestCase):
    @parameterized.expand(get_params())
    def test_iou_unique(self, name, ingredients):
        """
        Test that the iou calculations are correct
        """
        map_a = ingredients["map_a"]
        mask_a = ingredients["mask_a"]
        map_b = ingredients["map_b"]
        mask_b = ingredients["mask_b"]
        iou_correct = ingredients["iou"]
        iou_masked_correct = ingredients["iou_masked"]

        results = visualize_correlation_map(
            name=name,
            map_a=map_a,
            map_b=map_b,
            mask_a=mask_a,
            mask_b=mask_b,
            save_dir=OUTPUT_DIR,
        )

        # iou
        self.assertEqual(results["iou"][0], iou_correct[0])
        self.assertTrue(torch.all(iou_correct == results["iou"]))

        # masked iou
        self.assertEqual(results["iou_masked"][0], iou_masked_correct[0])
        self.assertTrue(torch.all(iou_masked_correct == results["iou_masked"]))

        # corr_map_a = results["corr_map_a"].detach().cpu().numpy()
        # corr_map_b = results["corr_map_b"].detach().cpu().numpy()
        # corr_map_a_masked = results["corr_map_a_masked"].detach().cpu().numpy()
        # corr_map_b_masked = results["corr_map_b_masked"].detach().cpu().numpy()

        # # Validate masked correspondences share pixel ids
        # b, h, w = map_a.shape
        # print(f"{corr_map_a_masked.reshape(b, h, w) = }")
        # print(f"{corr_map_b_masked.reshape(b, h, w) = }")
        # idxs_a_masked = np.where(corr_map_a_masked.reshape(b, h, w))
        # idxs_b_masked = np.where(corr_map_b_masked.reshape(b, h, w))
        # print(f"{idxs_a_masked = }")
        # print(f"{idxs_b_masked = }")
        # assert torch.all(map_a[idxs_a_masked] == map_b[idxs_b_masked])
        #
        # # Validate correspondences share pixel ids
        # print(f"{corr_map_a.reshape(b, h, w) = }")
        # print(f"{corr_map_b.reshape(b, h, w) = }")
        # idxs_a = np.where(corr_map_a.reshape(b, h, w))
        # idxs_b = np.where(corr_map_b.reshape(b, h, w))
        # assert torch.all(map_a[idxs_a] == map_b[idxs_b])


if __name__ == "__main__":
    unittest.main()
