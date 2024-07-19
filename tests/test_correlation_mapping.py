import unittest

import numpy as np
import torch

from tools.correlation_mapping import visualize_correlation_map

OUTPUT_DIR = "./artifacts"


class TestCorrelationMapping(unittest.TestCase):
    def test_iou(self):
        """
        Test that the iou calculations are correct
        """
        batch_size = 4
        height, width = 10, 10
        crop_height, crop_width = height // 2, width // 2

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

        results = visualize_correlation_map(
            map_a=map_a, map_b=map_b, mask_a=mask_a, mask_b=mask_b, save_dir=OUTPUT_DIR
        )

        # iou
        iou_correct = torch.ones(batch_size) * (12 / (12 + 25 - 12 + 25 - 12))
        self.assertEqual(results["iou"][0], iou_correct[0])
        self.assertTrue(torch.all(iou_correct == results["iou"]))

        # masked iou
        iou_masked_correct = torch.ones(batch_size) * (1 / 3)
        self.assertEqual(results["iou_masked"][0], iou_masked_correct[0])
        self.assertTrue(torch.all(iou_masked_correct == results["iou_masked"]))

        iou = results["iou"].detach().cpu().numpy()
        corr_map_a = results["corr_map_a"].detach().cpu().numpy()
        corr_map_b = results["corr_map_b"].detach().cpu().numpy()
        iou_masked = results["iou_masked"].detach().cpu().numpy()
        corr_map_a_masked = results["corr_map_a_masked"].detach().cpu().numpy()
        corr_map_b_masked = results["corr_map_b_masked"].detach().cpu().numpy()

        # Validate masked correspondences share pixel ids
        b, h, w = map_a.shape
        idxs_a_masked = np.where(corr_map_a_masked.reshape(b, h, w))
        idxs_b_masked = np.where(corr_map_b_masked.reshape(b, h, w))
        assert torch.all(map_a[idxs_a_masked] == map_b[idxs_b_masked])

        # Validate correlations share pixel ids
        idxs_a = np.where(corr_map_a.reshape(b, h, w))
        idxs_b = np.where(corr_map_b.reshape(b, h, w))
        assert torch.all(map_a[idxs_a] == map_b[idxs_b])


if __name__ == "__main__":
    unittest.main()
