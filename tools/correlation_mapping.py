"""
Utilities for correlation mapping
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_masked_iou(
    map_a: torch.Tensor,
    map_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
):
    batch_size = map_a.shape[0]

    # Assumes maps and masks are 1D tensors
    assert len(map_a.shape) == 2, f"{map_a.shape = }"
    assert len(mask_a.shape) == 2, f"{mask_a.shape = }"

    # The zero index is ignored, therefore increment all
    # ids by one then zero out the masked ones
    zeros = torch.zeros(batch_size, 1)
    ids = torch.cat([zeros, map_a + 1, map_b + 1], dim=1)
    masks = torch.cat([zeros, mask_a, mask_b], dim=1)
    assert batch_size == ids.shape[0], f"{ids.shape = }"

    # Calculate the ious
    ious = torch.zeros(batch_size)
    for i in range(ids.shape[0]):
        u, c = torch.unique(ids[i] * masks[i], return_counts=True, sorted=True)
        # The union is the number of unique ids
        union = len(u) - 1  # minus the zero (ignored) ids
        # The intersection is the number of shared ids
        c = c[1:]  # ignore the zero id
        intersection = len(c[c > 1])
        ious[i] = intersection / union

    return ious


def get_correlation_map(map_a: torch.Tensor, map_b: torch.Tensor):
    """
    Get the correlation map between two ID maps.
    """
    # Assumes a batch of maps of size (B, H, W)
    assert len(map_a.shape) == 3, f"{map_a.shape = }"
    # Requires Tensors
    assert isinstance(map_a, torch.Tensor), f"{type(map_a) = }"

    #
    # Generate the pixel correspondence map
    #
    # (B, HW, HW)
    batch_size = map_a.shape[0]
    corr_map = ~(
        (
            map_a.reshape(batch_size, -1)[:, :, None]
            - map_b.reshape(batch_size, -1)[:, None, :]
        ).to(torch.bool)
    )
    # Generate them per map
    # (B, HW)
    corr_map_a = corr_map.sum(2)
    corr_map_b = corr_map.sum(1)
    # Get the iou (B,)
    # intersection = corr_map_a.sum(1)
    # union = (
    #     intersection
    #     + (corr_map_b.shape[1] - intersection)
    #     + (corr_map_a.shape[1] - intersection)
    # )
    # iou = intersection / union
    iou = get_masked_iou(
        map_a=map_a.reshape(batch_size, -1),
        map_b=map_b.reshape(batch_size, -1),
        mask_a=torch.ones(batch_size, map_a.shape[1] * map_a.shape[2]),
        mask_b=torch.ones(batch_size, map_b.shape[1] * map_b.shape[2]),
    )

    return {
        "corr_map": corr_map,
        "corr_map_a": corr_map_a,
        "corr_map_b": corr_map_b,
        "iou": iou,
    }


def get_masked_correlation_map(
    map_a: torch.Tensor,
    map_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
):
    """
    Get the pixel to pixel correlation maps after applying a mask
    """
    # Get correlation maps
    results = get_correlation_map(map_a, map_b)
    iou = results["iou"]
    corr_map = results["corr_map"]
    corr_map_a = results["corr_map_a"]
    corr_map_b = results["corr_map_b"]

    # Generate the masked correlations
    batch_size = mask_a.shape[0]
    mask = torch.einsum(
        "nx,ny->nxy",
        [mask_a.reshape(batch_size, -1), mask_b.reshape(batch_size, -1)],
    )

    # Filter the correlation map
    assert mask.shape == corr_map.shape, f"{mask.shape = }, {corr_map.shape = }"
    corr_mask = corr_map * mask
    # Generate them per map
    # (B, HW)
    corr_map_a_masked = corr_mask.sum(2)
    corr_map_b_masked = corr_mask.sum(1)
    # # Get the iou (B,)
    # intersection = corr_map_a_masked.sum(1)
    # # union = corr_map_a_masked.shape[1] + corr_map_b_masked.shape[1]
    # union = (
    #     intersection
    #     + (mask_a.sum((1, 2)) - intersection)
    #     + (mask_b.sum((1, 2)) - intersection)
    # )
    # iou_masked = intersection / union
    iou_masked = get_masked_iou(
        map_a=map_a.reshape(batch_size, -1),
        map_b=map_b.reshape(batch_size, -1),
        mask_a=mask_a.reshape(batch_size, -1),
        mask_b=mask_b.reshape(batch_size, -1),
    )

    return {
        "corr_map": corr_map,
        "corr_mask": corr_mask,
        "corr_map_a": corr_map_a,
        "corr_map_a_masked": corr_map_a_masked,
        "corr_map_b": corr_map_b,
        "corr_map_b_masked": corr_map_b_masked,
        "iou": iou,
        "iou_masked": iou_masked,
    }


def visualize_correlation_map(
    map_a: torch.Tensor,
    map_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    save_dir: str,
):
    # Get correlation maps
    results = get_masked_correlation_map(
        map_a=map_a,
        map_b=map_b,
        mask_a=mask_a,
        mask_b=mask_b,
    )
    iou = results["iou"].detach().cpu().numpy()
    corr_map_a = results["corr_map_a"].detach().cpu().numpy()
    corr_map_b = results["corr_map_b"].detach().cpu().numpy()
    iou_masked = results["iou_masked"].detach().cpu().numpy()
    corr_map_a_masked = results["corr_map_a_masked"].detach().cpu().numpy()
    corr_map_b_masked = results["corr_map_b_masked"].detach().cpu().numpy()

    print(f"{results['corr_mask'].shape = }")
    print(f"{results['corr_map'].shape = }")
    print(f"{corr_map_a_masked.shape = }")
    print(f"{corr_map_a.shape = }")

    # Convert back to numpy arrays
    map_a = map_a.detach().numpy()
    map_b = map_b.detach().numpy()

    # Prints
    print(f"{iou.shape = }")
    print(f"{corr_map_a.shape = }")
    print(f"{corr_map_b.shape = }")

    # Plot histogram of IoU values
    plt.figure(figsize=(10, 4))
    plt.hist(iou, bins="auto")
    plt.title("Histogram of IoU values")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    print("Saving the iou histogram")
    plt.savefig(os.path.join(save_dir, "iou_histogram.png"))
    # Plot histogram of masked IoU values
    plt.figure(figsize=(10, 4))
    plt.hist(iou_masked, bins="auto")
    plt.title("Histogram of Masked IoU values")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    print("Saving the masked iou histogram")
    plt.savefig(os.path.join(save_dir, "masked_iou_histogram.png"))

    # Plot original maps and correlation maps
    print("Plotting the correlation maps")
    batch_size = map_a.shape[0]
    height, width = map_a.shape[1], map_a.shape[2]
    fig, axes = plt.subplots(batch_size, 10, figsize=(20, 20))
    for i in range(batch_size):
        print(f"{iou[i] = }")
        print(f"{iou_masked[i] = }")
        vmin = min(np.min(map_a[i]), np.min(map_b[i]))
        vmax = max(np.max(map_a[i]), np.max(map_b[i]))

        axes[i, 0].imshow(map_a[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"map_a[{i}]")
        axes[i, 1].imshow(corr_map_a[i].reshape(height, width), cmap="gray")
        axes[i, 1].set_title(f"corr_map_a[{i}]")
        axes[i, 2].imshow(mask_a[i], cmap="grey")
        axes[i, 2].set_title(f"mask_a[{i}]")
        axes[i, 3].imshow(mask_a[i] * map_a[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 3].set_title(f"mask_a[{i}]*map_a[{i}]")
        axes[i, 4].imshow(corr_map_a_masked[i].reshape(height, width), cmap="gray")
        axes[i, 4].set_title(f"corr_map_a_masked[{i}]")

        axes[i, 5].imshow(map_b[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 5].set_title(f"map_b[{i}]")
        axes[i, 6].imshow(corr_map_b[i].reshape(height, width), cmap="gray")
        axes[i, 6].set_title(f"corr_map_b[{i}]")
        axes[i, 7].imshow(mask_b[i], cmap="grey")
        axes[i, 7].set_title(f"mask_b[{i}]")
        axes[i, 8].imshow(mask_b[i] * map_b[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 8].set_title(f"mask_b[{i}]*map_b[{i}]")
        axes[i, 9].imshow(corr_map_b_masked[i].reshape(height, width), cmap="gray")
        axes[i, 9].set_title(f"corr_map_b_masked[{i}]")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "maps_visualization.png"))

    return results


if __name__ == "__main__":
    # Example data
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

    offset_h = np.random.randint(0, height // 3)
    offset_w = np.random.randint(0, width // 3)

    print(f"{offset_h = }, {offset_w = }, {height = }, {width = }, {batch_size = }")
    print(f"{crop_height = }, {crop_width = }")

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
        map_a=map_a,
        map_b=map_b,
        mask_a=mask_a,
        mask_b=mask_b,
        save_dir="./artifacts",
    )
