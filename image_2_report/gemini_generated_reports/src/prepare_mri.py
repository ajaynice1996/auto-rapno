from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def prepare_mri_visualization(npz_path, output_path):

    data = np.load(npz_path, allow_pickle=False)

    image = data["image"].astype(np.float32)
    seg = data["seg"]

    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(
            f"Expected image shape (4, X, Y, Z), got {image.shape}"
        )

    if seg.shape != image.shape[1:]:
        raise ValueError("Image and segmentation shapes do not match.")

    # Select the slice with the largest foreground area.
    best_axis = None
    best_index = None
    best_area = -1

    for axis in range(3):

        areas = []

        for index in range(seg.shape[axis]):

            slice_mask = np.take(seg > 0, index, axis=axis)
            areas.append(np.count_nonzero(slice_mask))

        index = int(np.argmax(areas))
        area = int(areas[index])

        if area > best_area:
            best_area = area
            best_axis = axis
            best_index = index

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for channel in range(4):

        mri_slice = np.take(
            image[channel], best_index, axis=best_axis
        )

        mask_slice = np.take(
            seg, best_index, axis=best_axis
        )

        valid = mri_slice[np.isfinite(mri_slice)]

        if valid.size == 0:
            raise ValueError("MRI slice contains no finite values.")

        vmin, vmax = np.percentile(valid, [1, 99])

        axes[channel].imshow(
            mri_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        # Generic foreground overlay; label meanings are not assumed.
        overlay = np.ma.masked_where(
            mask_slice == 0,
            mask_slice > 0
        )

        axes[channel].imshow(
            overlay,
            cmap="autumn",
            alpha=0.45
        )

        axes[channel].set_title(f"Channel {channel}")
        axes[channel].axis("off")

    fig.suptitle(
        f"MRI channels with foreground mask | Axis {best_axis}, "
        f"slice {best_index}"
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "axis": best_axis,
        "slice_index": best_index,
        "foreground_voxels": int(np.count_nonzero(seg)),
        "visualization": str(output_path)
    }
