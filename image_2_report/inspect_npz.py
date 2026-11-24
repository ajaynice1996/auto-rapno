
from pathlib import Path
import numpy as np

# --------------------------------------------------
# 1. Set NPZ file path
# --------------------------------------------------

NPZ_FILE = Path(
    "data/cache_128/BraTSPED_00001.npz"
)

# Change the filename above to an actual NPZ file.

# --------------------------------------------------
# 2. Inspect file information
# --------------------------------------------------

print("=" * 70)
print("NPZ FILE INSPECTION")
print("=" * 70)

print("File:", NPZ_FILE)
print("Exists:", NPZ_FILE.exists())

if not NPZ_FILE.exists():
    raise FileNotFoundError(f"File not found: {NPZ_FILE}")

print("File size (MB):", round(NPZ_FILE.stat().st_size / (1024**2), 2))

# --------------------------------------------------
# 3. Load NPZ file safely
# --------------------------------------------------

with np.load(NPZ_FILE, allow_pickle=False) as data:

    print("\nAvailable keys:")
    print("-" * 70)

    for key in data.files:
        print(key)

    # --------------------------------------------------
    # 4. Inspect every array
    # --------------------------------------------------

    print("\nARRAY DETAILS")
    print("=" * 70)

    for key in data.files:

        array = data[key]

        print(f"\nKey: {key}")
        print("Shape:", array.shape)
        print("Dimensions:", array.ndim)
        print("Data type:", array.dtype)
        print("Size:", array.size)

        if array.size == 0:
            print("Empty array")
            continue

        if np.issubdtype(array.dtype, np.number):

            print("Minimum:", np.nanmin(array))
            print("Maximum:", np.nanmax(array))
            print("Mean:", np.nanmean(array))
            print("Standard deviation:", np.nanstd(array))

            unique_values = np.unique(array)

            if unique_values.size <= 30:
                print("Unique values:", unique_values)
            else:
                print("Number of unique values:", unique_values.size)
                print("First 30 unique values:", unique_values[:30])

        else:
            print("Non-numeric array")
            print("Sample:", array.flat[:5])

print("\nInspection completed.")



from pathlib import Path
import numpy as np

file_path = Path("data/cache_128/BraTSPED_00001.npz")

with np.load(file_path, allow_pickle=False) as data:

    image = data["image"].astype(np.float32)
    seg = data["seg"]
    modality_mask = data["modality_mask"]
    tumor_feats = data["tumor_feats"]
    has_seg = data["has_seg"]

    print("\n========== MRI CHANNEL STATISTICS ==========")

    for i in range(image.shape[0]):
        channel = image[i]

        print(f"\nChannel {i}")
        print("Shape:", channel.shape)
        print("Min:", np.min(channel))
        print("Max:", np.max(channel))
        print("Mean:", np.mean(channel))
        print("Std:", np.std(channel))
        print("Nonzero voxels:", np.count_nonzero(channel))
        print("NaN:", np.isnan(channel).sum())
        print("Inf:", np.isinf(channel).sum())

    print("\n========== MODALITY MASK ==========")
    print(modality_mask)

    print("\n========== SEGMENTATION LABEL COUNTS ==========")

    total_voxels = seg.size

    for label in np.unique(seg):

        count = np.sum(seg == label)
        percentage = count / total_voxels * 100

        print(
            f"Label {label}: "
            f"{count:,} voxels "
            f"({percentage:.4f}%)"
        )

    print("\n========== TUMOR FEATURES ==========")

    for i, value in enumerate(tumor_feats):
        print(f"Feature {i:02d}: {value:.6f}")

    print("\nHas segmentation:", has_seg)

