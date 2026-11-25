from pathlib import Path
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

ROOT = Path(".")
SUBJECT = "BraTSPED_00002"

IMAGE_DIR = ROOT / "nnUNet_raw/Dataset202_BraTS/imagesTr_all"
MASK_PATH = ROOT / "nnUNet_raw/Dataset202_BraTS/labelsTr_all" / f"{SUBJECT}.nii.gz"

SEQUENCES = {
    "T1n": "0000",
    "T1c": "0001",
    "T2f": "0002",
    "T2w": "0003",
}

def report(name, img):
    print(f"\n{name}")
    print("-" * 65)
    print("Shape:", img.shape)
    print("Orientation:", nib.aff2axcodes(img.affine))
    print("Voxel sizes:", nib.affines.voxel_sizes(img.affine))
    print("Affine:\n", np.array_str(img.affine, precision=3))

    qform, qcode = img.get_qform(coded=True)
    sform, scode = img.get_sform(coded=True)

    print("qform code:", qcode)
    print("sform code:", scode)

    if qform is not None:
        print("qform:\n", np.array_str(qform, precision=3))
    if sform is not None:
        print("sform:\n", np.array_str(sform, precision=3))


print("=" * 75)
print("ORIGINAL NIFTI GEOMETRY")
print("=" * 75)

raw_images = {}

for name, channel in SEQUENCES.items():
    path = IMAGE_DIR / f"{SUBJECT}_{channel}.nii.gz"
    raw_images[name] = nib.load(str(path))
    report(name, raw_images[name])

raw_mask = nib.load(str(MASK_PATH))
report("ORIGINAL SEGMENTATION", raw_mask)

print("\n" + "=" * 75)
print("CANONICAL RAS GEOMETRY")
print("=" * 75)

canonical_images = {}

for name, img in raw_images.items():
    canonical_images[name] = nib.as_closest_canonical(img)
    report(name + " canonical", canonical_images[name])

canonical_mask = nib.as_closest_canonical(raw_mask)
report("MASK canonical", canonical_mask)

reference = canonical_images["T1n"]

print("\n" + "=" * 75)
print("RESAMPLING SEGMENTATION TO T1n")
print("=" * 75)

print("T1n shape:", reference.shape)
print("Mask shape before:", canonical_mask.shape)

aligned_mask_img = resample_from_to(
    canonical_mask,
    (reference.shape, reference.affine),
    order=0
)

aligned_mask = np.rint(
    np.asarray(aligned_mask_img.dataobj)
).astype(np.int16)

print("Mask shape after:", aligned_mask.shape)
print("Aligned mask labels:", np.unique(aligned_mask))

original_data = np.rint(
    np.asarray(canonical_mask.dataobj)
).astype(np.int16)

for label, name in {
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}.items():
    before = int(np.sum(original_data == label))
    after = int(np.sum(aligned_mask == label))

    print(
        f"{name}: before={before:,}, "
        f"after={after:,}, "
        f"difference={after-before:+,}"
    )

print("\n" + "=" * 75)
print("WORLD-COORDINATE CENTROIDS")
print("=" * 75)

for label, name in {
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}.items():
    coords = np.argwhere(aligned_mask == label)

    if len(coords) == 0:
        print(name, ": absent")
        continue

    centroid_voxel = coords.mean(axis=0)
    centroid_ras = nib.affines.apply_affine(
        reference.affine,
        centroid_voxel
    )

    print(
        f"{name}: voxel XYZ={np.round(centroid_voxel, 2)}, "
        f"RAS mm={np.round(centroid_ras, 2)}"
    )

print("\n" + "=" * 75)
print("CONNECTED COMPONENTS ON AXIAL SLICE 78")
print("=" * 75)

z = 78

for label, name in [(1, "ET"), (2, "NET"), (3, "CC"), (4, "ED")]:
    mask = aligned_mask[:, :, z] == label
    components, count = ndimage.label(mask)

    print(f"\n{name}: {int(mask.sum()):,} voxels; {count} components")

    if count:
        sizes = np.bincount(components.ravel())[1:]

        for i, size in enumerate(sizes, start=1):
            coords = np.argwhere(components == i)
            print(
                f"  Component {i}: {size:,} voxels; "
                f"X={coords[:,0].min()}-{coords[:,0].max()}, "
                f"Y={coords[:,1].min()}-{coords[:,1].max()}"
            )

print("\n" + "=" * 75)
print("GEOMETRY CHECK COMPLETED")
print("=" * 75)
