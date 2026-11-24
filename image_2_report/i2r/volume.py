"""MRI / segmentation loading, normalisation, resampling and tumor measurements."""
import numpy as np

MODALITIES = ["t1n", "t1c", "t2f", "t2w"]  # nnUNet channel order _0000.._0003
SEG_LABELS = {1: "ET", 2: "NET", 3: "CC", 4: "ED"}
TUMOR_FEATURE_NAMES = [
    "vol_ET_ml", "vol_NET_ml", "vol_CC_ml", "vol_ED_ml", "vol_whole_tumor_ml", "vol_core_ml",
    "frac_ET", "frac_NET", "frac_CC", "frac_ED", "frac_tumor_of_brain",
    "cx", "cy", "cz", "ex", "ey", "ez", "midline_offset_x", "n_components_norm",
]
NUM_TUMOR_FEATURES = len(TUMOR_FEATURE_NAMES)


def _nib():
    import nibabel as nib
    return nib


def load_nifti(path):
    img = _nib().load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img.header.get_zooms()[:3]


def brain_bbox(volumes, margin=2):
    """Bounding box (slices) of the non-zero (brain) region over all available modalities."""
    union = np.zeros(volumes[0].shape, dtype=bool)
    for v in volumes:
        union |= v != 0
    if not union.any():
        return tuple(slice(0, s) for s in union.shape)
    idx = np.where(union)
    return tuple(slice(max(int(i.min()) - margin, 0), min(int(i.max()) + 1 + margin, s)) for i, s in zip(idx, union.shape))


def zscore_in_brain(v):
    mask = v != 0
    if mask.sum() < 10:
        return np.zeros_like(v)
    mu, sd = v[mask].mean(), v[mask].std() + 1e-6
    out = (v - mu) / sd
    out[~mask] = 0
    return np.clip(out, -5, 5)


def resize(vol, size, order):
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(size, vol.shape)]
    return zoom(vol, factors, order=order, mode="nearest", prefilter=order > 1)


def tumor_features(seg, zooms, brain_shape_box):
    """Measurements from a full-resolution label map. Positions are normalised to the brain bounding box."""
    vox_ml = float(np.prod(zooms)) / 1000.0
    labels = {n: (seg == k) for k, n in SEG_LABELS.items()}
    vols = {n: float(m.sum()) * vox_ml for n, m in labels.items()}
    whole = seg > 0
    core = (seg == 1) | (seg == 2) | (seg == 3)
    v_whole = float(whole.sum()) * vox_ml
    v_core = float(core.sum()) * vox_ml
    brain_vox = float(np.prod([s.stop - s.start for s in brain_shape_box]))
    feats = [vols["ET"], vols["NET"], vols["CC"], vols["ED"], v_whole, v_core]
    feats += [vols[n] / v_whole if v_whole > 0 else 0.0 for n in ("ET", "NET", "CC", "ED")]
    feats += [float(whole.sum()) / brain_vox]
    if whole.any():
        idx = np.argwhere(whole)
        c = []
        e = []
        for ax in range(3):
            lo, hi = brain_shape_box[ax].start, brain_shape_box[ax].stop
            span = max(hi - lo, 1)
            c.append((idx[:, ax].mean() - lo) / span)
            e.append((idx[:, ax].max() - idx[:, ax].min() + 1) / span)
        from scipy.ndimage import label as cc_label
        n_comp = cc_label(whole)[1]
        feats += c + e + [c[0] - 0.5, min(n_comp, 10) / 10.0]
    else:
        feats += [0.0] * 8
    return np.asarray(feats, dtype=np.float32)


def preprocess_case(modality_paths, seg_path=None, size=128):
    """modality_paths: dict modality-> path or None (missing). Returns dict with arrays.

    image: (4, S, S, S) float16 z-scored, zeros for missing modalities
    seg:   (S, S, S) uint8 resized label map (nearest) or zeros when unavailable
    modality_mask: (4,) 1 if the modality is present
    tumor_feats: (NUM_TUMOR_FEATURES,) measurements from the original-resolution segmentation
    """
    vols, zooms = {}, None
    for m in MODALITIES:
        p = modality_paths.get(m)
        if p:
            vols[m], zooms = load_nifti(p)
    if not vols:
        raise ValueError("no modality available")
    shape = next(iter(vols.values())).shape
    box = brain_bbox(list(vols.values()))
    chans = []
    mask = np.zeros(len(MODALITIES), dtype=np.float32)
    for i, m in enumerate(MODALITIES):
        if m in vols:
            chans.append(resize(zscore_in_brain(vols[m])[box], (size,) * 3, 1))
            mask[i] = 1
        else:
            chans.append(np.zeros((size,) * 3, dtype=np.float32))
    image = np.stack(chans).astype(np.float16)
    if seg_path:
        seg_full, _ = load_nifti(seg_path)
        seg_full = np.rint(seg_full).astype(np.uint8)
        if seg_full.shape != shape:
            raise ValueError(f"segmentation shape {seg_full.shape} != image shape {shape}")
        seg = resize(seg_full[box], (size,) * 3, 0).astype(np.uint8)
        feats = tumor_features(seg_full, zooms, box)
        has_seg = 1
    else:
        seg, feats, has_seg = np.zeros((size,) * 3, np.uint8), np.zeros(NUM_TUMOR_FEATURES, np.float32), 0
    return {"image": image, "seg": seg, "modality_mask": mask, "tumor_feats": feats, "has_seg": has_seg}
