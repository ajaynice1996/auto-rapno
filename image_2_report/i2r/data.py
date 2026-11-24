"""Dataset over cached volumes + report JSON targets + auxiliary attribute labels."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .attributes import (AUX_FINDINGS, AUX_LOCATIONS, GRADE_GROUPS, LATERALITIES, TRI_STATES, TUMOR_TYPES)
from .volume import NUM_TUMOR_FEATURES

CAT_FIELDS = [("tumor_type", TUMOR_TYPES), ("grade_group", GRADE_GROUPS), ("laterality", LATERALITIES)]


def encode_aux(attrs):
    cat = [labels.index(attrs[k]) for k, labels in CAT_FIELDS]
    find = [TRI_STATES.index(attrs["findings"][f]) for f in AUX_FINDINGS]
    loc = [float(attrs["locations"][l]) for l in AUX_LOCATIONS]
    return {"cat": torch.tensor(cat), "find": torch.tensor(find), "loc": torch.tensor(loc)}


def scale_tumor_feats(f):
    f = np.asarray(f, dtype=np.float32).copy()
    f[:6] = np.log1p(f[:6])  # volumes in mL span several orders of magnitude
    return torch.from_numpy(f)


class CaseDataset(Dataset):
    """Loads every case of a split into RAM (cached npz are ~17 MB each)."""

    def __init__(self, case_ids, reports, cache_dir, tokenizer=None, max_len=448, require_report=True):
        self.ids = list(case_ids)
        self.items = []
        for cid in self.ids:
            z = np.load(f"{cache_dir}/{cid}.npz")
            item = {
                "case_id": cid,
                "image": torch.from_numpy(z["image"]),
                "seg": torch.from_numpy(z["seg"]),
                "modality_mask": torch.from_numpy(z["modality_mask"]),
                "tumor_feats": scale_tumor_feats(z["tumor_feats"]),
            }
            rep = reports.get(cid)
            if rep is not None:
                item["aux"] = encode_aux(rep["attributes"])
                if tokenizer is not None:
                    item["ids"] = torch.tensor(tokenizer.encode(rep["target_json"], max_len))
            elif require_report:
                raise KeyError(cid)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id=0):
    out = {
        "case_id": [b["case_id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "seg": torch.stack([b["seg"] for b in batch]),
        "modality_mask": torch.stack([b["modality_mask"] for b in batch]),
        "tumor_feats": torch.stack([b["tumor_feats"] for b in batch]),
    }
    if "aux" in batch[0]:
        out["aux"] = {k: torch.stack([b["aux"][k] for b in batch]) for k in batch[0]["aux"]}
    if "ids" in batch[0]:
        L = max(len(b["ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, : len(b["ids"])] = b["ids"]
        out["ids"] = ids
    return out


assert NUM_TUMOR_FEATURES == 19
