"""3D image encoder (+ optional segmentation branch) -> cross-attention text decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attributes import AUX_FINDINGS, AUX_LOCATIONS, GRADE_GROUPS, LATERALITIES, TRI_STATES, TUMOR_TYPES
from .volume import MODALITIES, NUM_TUMOR_FEATURES

N_SEG_CHANNELS = 4  # one-hot ET / NET / CC / ED (background implicit)


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        g = lambda c: nn.GroupNorm(8, c)
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = g(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = g(cout)
        self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), g(cout)) if (cin != cout or stride != 1) else nn.Identity()

    def forward(self, x):
        y = F.silu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.silu(y + self.skip(x))


class Encoder3D(nn.Module):
    """Residual 3D CNN. A (B, C, S, S, S) volume -> (B, 64, d) tokens (4x4x4 grid)."""

    def __init__(self, in_ch, d_model=256, widths=(32, 64, 128, 192), grid=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_ch, widths[0], 3, 2, 1, bias=False), nn.GroupNorm(8, widths[0]), nn.SiLU())
        blocks, c = [], widths[0]
        for w in widths:
            blocks += [ResBlock3D(c, w, 2), ResBlock3D(w, w, 1)]
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv3d(c, d_model, 1)
        self.grid = grid
        self.pos = nn.Parameter(torch.zeros(1, grid ** 3, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.proj(self.blocks(self.stem(x)))
        x = F.adaptive_avg_pool3d(x, self.grid).flatten(2).transpose(1, 2)
        return x + self.pos


class AuxHeads(nn.Module):
    """Predict report attributes (tumor type, grade group, laterality, findings, locations) from pooled features."""

    def __init__(self, d):
        super().__init__()
        self.cat = nn.ModuleList([nn.Linear(d, n) for n in (len(TUMOR_TYPES), len(GRADE_GROUPS), len(LATERALITIES))])
        self.find = nn.Linear(d, len(AUX_FINDINGS) * len(TRI_STATES))
        self.loc = nn.Linear(d, len(AUX_LOCATIONS))

    def forward(self, pooled):
        return {
            "cat": [h(pooled) for h in self.cat],
            "find": self.find(pooled).view(-1, len(AUX_FINDINGS), len(TRI_STATES)),
            "loc": self.loc(pooled),
        }

    @staticmethod
    def loss(out, aux):
        l = sum(F.cross_entropy(o, aux["cat"][:, i]) for i, o in enumerate(out["cat"])) / len(out["cat"])
        l = l + F.cross_entropy(out["find"].flatten(0, 1), aux["find"].flatten())
        l = l + F.binary_cross_entropy_with_logits(out["loc"], aux["loc"])
        return l / 3

    @staticmethod
    def decode(out):
        res = {
            "tumor_type": [TUMOR_TYPES[i] for i in out["cat"][0].argmax(-1).tolist()],
            "grade_group": [GRADE_GROUPS[i] for i in out["cat"][1].argmax(-1).tolist()],
            "laterality": [LATERALITIES[i] for i in out["cat"][2].argmax(-1).tolist()],
        }
        find = out["find"].argmax(-1).tolist()
        loc = (out["loc"].sigmoid() > 0.5).long().tolist()
        res["findings"] = [{f: TRI_STATES[v] for f, v in zip(AUX_FINDINGS, row)} for row in find]
        res["locations"] = [{l: v for l, v in zip(AUX_LOCATIONS, row)} for row in loc]
        return res


class ScratchDecoder(nn.Module):
    """Transformer decoder with cross-attention over visual tokens, trained from scratch."""

    def __init__(self, vocab_size, pad_id, bos_id, eos_id, d_model=256, layers=4, heads=8, ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=d_model ** -0.5)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.max_len = max_len

    def logits(self, visual, ids):
        L = ids.size(1)
        x = self.embed(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        causal = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), 1)
        return self.out(self.dec(x, visual, tgt_mask=causal, tgt_key_padding_mask=ids.eq(self.pad_id), tgt_is_causal=True))

    def loss(self, visual, ids, label_smoothing=0.1):
        lg = self.logits(visual, ids[:, :-1])
        return F.cross_entropy(lg.flatten(0, 1).float(), ids[:, 1:].flatten(), ignore_index=self.pad_id, label_smoothing=label_smoothing)

    @torch.no_grad()
    def generate(self, visual, max_len=448, min_len=20):
        B = visual.size(0)
        ids = torch.full((B, 1), self.bos_id, device=visual.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=visual.device)
        for t in range(min(max_len, self.max_len) - 1):
            lg = self.logits(visual, ids)[:, -1].float()
            if t < min_len:
                lg[:, self.eos_id] = float("-inf")
            nxt = lg.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt.eq(self.eos_id)
            if done.all():
                break
        return [self._strip(r) for r in ids.tolist()]

    def _strip(self, row):
        row = row[1:]
        return row[: row.index(self.eos_id)] if self.eos_id in row else [t for t in row if t != self.pad_id]


class HFSeq2SeqDecoder(nn.Module):
    """Pretrained encoder-decoder LM (e.g. facebook/bart-base, GanjinZero/biobart-v2-base).

    Visual tokens are projected into the LM embedding space and fed to the LM *encoder* as inputs_embeds,
    so the pretrained text decoder attends to the image through its own cross-attention.
    """

    def __init__(self, name, d_visual, pad_id, eos_id):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(name)
        d = self.lm.config.d_model
        self.proj = nn.Sequential(nn.LayerNorm(d_visual), nn.Linear(d_visual, d))
        self.pad_id, self.eos_id = pad_id, eos_id

    def loss(self, visual, ids, label_smoothing=0.1):
        labels = ids.masked_fill(ids.eq(self.pad_id), -100)
        out = self.lm(inputs_embeds=self.proj(visual), labels=labels)
        return F.cross_entropy(out.logits.flatten(0, 1).float(), labels.flatten(), ignore_index=-100, label_smoothing=label_smoothing)

    @torch.no_grad()
    def generate(self, visual, max_len=448, min_len=20):
        emb = self.proj(visual)
        out = self.lm.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=emb.device, dtype=torch.long),
                               max_length=max_len, min_length=min_len, num_beams=1, do_sample=False)
        res = []
        for row in out.tolist():
            row = [t for t in row[1:] if t != self.pad_id]
            res.append(row[: row.index(self.eos_id)] if self.eos_id in row else row)
        return res


class Image2Report(nn.Module):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.use_seg = cfg["use_seg"]
        d = cfg["d_model"]
        in_ch = len(MODALITIES) + (N_SEG_CHANNELS if self.use_seg else 0)
        self.encoder = Encoder3D(in_ch, d, grid=cfg.get("grid", 4))
        self.mod_embed = nn.Linear(len(MODALITIES), d)
        if self.use_seg:
            self.tumor_mlp = nn.Sequential(nn.Linear(NUM_TUMOR_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.aux = AuxHeads(d)
        if cfg["decoder"] == "scratch":
            self.decoder = ScratchDecoder(tokenizer.vocab_size, tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, d_model=d,
                                          layers=cfg.get("dec_layers", 4), max_len=cfg.get("max_len", 448) + 8)
        else:
            self.decoder = HFSeq2SeqDecoder(cfg["decoder"], d, tokenizer.pad_id, tokenizer.eos_id)

    def encode(self, batch):
        x = batch["image"].float()
        if self.use_seg:
            seg1h = F.one_hot(batch["seg"].long(), N_SEG_CHANNELS + 1)[..., 1:].permute(0, 4, 1, 2, 3).float()
            x = torch.cat([x, seg1h], 1)
        tokens = self.encoder(x) + self.mod_embed(batch["modality_mask"].float())[:, None]
        pooled = tokens.mean(1)
        if self.use_seg:
            tokens = torch.cat([self.tumor_mlp(batch["tumor_feats"].float())[:, None], tokens], 1)
        return tokens, pooled

    def forward(self, batch, aux_weight=0.5):
        tokens, pooled = self.encode(batch)
        lm = self.decoder.loss(tokens, batch["ids"], self.cfg.get("label_smoothing", 0.1))
        aux = AuxHeads.loss(self.aux(pooled), batch["aux"]) if aux_weight > 0 else torch.zeros((), device=lm.device)
        return {"loss": lm + aux_weight * aux, "lm_loss": lm.detach(), "aux_loss": aux.detach()}

    @torch.no_grad()
    def generate(self, batch, max_len=448):
        tokens, pooled = self.encode(batch)
        ids = self.decoder.generate(tokens, max_len=max_len)
        return ids, AuxHeads.decode(self.aux(pooled))
