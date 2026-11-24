"""Tokenizers for the report JSON text.

* ScratchTokenizer: byte-level BPE trained on the *training split only* (no leakage).
* HFTokenizer: wraps a pretrained tokenizer (e.g. facebook/bart-base, GanjinZero/biobart-v2-base).
Both expose encode(text)->ids (with bos/eos), decode(ids)->text, pad_id/bos_id/eos_id/vocab_size.
"""
import json
from pathlib import Path


class ScratchTokenizer:
    SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, tok):
        self.tok = tok
        self.pad_id, self.bos_id, self.eos_id = (tok.token_to_id(t) for t in self.SPECIALS[:3])
        self.vocab_size = tok.get_vocab_size()

    @classmethod
    def train(cls, texts, vocab_size=4000):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(texts, trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=cls.SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        return cls(tok)

    def save(self, path):
        self.tok.save(str(path))

    @classmethod
    def load(cls, path):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text, max_len=None):
        ids = [self.bos_id] + self.tok.encode(text).ids + [self.eos_id]
        if max_len and len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.tok.decode(ids)


class HFTokenizer:
    def __init__(self, name_or_path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.name = name_or_path
        self.pad_id, self.bos_id, self.eos_id = self.tok.pad_token_id, self.tok.bos_token_id, self.tok.eos_token_id
        self.vocab_size = len(self.tok)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tok.save_pretrained(str(Path(path).parent / "hf_tokenizer"))
        json.dump({"name": self.name}, open(path, "w"))

    @classmethod
    def load(cls, path):
        info = json.load(open(path))
        local = Path(path).parent / "hf_tokenizer"
        return cls(str(local) if local.exists() else info["name"])

    def encode(self, text, max_len=None):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        ids = [self.bos_id] + ids + [self.eos_id]
        if max_len and len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.tok.decode(ids, skip_special_tokens=True)
