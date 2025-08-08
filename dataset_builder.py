# dataset_builder.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class TextListDataset(Dataset):
    def __init__(self, texts: List[str], lengths: Optional[List[int]] = None):
        self.texts = texts
        self.lengths = lengths or [len(t) for t in texts]
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]

class MixtureDataBuilder:
    """
    Reusable builder for task mixtures (syntax/code/math/logic) with token budgets.
    Returns a DataLoader that pads to the longest within each batch (via tokenizer).
    Caller should set tokenizer.padding_side = 'left' for decoder-only LMs.
    """

    def __init__(self, tokenizer, max_length: int = 512, seed: int = 42, verbose: bool = True):
        self.tok = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.verbose = verbose

    # ---- sources ----
    def _load_syntax(self):
        ds = load_dataset("wikitext", name="wikitext-2-raw-v1", split="validation")
        def get_text(ex): return (ex.get("text") or "").strip()
        return ds, get_text

    def _load_code(self):
        # Prefer codeparrot; add code_contests if available
        parts = []
        try:
            ds1 = load_dataset("codeparrot/codeparrot-clean", split="train")
            parts.append(("codeparrot", ds1, lambda ex: (ex.get("content") or ex.get("code") or ex.get("text") or "").strip()))
        except Exception:
            pass
        try:
            ds2 = load_dataset("deepmind/code_contests", split="train")
            parts.append(("codecontests", ds2, lambda ex: (ex.get("problem") or ex.get("problem_statement") or "").strip()))
        except Exception:
            pass
        if not parts:
            return self._load_syntax()  # fallback

        def chained_iter():
            for _, dsi, _ in parts:
                for ex in dsi:
                    yield ex
        class _Chain:
            def __iter__(self): return chained_iter()
        def get_text(ex):
            for _, _, fn in parts:
                t = fn(ex)
                if t: return t
            return ""
        return _Chain(), get_text

    def _load_math(self):
        try:
            ds = load_dataset("gsm8k", "main", split="train")
            def get_text(ex): return (ex.get("question") or ex.get("problem") or ex.get("input") or "").strip()
            return ds, get_text
        except Exception:
            ds = load_dataset("hellaswag", split="validation")
            def get_text(ex): return (ex.get("ctx") or "").strip()
            return ds, get_text

    def _load_logic(self):
        ds = load_dataset("hellaswag", split="validation")
        def get_text(ex): return (ex.get("ctx") or "").strip()
        return ds, get_text

    REGISTRY = {
        "syntax": _load_syntax,
        "code":   _load_code,
        "math":   _load_math,
        "logic":  _load_logic,
    }

    # ---- builder ----
    def build_text_mixture(self, token_targets: Dict[str, int]) -> Tuple[List[str], List[int], Dict]:
        """
        Sample texts per domain until ~token_targets[domain] tokens are reached.
        Returns (texts_sorted_by_len_desc, lengths, summary_dict)
        """
        if self.verbose:
            print(f"Building project mixture with token targets {token_targets} (max_length={self.max_length})")

        def gather(domain: str, target_tokens: int) -> Tuple[List[str], int]:
            ds, get_text = self.REGISTRY[domain](self)
            picked, tok_count, buf = [], 0, []
            pbar = tqdm(total=target_tokens, desc=f"{domain:>6} tokens", unit="tok",
                        leave=False) if self.verbose else None
            for ex in ds:
                t = get_text(ex)
                if len(t) < 32:
                    continue
                buf.append(t)
                if len(buf) >= 256:
                    enc = self.tok(buf, truncation=True, max_length=self.max_length, add_special_tokens=True)
                    for txt, ids in zip(buf, enc["input_ids"]):
                        tok_len = len(ids)
                        tok_count += tok_len
                        if pbar: pbar.update(tok_len)
                        picked.append(txt)
                        if tok_count >= target_tokens:
                            if pbar: pbar.close()
                            return picked, tok_count
                    buf.clear()
            if buf:
                enc = self.tok(buf, truncation=True, max_length=self.max_length, add_special_tokens=True)
                for txt, ids in zip(buf, enc["input_ids"]):
                    tok_len = len(ids)
                    tok_count += tok_len
                    if pbar: pbar.update(tok_len)
                    picked.append(txt)
            if pbar: pbar.close()
            return picked, tok_count

        texts_all, per_domain = [], {}
        for dom, tgt in (token_targets or {}).items():
            if dom not in self.REGISTRY or tgt <= 0: continue
            texts, n_tok = gather(dom, int(tgt))
            texts_all.extend(texts)
            per_domain[dom] = int(n_tok)

        if not texts_all:
            return [], [], {"domains": {}, "total_texts": 0}

        enc = self.tok(texts_all, truncation=True, max_length=self.max_length, add_special_tokens=True)
        lengths = [len(ids) for ids in enc["input_ids"]]
        order = np.argsort(np.array(lengths))[::-1]  # longest first
        texts_sorted = [texts_all[i] for i in order]
        lengths_sorted = [lengths[i] for i in order]
        summary = {"domains": per_domain, "total_texts": len(texts_sorted), "pad_style": "pad-to-longest-per-batch", "token_max_length": self.max_length}
        if self.verbose:
            print(f"Prepared project mix: {summary}")
        return texts_sorted, lengths_sorted, summary

    # ---- dataloader ----
    def make_dataloader(
        self,
        texts_sorted: List[str],
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Returns a DataLoader that tokenizes in the collate_fn and pads to the longest in the batch.
        """
        ds = TextListDataset(texts_sorted)

        def collate(batch_texts: List[str]):
            tok = self.tok(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,      # pad to longest in this batch
                return_tensors="pt"
            )
            return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,               # keep False to preserve length-sorting
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
