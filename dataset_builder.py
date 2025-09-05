# dataset_builder.py
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Iterable
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def collate_with_labels(tokenizer, max_length: int):
    """
    Trainer-native collator: pad-to-longest with tokenizer and create labels
    with -100 on pad positions. No re-tokenization overhead beyond batch tokenization.
    """
    def _collate(batch_texts):
        tok = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,           # pad to longest in *this* batch
            return_tensors="pt"
        )
        input_ids = tok["input_ids"]
        attention = tok["attention_mask"]
        labels = input_ids.clone()
        labels[attention == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
    return _collate


class TextListMapDataset(Dataset):
    """Map-style dataset that returns one raw text per index (tokenized in collator)."""
    def __init__(self, texts):
        self.texts = list(texts)
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]


class TextListDataset(Dataset):
    def __init__(self, texts: List[str], lengths: Optional[List[int]] = None):
        self.texts = texts
        self.lengths = lengths or [len(t) for t in texts]
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]

def _collate_builder(tokenizer, max_length):
    def collate(batch_texts):
        tok = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,          # pad to longest in the batch
            return_tensors="pt"
        )
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
    return collate


class MixtureDataBuilder:
    """
    Reusable builder for task mixtures (syntax/code/math/logic) with token budgets.
    Returns DataLoader(s) that pad to longest within each batch (via tokenizer).
    Caller should set tokenizer.padding_side = 'left' for decoder-only LMs.
    """

    def __init__(self, tokenizer, max_length: int = 512, seed: int = 42, verbose: bool = True):
        self.tok = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.verbose = verbose
        self._rng = random.Random(seed)

    # ----------------- split helpers & policy -----------------

    def _try_load(self, *args, **kwargs):
        """Small helper: wrap load_dataset calls to gracefully fail over."""
        try:
            return load_dataset(*args, **kwargs)
        except Exception:
            return None

    def _prefer_splits(self, preferred: str) -> List[str]:
        """Return ordered split preference chain for a requested split."""
        if preferred == "test":
            return ["test", "validation", "valid", "dev", "train"]
        if preferred == "validation" or preferred == "valid":
            return ["validation", "valid", "dev", "train"]
        # default: train preference chain
        return ["train", "validation", "valid", "dev", "test"]

    # Each loader returns (iterable, get_text, used_split_name)
    # They are now split-aware and will fall back sensibly if the split is missing.

    def _load_syntax_split(self, split: str):
        # wikitext-2-raw-v1 has train/validation/test
        for s in self._prefer_splits(split):
            ds = self._try_load("wikitext", name="wikitext-2-raw-v1", split=s)
            if ds is not None:
                def get_text(ex): return (ex.get("text") or "").strip()
                return ds, get_text, s
        raise RuntimeError("Failed to load any split for syntax.")

    def _load_code_split(self, split: str):
        parts = []
        # codeparrot-clean (train only, typically)
        for s in self._prefer_splits(split):
            ds1 = self._try_load("codeparrot/codeparrot-clean", split=s)
            if ds1 is not None:
                parts.append(("codeparrot", ds1, lambda ex: (ex.get("content") or ex.get("code") or ex.get("text") or "").strip(), s))
                break
        # deepmind/code_contests (train typically)
        for s in self._prefer_splits(split):
            ds2 = self._try_load("deepmind/code_contests", split=s)
            if ds2 is not None:
                parts.append(("codecontests", ds2, lambda ex: (ex.get("problem") or ex.get("problem_statement") or "").strip(), s))
                break

        if not parts:
            # fallback: syntax split
            ds, fn, used = self._load_syntax_split(split)
            return ds, fn, used

        def chained_iter():
            # Shuffle ordering between subparts deterministically
            idx_order = list(range(len(parts)))
            self._rng.shuffle(idx_order)
            for i in idx_order:
                _, dsi, _, _ = parts[i]
                # try to use Dataset.shuffle if present for determinism
                try:
                    dsi = dsi.shuffle(seed=self.seed)
                except Exception:
                    pass
                for ex in dsi:
                    yield ex

        class _Chain:
            def __iter__(self) -> Iterable: return chained_iter()

        def get_text(ex):
            for _, _, fn, _ in parts:
                t = fn(ex)
                if t: return t
            return ""

        # If multiple parts used different fallbacks, report the first one that succeeded
        used_split_report = next((sp for *_, sp in parts), split)
        return _Chain(), get_text, used_split_report

    def _load_math_split(self, split: str):
        # gsm8k main has train/test
        for s in self._prefer_splits(split):
            ds = self._try_load("gsm8k", "main", split=s)
            if ds is not None:
                def get_text(ex): return (ex.get("question") or ex.get("problem") or ex.get("input") or "").strip()
                return ds, get_text, s
        # fallback: hellaswag (val/test/train exist)
        for s in self._prefer_splits(split):
            ds = self._try_load("hellaswag", split=s)
            if ds is not None:
                def get_text(ex): return (ex.get("ctx") or "").strip()
                return ds, get_text, s
        raise RuntimeError("Failed to load any split for math.")

    def _load_logic_split(self, split: str):
        for s in self._prefer_splits(split):
            ds = self._try_load("hellaswag", split=s)
            if ds is not None:
                def get_text(ex): return (ex.get("ctx") or "").strip()
                return ds, get_text, s
        raise RuntimeError("Failed to load any split for logic.")

    REGISTRY_SPLIT = {
        "syntax": _load_syntax_split,
        "code":   _load_code_split,
        "math":   _load_math_split,
        "logic":  _load_logic_split,
    }

    # --------------- core sampler (dedup + token-budget) ----------------

    def _gather_for_budget(
        self,
        domain: str,
        preferred_split: str,
        target_tokens: int,
        exclude_texts: set,
        summary_log: dict,
        allow_val_from_train: bool = True,
    ) -> Tuple[List[str], int, str]:
        """
        Collect texts for a domain until ~target_tokens are reached, using the split policy.
        Ensures texts are mutually exclusive across other splits by skipping exclude_texts.
        Returns (picked_texts, token_count, used_split_name).
        """
        ds, get_text, used_split = self.REGISTRY_SPLIT[domain](self, preferred_split)

        if self.verbose and used_split != preferred_split:
            print(f"[Mixture] Fallback for domain '{domain}': requested '{preferred_split}' but using '{used_split}'")

        # If we asked for validation but ended up on train because no val split exists,
        # and allow_val_from_train=True, we'll still *partition* from train using exclude_texts
        # so it's disjoint from actual train.
        picked, tok_count, buf = [], 0, []

        # Try to shuffle for determinism if Dataset supports it
        try:
            ds = ds.shuffle(seed=self.seed)
        except Exception:
            pass

        for ex in ds:
            t = get_text(ex)
            if not t or len(t) < 32:
                continue
            if t in exclude_texts:
                continue  # keep splits mutually exclusive
            buf.append(t)
            if len(buf) >= 256:
                enc = self.tok(buf, truncation=True, max_length=self.max_length, add_special_tokens=True)
                for txt, ids in zip(buf, enc["input_ids"]):
                    if txt in exclude_texts:  # re-check after batching (paranoia)
                        continue
                    tok_len = len(ids)
                    picked.append(txt)
                    exclude_texts.add(txt)
                    tok_count += tok_len
                    if tok_count >= target_tokens:
                        summary_log["used_splits"][domain] = used_split
                        return picked, tok_count, used_split
                buf.clear()

        if buf:
            enc = self.tok(buf, truncation=True, max_length=self.max_length, add_special_tokens=True)
            for txt, ids in zip(buf, enc["input_ids"]):
                if txt in exclude_texts:
                    continue
                tok_len = len(ids)
                picked.append(txt)
                exclude_texts.add(txt)
                tok_count += tok_len

        summary_log["used_splits"][domain] = used_split
        return picked, tok_count, used_split

    def make_dataloader(
            self,
            texts_sorted,
            batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = True,
            shuffle: bool = False,
    ):
        ds = TextListDataset(texts_sorted)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_builder(self.tok, self.max_length),
            persistent_workers=(num_workers > 0),
        )

    # --------------- single-mixture builder (unchanged API) ---------------

    def build_text_mixture(self, token_targets: Dict[str, int]) -> Tuple[List[str], List[int], Dict]:
        """
        Sample texts per domain until ~token_targets[domain] tokens are reached, from default policy
        (syntax->val, code->train, math->train, logic->val historically). Now uses split-aware loaders
        with sensible fallbacks. Returns (texts_sorted_by_len_desc, lengths, summary_dict).
        """
        # Preserve your original console prints and behavior…
        if self.verbose:
            print(f"Building project mixture with token targets {token_targets} (max_length={self.max_length})")

        # Choose default preferred split per domain (same spirit as your originals)
        default_pref = {"syntax": "validation", "code": "train", "math": "train", "logic": "validation"}

        texts_all, per_domain, used_splits = [], {}, {}
        exclude_texts = set()
        summary_log = {"used_splits": used_splits}

        for dom, tgt in (token_targets or {}).items():
            if dom not in self.REGISTRY_SPLIT or tgt <= 0:
                continue
            texts, n_tok, _used = self._gather_for_budget(dom, default_pref.get(dom, "train"), int(tgt), exclude_texts, summary_log)
            texts_all.extend(texts)
            per_domain[dom] = int(n_tok)

        if not texts_all:
            return [], [], {"domains": {}, "total_texts": 0}

        enc = self.tok(texts_all, truncation=True, max_length=self.max_length, add_special_tokens=True)
        lengths = [len(ids) for ids in enc["input_ids"]]
        order = np.argsort(np.array(lengths))[::-1]
        texts_sorted = [texts_all[i] for i in order]
        lengths_sorted = [lengths[i] for i in order]
        summary = {
            "domains": per_domain,
            "total_texts": len(texts_sorted),
            "pad_style": "pad-to-longest-per-batch",
            "token_max_length": self.max_length,
            "used_splits": used_splits,
            "lengths": lengths_sorted,
        }
        # if self.verbose:
        #     print(f"Prepared project mix: {summary}")
        return texts_sorted, lengths_sorted, summary

    # --------------- multi-split mixtures (mutually exclusive) ---------------

    def build_split_mixtures(
            self,
            token_targets_by_split: Dict[str, Dict[str, int]],
            *,
            val_from_train_when_missing: bool = True,
            order_of_splits: Optional[List[str]] = None,
    ):
        """
        Build mutually exclusive mixtures only for the splits explicitly provided
        in token_targets_by_split. For each requested split, each domain tries that
        split first and then falls back (e.g., test→validation→train) if needed.
        """
        if not token_targets_by_split:
            return {}, {}, {}

        # default build order: reserve scarce eval material first
        default_order = ["test", "validation", "train"]
        order = order_of_splits or default_order

        # strictly honor requested keys; keep their relative order according to 'order'
        requested = [s for s in order if s in token_targets_by_split] + \
                    [s for s in token_targets_by_split.keys() if s not in order]

        exclude_texts = set()
        texts_by_split, lengths_by_split, summary_by_split = {}, {}, {}

        for split_name in requested:
            dom_targets = token_targets_by_split.get(split_name, {})
            if not dom_targets:
                texts_by_split[split_name] = []
                lengths_by_split[split_name] = []
                summary_by_split[split_name] = {"domains": {}, "total_texts": 0, "used_splits": {}}
                continue

            texts_all, per_domain, used_splits = [], {}, {}
            summary_log = {"used_splits": used_splits}

            for dom, tgt in dom_targets.items():
                if dom not in self.REGISTRY_SPLIT or tgt <= 0:
                    continue
                picked, n_tok, used_split = self._gather_for_budget(
                    dom, split_name, int(tgt), exclude_texts, summary_log,
                    allow_val_from_train=val_from_train_when_missing,
                )
                texts_all.extend(picked)
                per_domain[dom] = int(n_tok)
                used_splits[dom] = used_split

            if not texts_all:
                texts_by_split[split_name] = []
                lengths_by_split[split_name] = []
                summary_by_split[split_name] = {"domains": {}, "total_texts": 0, "used_splits": used_splits}
                if self.verbose:
                    print(
                        f"[Mixture] Built empty '{split_name}' (no available items for budgets). used_splits={used_splits}")
                continue

            enc = self.tok(texts_all, truncation=True, max_length=self.max_length, add_special_tokens=True)
            lengths = [len(ids) for ids in enc["input_ids"]]
            order_idx = np.argsort(np.array(lengths))[::-1]
            texts_sorted = [texts_all[i] for i in order_idx]
            lengths_sorted = [lengths[i] for i in order_idx]

            texts_by_split[split_name] = texts_sorted
            lengths_by_split[split_name] = lengths_sorted
            summary_by_split[split_name] = {
                "domains": per_domain,
                "total_texts": len(texts_sorted),
                "pad_style": "pad-to-longest-per-batch",
                "token_max_length": self.max_length,
                "used_splits": used_splits,
            }

            if self.verbose:
                print(f"[Mixture] Built '{split_name}': domains={per_domain} | used_splits={used_splits}")

        return texts_by_split, lengths_by_split, summary_by_split

    # --------------- NEW: convenience to get DataLoaders per split ---------------

    def make_dataloaders_for_splits(
        self,
        texts_by_split: Dict[str, List[str]],
        *,
        batch_sizes: Dict[str, int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle: bool = False,
    ) -> Dict[str, DataLoader]:
        """
        Turn the output of build_split_mixtures(...) into one DataLoader per split.
        - batch_sizes: per-split batch size (default 8 for any split not specified).
        """
        dls = {}
        for split_name, texts in texts_by_split.items():
            if not texts:
                dls[split_name] = None
                continue
            bs = (batch_sizes or {}).get(split_name, 8)
            dls[split_name] = self.make_dataloader(
                texts_sorted=texts,
                batch_size=bs,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=shuffle,  # keep False to preserve length-sorted batches
            )
        return dls

    def as_trainer_dataset(self, texts_sorted):
        """
        Return a map-style dataset for Trainer (one text per item).
        Trainer will build the DataLoader and call our collator to tokenize/pad/label.
        """
        return TextListMapDataset(texts_sorted)
