import os, re, gc, json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_dataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from dataset_builder import MixtureDataBuilder

# -----------------------
# Utility helpers
# -----------------------

def _slugify(name: str) -> str:
    # File-system friendly id: "Qwen/Qwen2.5-7B" -> "qwen-qwen2.5-7b"
    slug = name.strip().lower()
    slug = slug.replace("/", "-").replace(" ", "-")
    slug = re.sub(r"[^a-z0-9\-_.]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug

def _pick_torch_dtype(dtype: str) -> torch.dtype:
    """Map user string to torch dtype with safety checks."""
    d = dtype.lower()
    if d in {"bf16", "bfloat16"}:
        # prefer bf16 when supported
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # CPU bf16 is not universally reliable; fall back to fp16 / fp32
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if d in {"fp16", "float16", "half"}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if d in {"fp32", "float32"}:
        return torch.float32
    # default safe fallback
    return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

def _arch_probe(model):
    """
    Try to locate:
      - the list of decoder layers
      - the input embedding module

    Supports Llama/Mistral/Gemma/Qwen families, GPT-2/NeoX, and generic HF causal models.
    """
    # Llama/Mistral/Gemma/Qwen style
    if hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "embed_tokens"):
        return model.model.layers, model.model.embed_tokens

    # Some Qwen variants expose top-level 'layers' and 'embed_tokens'
    if hasattr(model, "layers") and hasattr(model, "embed_tokens"):
        return model.layers, model.embed_tokens

    # GPT-2 family
    if hasattr(model, "transformer") and hasattr(model.transformer, "h") and hasattr(model.transformer, "wte"):
        return model.transformer.h, model.transformer.wte

    # GPT-NeoX family
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers") and hasattr(model.gpt_neox, "embed_in"):
        return model.gpt_neox.layers, model.gpt_neox.embed_in

    # Fallback: treat as generic decoder-only with config count
    # In this case we will hook the top module (may not always work)
    if hasattr(model.config, "num_hidden_layers"):
        # Best-effort: try common attribute paths
        for path in [
            ("model", "layers"),
            ("transformer", "h"),
            ("decoder", "layers"),
        ]:
            obj = model
            ok = True
            for p in path:
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (list, tuple)):
                # try to discover embedding alongside
                embed = None
                for epath in [
                    ("model", "embed_tokens"),
                    ("transformer", "wte"),
                    ("decoder", "embed_tokens"),
                ]:
                    eobj = model
                    eok = True
                    for ep in epath:
                        if hasattr(eobj, ep):
                            eobj = getattr(eobj, ep)
                        else:
                            eok = False
                            break
                    if eok:
                        embed = eobj
                        break
                return obj, embed
    raise ValueError("Model architecture not recognized (failed to locate layers/embeddings).")


class LayerPruningAnalyzer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        results_dir: str = "./results/analysis",
        device: str = "cuda",
        use_8bit: bool = False,             # default: stock model (no quantization)
        use_4bit: bool = False,
        dtype: str = "bf16",                # default to BF16 when supported
        batch_size: int = 4,                # for evaluation-time batching
        max_samples: int = 200,             # cap to avoid long runs by default
        trust_remote_code: bool = True,
    ):
        """
        Analyzer for transformer layer pruning similarity.

        Args:
            model_name: HF hub model id
            results_dir: Analysis results path
            device: "cuda" or "cpu"
            use_8bit: load in 8-bit (BitsAndBytes). Default False.
            use_4bit: load in 4-bit (BitsAndBytes). Default False.
            dtype: one of {"bf16","fp16","fp32"} (case-insensitive). Default "bf16".
            batch_size: evaluation batch size.
            max_samples: cap number of texts used for analysis.
            trust_remote_code: pass-through to HF loaders.
        """
        self.model_name = model_name
        self.slug = _slugify(model_name)
        self.results_dir = results_dir + "/" + self.slug
        self.trust_remote_code = trust_remote_code

        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.use_8bit = bool(use_8bit)
        self.use_4bit = bool(use_4bit)
        self.torch_dtype = _pick_torch_dtype(dtype)
        self.batch_size = int(max(1, batch_size))
        self.max_samples = int(max(1, max_samples))

        self.model = None
        self._tokenizer = None
        self.layer_outputs: Dict[int, torch.Tensor] = {}
        self.angular_distances: np.ndarray = np.array([])
        self.optimal_layers: Dict[int, Dict[str, float]] = {}

        # Quantization config (only when requested); do not double-specify dtype
        self.bnb_config: Optional[BitsAndBytesConfig] = None
        if self.use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,    # single source of truth for compute dtype
            )
        elif self.use_8bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # ensure the result path directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    @property
    def tokenizer(self):
        # Lazy-load on first access
        if getattr(self, "_tokenizer", None) is None:
            self.load_tokenizer()
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        # Keep backward-compat if something assigns to self.tokenizer
        self._tokenizer = value

    # -----------------------
    # 1) Load tokenizer
    # -----------------------

    def load_tokenizer(self):
        """Load/configure tokenizer without loading the full model."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = getattr(self._tokenizer, "eos_token", None) or self._tokenizer.unk_token
            self._tokenizer.padding_side = "left"

    # -----------------------
    # 2) Load model
    # -----------------------
    def load_model(self) -> int:
        """Load tokenizer and model with the requested precision/quantization."""
        print(f"Loading model {self.model_name} ...")

        # Clear caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Tokenizer
        if self._tokenizer is None:
            self.load_tokenizer()
        else:
            print("Tokenizer already loaded.")

        # Model kwargs: do NOT set dtype twice. If using quantization, omit torch_dtype.
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": "auto" if self.device.startswith("cuda") else None,
        }
        if self.bnb_config is not None:
            model_kwargs["quantization_config"] = self.bnb_config
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype

        # Use AutoModelForCausalLM for broader compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model.eval()

        # Probe architecture (layers + embedding)
        layers, _ = _arch_probe(self.model)

        # Count layers for user feedback
        num_layers = len(layers)
        print(f"Model loaded with {num_layers} transformer blocks.")
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        return num_layers

    # -----------------------
    # 2) Prepare project mix
    # -----------------------

    def prepare_project_mix(
            self,
            token_targets: dict = None,  # e.g., {"syntax": 50_000, "code": 50_000, "math": 50_000}
            max_length: int = 512,
            seed: int = 42,
    ):
        """
        Build a raw-text list according to the Project Plan's mixtures.
        It samples per-domain until ~token_targets[domain] tokens are accumulated,
        then returns a single length-sorted list for efficient pad-to-longest batching.

        Domains:
          - "syntax"  -> LM-syntax/general
          - "code"    -> coding
          - "math"    -> math/reasoning
          - "logic"   -> (optional) general logic/commonsense

        Defaults follow the plan: 50k tokens per task.  (Plan §2.2)
        """
        import math
        from datasets import load_dataset

        if token_targets is None:
            token_targets = {"syntax": 50_000, "code": 50_000, "math": 50_000}

        # --- domain -> list of (dataset_id, split, extractor_fn) ---
        # You can swap any of these to your exact sources; I supply robust extractors.
        def _syntx_loader():
            # LM-syntax fallback: WikiText-2 validation
            ds = load_dataset("wikitext", name="wikitext-2-raw-v1", split="validation")

            def get_text(ex): return (ex.get("text") or "").strip()

            return ds, get_text

        def _code_loader():
            # Code: CodeParrot-clean (content) + CodeContests problems; we take whichever loads.
            try:
                ds1 = load_dataset("codeparrot/codeparrot-clean", split="train")
            except Exception:
                ds1 = None
            try:
                ds2 = load_dataset("deepmind/code_contests", split="train")
            except Exception:
                ds2 = None

            all_parts = []
            if ds1 is not None: all_parts.append(
                ("codeparrot", ds1, lambda ex: (ex.get("content") or ex.get("code") or ex.get("text") or "").strip()))
            if ds2 is not None: all_parts.append(
                ("codecontests", ds2, lambda ex: (ex.get("problem") or ex.get("problem_statement") or "").strip()))
            if not all_parts:
                # As a final fallback, reuse syntax pool
                ds_fallback, fn = _syntx_loader()
                return ds_fallback, fn

            # Simple chained iterator over multiple sources
            def chained_gen():
                for _, dsi, _ in all_parts:
                    for ex in dsi:
                        yield ex

            class _Chain:
                def __iter__(self): return chained_gen()

            # extractor tries per-source lambda; use first that returns non-empty
            def get_text(ex):
                for _, _, fn in all_parts:
                    t = fn(ex)
                    if t: return t
                return ""

            return _Chain(), get_text

        def _math_loader():
            # Math/Reasoning: GSM8K questions (stable) + fallback to any "question"-like field
            try:
                ds = load_dataset("gsm8k", "main", split="train")

                def get_text(ex):
                    return (ex.get("question") or ex.get("problem") or ex.get("input") or "").strip()

                return ds, get_text
            except Exception:
                # Fallback: Hellaswag contexts (logic-leaning)
                ds = load_dataset("hellaswag", split="validation")

                def get_text(ex):
                    return (ex.get("ctx") or "").strip()

                return ds, get_text

        def _logic_loader():
            # Logic/commonsense: HellaSwag ctx
            ds = load_dataset("hellaswag", split="validation")

            def get_text(ex): return (ex.get("ctx") or "").strip()

            return ds, get_text

        REGISTRY = {
            "syntax": _syntx_loader,
            "code": _code_loader,
            "math": _math_loader,
            "logic": _logic_loader,
        }

        # --- sample until hitting token budget per domain (approximate) ---
        rng = np.random.default_rng(seed)
        all_texts = []
        per_domain_counts = {}

        def _gather_for_domain(domain, target_tokens):
            ds, get_text = REGISTRY[domain]()
            picked = []
            tokens = 0
            # Lightweight reservoir: iterate & keep non-trivial entries
            buf = []
            for ex in ds:
                t = get_text(ex)
                if len(t) < 32:  # skip empty/ultra-short
                    continue
                buf.append(t)
                if len(buf) >= 256:
                    enc = self._tokenizer(buf, truncation=True, max_length=max_length, add_special_tokens=True)
                    for txt, ids in zip(buf, enc["input_ids"]):
                        tokens += len(ids)
                        picked.append(txt)
                        if tokens >= target_tokens:
                            return picked, tokens
                    buf.clear()
            # flush remainder
            if buf:
                enc = self._tokenizer(buf, truncation=True, max_length=max_length, add_special_tokens=True)
                for txt, ids in zip(buf, enc["input_ids"]):
                    tokens += len(ids)
                    picked.append(txt)
            return picked, tokens

        for domain, tgt in token_targets.items():
            if domain not in REGISTRY or tgt <= 0:
                continue
            texts, tok_cnt = _gather_for_domain(domain, int(tgt))
            per_domain_counts[domain] = int(tok_cnt)
            all_texts.extend(texts)

        # Sort globally by tokenized length (desc) to reduce padding waste per-batch. (Plan §1.2)
        if not all_texts:
            print("Warning: project mix produced 0 usable samples.")
            self.dataset_summary = {"domains": {}, "total_texts": 0}
            return []
        enc = self._tokenizer(all_texts, truncation=True, max_length=max_length, add_special_tokens=True)
        lengths = [len(ids) for ids in enc["input_ids"]]
        texts_sorted = [t for _, t in sorted(zip(lengths, all_texts), key=lambda x: x[0], reverse=True)]

        self.dataset_summary = {
            "domains": {k: int(v) for k, v in per_domain_counts.items()},
            "total_texts": len(texts_sorted),
            "pad_style": "pad-to-longest-per-batch",
            "token_max_length": max_length,
        }
        print(f"Prepared project mix: {self.dataset_summary}")
        return texts_sorted

    # -----------------------
    # 4) Get project mix using loader class
    # -----------------------

    def extract_layer_representations_from_loader(self, dataloader):
        """
        Same capture logic as extract_layer_representations, but consumes a DataLoader
        that yields dicts with 'input_ids' and 'attention_mask'.
        """
        print("Extracting layer representations (DataLoader path)...")
        layers, embed_layer = _arch_probe(self.model)
        if embed_layer is None:
            raise ValueError("Failed to locate embedding layer for hooks.")
        num_layers = len(layers)
        layer_representations = {i: [] for i in range(num_layers + 1)}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                activations = {}
                hooks = []

                def get_activation(name):
                    def hook(_module, _inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        last_token_idx = torch.clamp(attention_mask.sum(1)-1, min=0)
                        bsz = hs.shape[0]
                        activations[name] = hs[range(bsz), last_token_idx].detach().cpu().float()

                    return hook

                hooks.append(embed_layer.register_forward_hook(get_activation("embedding")))
                for li, layer in enumerate(layers):
                    hooks.append(layer.register_forward_hook(get_activation(f"layer_{li}")))

                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

                if "embedding" in activations:
                    layer_representations[0].append(activations["embedding"])
                for li in range(num_layers):
                    key = f"layer_{li}"
                    if key in activations:
                        layer_representations[li + 1].append(activations[key])

                for h in hooks: h.remove()
                del input_ids, attention_mask
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        for k in layer_representations:
            layer_representations[k] = torch.cat(layer_representations[k], dim=0) if layer_representations[
                k] else torch.empty(0)

        self.layer_outputs = layer_representations
        print(f"Extracted representations for {len(layer_representations)} layers (including embeddings).")
        return layer_representations

    # -----------------------
    # 5) Prepare dataset
    # -----------------------
    def prepare_dataset(self, num_samples: Optional[int] = None, max_length: int = 512):
        """
        Prepare a small text dataset for evaluation-time analysis.
        Returns a length-sorted list of raw texts (descending), so that padding per batch is minimal.
        """
        n = num_samples if num_samples is not None else self.max_samples
        print(f"Preparing up to {n} samples from Wikitext-2 (validation)...")

        ds = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="validation")
        texts = []
        for sample in ds:
            if len(texts) >= n:
                break
            txt = (sample.get("text") or "").strip()
            if len(txt) > 50:
                texts.append(txt)

        # Approximate sequence lengths without padding (for sorting)
        enc = self._tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        lengths = [len(ids) for ids in enc["input_ids"]]

        # Sort by length (descending) => batches are more uniform, less padding waste
        texts_sorted = [t for _, t in sorted(zip(lengths, texts), key=lambda x: x[0], reverse=True)]

        print(f"{len(texts_sorted)} samples prepared (sorted by length).")
        return texts_sorted

    # -----------------------
    # 6) Extract activations
    # -----------------------
    def extract_layer_representations(
            self,
            tokenized_texts,
            batch_size: Optional[int] = None,
            max_batches: Optional[int] = None,
            max_length: int = 512,
    ):
        """
        Supports two inputs:
          - List[str] raw texts  -> tokenizes per-batch with padding='longest' (recommended)
          - Legacy List[Dict[str, Tensor]] -> uses pre-padded tensors
        """
        print("Extracting layer representations...")
        layers, embed_layer = _arch_probe(self.model)
        if embed_layer is None:
            raise ValueError("Failed to locate embedding layer for hooks.")
        num_layers = len(layers)
        layer_representations: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers + 1)}

        bs = batch_size if batch_size is not None else self.batch_size
        total = len(tokenized_texts)
        if max_batches is not None:
            total = min(total, max_batches * bs)

        is_legacy = isinstance(tokenized_texts[0], dict) and "input_ids" in tokenized_texts[0]

        with torch.no_grad():
            for i in tqdm(range(0, total, bs), desc="Processing"):
                batch = tokenized_texts[i:i + bs]
                if not batch:
                    break

                if is_legacy:
                    input_ids = torch.cat([b["input_ids"] for b in batch], dim=0).to(self.device)
                    attention_mask = torch.cat([b["attention_mask"] for b in batch], dim=0).to(self.device)
                else:
                    tok = self._tokenizer(
                        batch,
                        truncation=True,
                        max_length=max_length,
                        padding=True,  # pad-to-longest in this batch
                        return_tensors="pt"
                    )
                    input_ids = tok["input_ids"].to(self.device)
                    attention_mask = tok["attention_mask"].to(self.device)

                activations: Dict[str, torch.Tensor] = {}
                hooks = []

                def get_activation(name):
                    def hook(_module, _inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        last_token_idx = torch.clamp(attention_mask.sum(1)-1, min=0)
                        bsz = hs.shape[0]
                        activations[name] = hs[range(bsz), last_token_idx].detach().cpu().float()

                    return hook

                hooks.append(embed_layer.register_forward_hook(get_activation("embedding")))
                for li, layer in enumerate(layers):
                    hooks.append(layer.register_forward_hook(get_activation(f"layer_{li}")))

                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

                if "embedding" in activations:
                    layer_representations[0].append(activations["embedding"])
                for li in range(num_layers):
                    key = f"layer_{li}"
                    if key in activations:
                        layer_representations[li + 1].append(activations[key])

                for h in hooks: h.remove()
                del input_ids, attention_mask
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        for k in layer_representations:
            layer_representations[k] = torch.cat(layer_representations[k], dim=0) if layer_representations[k] else torch.empty(0)

        self.layer_outputs = layer_representations
        print(f"Extracted representations for {len(layer_representations)} layers (including embeddings).")
        return layer_representations

    # -----------------------
    # 7) Similarity analysis
    # -----------------------
    @staticmethod
    def compute_angular_distance(x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Angular distance in [0,1] via acos of cosine similarity."""
        x1n = F.normalize(x1.float(), p=2, dim=-1)
        x2n = F.normalize(x2.float(), p=2, dim=-1)
        cos = torch.sum(x1n * x2n, dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos) / np.pi
        return ang.mean().item()

    def analyze_layer_similarities(self, max_block_size: Optional[int] = None) -> Tuple[np.ndarray, Dict[int, Dict]]:
        """Compute distances for consecutive layer blocks and pick the best start per block size."""
        if not self.layer_outputs:
            raise ValueError("Please extract representations first.")

        num_layers = len(self.layer_outputs) - 1  # exclude embeddings
        if max_block_size is None:
            max_block_size = min(num_layers // 2, 16)  # keep time/mem bounded

        print(f"Analyzing layer similarities for blocks up to size {max_block_size}...")

        distance_matrix = np.zeros((max_block_size, num_layers))
        optimal_layers: Dict[int, Dict] = {}

        for block_size in tqdm(range(1, max_block_size + 1), desc="Block sizes"):
            distances_for_block: List[float] = []

            for start_layer in range(num_layers - block_size + 1):
                end_layer_exclusive = start_layer + block_size  # exclusive for indexing
                x_start = self.layer_outputs.get(start_layer)
                x_after = self.layer_outputs.get(end_layer_exclusive)
                if x_start is not None and x_after is not None and x_start.numel() > 0 and x_after.numel() > 0:
                    d = self.compute_angular_distance(x_start, x_after)
                    distances_for_block.append(d)
                    distance_matrix[block_size - 1, start_layer] = d
                else:
                    distances_for_block.append(float("inf"))

            valid = [d for d in distances_for_block if np.isfinite(d)]
            if valid:
                best_start = distances_for_block.index(min(valid))
                # Store both exclusive and inclusive ends; report inclusive later
                optimal_layers[block_size] = dict(
                    start_layer=best_start,
                    end_layer_exclusive=best_start + block_size,
                    end_layer_inclusive=best_start + block_size - 1,  # <-- fix (6b)
                    distance=distances_for_block[best_start],
                )

        self.angular_distances = distance_matrix
        self.optimal_layers = optimal_layers
        print("Layer similarity analysis completed.")
        return distance_matrix, optimal_layers

    # -----------------------
    # 8) Visualization
    # -----------------------
    def visualize_results(self) -> str:
        """Save four-panel figure and return the image path."""
        if self.angular_distances is None or self.angular_distances.size == 0:
            raise ValueError("Please run similarity analysis first.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Heatmap: y-axis (block size) starts at 1
        sns.heatmap(
            self.angular_distances,
            cmap="viridis",
            cbar_kws={"label": "Angular Distance"},
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Angular Distance Matrix")
        axes[0, 0].set_xlabel("Starting Layer")
        axes[0, 0].set_ylabel("Block Size")
        axes[0, 0].set_yticks(np.arange(0.5, self.angular_distances.shape[0] + 0.5, 1))
        axes[0, 0].set_yticklabels([str(i) for i in range(1, self.angular_distances.shape[0] + 1)])  # <-- fix (6a)

        if hasattr(self, "optimal_layers") and self.optimal_layers:
            block_sizes = list(self.optimal_layers.keys())
            distances = [self.optimal_layers[bs]["distance"] for bs in block_sizes]

            axes[0, 1].plot(block_sizes, distances, "o-", linewidth=2, markersize=6)
            axes[0, 1].set_xlabel("Block Size")
            axes[0, 1].set_ylabel("Minimum Angular Distance")
            axes[0, 1].set_title("Optimal Distance vs Block Size")
            axes[0, 1].grid(True, alpha=0.3)

            start_layers = [self.optimal_layers[bs]["start_layer"] for bs in block_sizes]
            axes[1, 0].plot(block_sizes, start_layers, "s-", linewidth=2, markersize=6)
            axes[1, 0].set_xlabel("Block Size")
            axes[1, 0].set_ylabel("Optimal Starting Layer")
            axes[1, 0].set_title("Best Starting Layer for Pruning")
            axes[1, 0].grid(True, alpha=0.3)

            all_distances = self.angular_distances[self.angular_distances > 0].flatten()
            axes[1, 1].hist(all_distances, bins=20, alpha=0.7)
            axes[1, 1].set_xlabel("Angular Distance")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Angular Distance Distribution")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        img_path = f"{self.results_dir}/{self.slug}_analysis_results.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        try:
            from IPython.display import display, Image as IPyImage
            display(IPyImage(filename=img_path))
        except Exception:
            pass
        plt.close(fig)
        print(f"Plots saved to {img_path}")
        return img_path

    # -----------------------
    # 9) Report
    # -----------------------
    def generate_report(self) -> Dict:
        """Generate a JSON report with inclusive end indices and file names based on the model."""
        if not hasattr(self, "optimal_layers") or not self.optimal_layers:
            raise ValueError("Please run similarity analysis first.")

        total_layers = len(self.layer_outputs) - 1
        distances = [self.optimal_layers[bs]["distance"] for bs in sorted(self.optimal_layers.keys())]
        qmode = "4-bit" if self.use_4bit else ("8-bit" if self.use_8bit else "none")

        report = {
            "model_info": {
                "model_name": self.model_name,
                "slug": self.slug,
                "total_layers": total_layers,
                "precision": str(self.torch_dtype).replace("torch.", ""),
                "quantization": qmode,
                "device": self.device,
                "analysis_date": pd.Timestamp.now().isoformat(),
            },
            "key_statistics": {
                "min_distance": float(np.min(distances)),
                "max_distance": float(np.max(distances)),
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
            },
            "pruning_recommendations_mixed_dataset": {},
        }

        if hasattr(self, "dataset_summary"):
            report["dataset_summary"] = self.dataset_summary

        # Provide recommendations for common percentages (inclusive end)
        for percentage in [10, 20, 30, 40, 50]:
            layers_to_remove = int(round(total_layers * percentage / 100.0))
            if layers_to_remove in self.optimal_layers:
                entry = self.optimal_layers[layers_to_remove]
                report["pruning_recommendations_mixed_dataset"][f"{percentage}%"] = {
                    "layers_to_remove": layers_to_remove,
                    "optimal_start_layer": int(entry["start_layer"]),
                    "optimal_end_layer": int(entry["end_layer_inclusive"]),  # <-- inclusive fix (6b)
                    "expected_distance": float(entry["distance"]),
                }

        out_path = f"{self.results_dir}/{self.slug}_pruning_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Console summary
        print("\n" + "=" * 60)
        print(f"{self.model_name} Layer Pruning Analysis Report")
        print("=" * 60)
        print(f"Total layers: {total_layers}")
        print(f"Minimum angular distance: {np.min(distances):.4f}")
        print(f"Maximum angular distance: {np.max(distances):.4f}")
        print(f"Average distance: {np.mean(distances):.4f}")
        print("\nLayer pruning recommendations (inclusive end indices):")
        for pct, info in report["pruning_recommendations_mixed_dataset"].items():
            print(f"\n{pct} pruning:")
            print(f"  - Layers to remove: {info['layers_to_remove']}")
            print(f"  - Optimal start layer: {info['optimal_start_layer']}")
            print(f"  - Optimal end layer: {info['optimal_end_layer']}")
            print(f"  - Angular distance: {info['expected_distance']:.4f}")

        print(f"\nComplete report saved to {out_path}")
        return report

    # -----------------------
    # 10) One-call pipeline
    # -----------------------
    def run_full_analysis(
        self,
        dataset_samples: Optional[int] = None,
        token_max_length: int = 512,
        max_block_size: Optional[int] = None,
        override_batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
        project_token_targets: Optional[dict] = None,  # e.g., {"syntax":50_000,"code":50_000,"math":50_000}
        dataloader=None,
    ) -> Tuple[str, str, Dict]:
        """
        End-to-end process:
          1) load model
          2) prepare dataset
          3) extract layer reps (batched)
          4) analyze similarities
          5) visualize + report

        Returns: (image_path, json_path, report_dict)
        """
        print(f"Starting {self.model_name} layer pruning analysis...")

        # 1) Load
        print("\n1. Loading model...")
        if self.model is None or self._tokenizer is None:
            _ = self.load_model()
        else:
            print("Model & tokenizer already loaded — skipping.")

        # 2) Data
        print("\n2. Preparing dataset...")
        dl = None
        legacy_items = None  # <-- single fallback container for pre-tokenized items

        if dataloader is not None:
            print("Dataloader already set up...")
            dl = dataloader
            # Single source of truth: if caller gave us a DataLoader,
            # mirror its batch_size and ignore override_batch_size.
            if getattr(dl, "batch_size", None):
                self.batch_size = dl.batch_size
            if override_batch_size is not None:
                print("Note: override_batch_size is ignored because an external dataloader was provided.")

        elif project_token_targets:
            builder = MixtureDataBuilder(self._tokenizer, max_length=token_max_length, seed=42)
            texts_sorted, _lengths, summary = builder.build_text_mixture(project_token_targets)
            self.dataset_summary = summary
            dl = builder.make_dataloader(
                texts_sorted,
                batch_size=override_batch_size or self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            # Fallback to your existing path (single dataset)
            data = self.prepare_dataset(num_samples=dataset_samples, max_length=token_max_length)
            # If this returns raw texts (strings), wrap with a builder DataLoader for pad-to-longest
            if isinstance(data, list) and data and isinstance(data[0], str):
                builder = MixtureDataBuilder(self._tokenizer, max_length=token_max_length, seed=42)
                dl = builder.make_dataloader(
                    data,
                    batch_size=override_batch_size or self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )
            else:
                # Legacy path: pre-tokenized dicts from prepare_dataset
                legacy_items = data

        # 3) Reps
        print("\n3. Extracting layer representations...")
        if dl is not None:
            self.extract_layer_representations_from_loader(dl)
        elif legacy_items is not None:
            # legacy path (pre-tokenized dicts)
            self.extract_layer_representations(
                legacy_items,
                batch_size=override_batch_size,
                max_batches=max_batches,
                max_length=token_max_length,
            )
        else:
            raise RuntimeError("No dataset prepared: expected a DataLoader or pre-tokenized items.")

        # 4) Similarities
        print("\n4. Analyzing layer similarities...")
        self.analyze_layer_similarities(max_block_size=max_block_size)

        # 5) Viz + Report
        print("\n5. Visualizing results...")
        img_path = self.visualize_results()

        print("\n6. Generating report...")
        report = self.generate_report()
        json_path = f"{self.slug}_pruning_report.json"

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("Generated files:")
        print(f"- {img_path} (visualization plots)")
        print(f"- {json_path} (complete report)")
        print("=" * 60)

        return img_path, json_path, report

    # -----------------------
    # 11) Analyze each task separately
    # -----------------------
    def run_multitask_aggregation(
            self,
            task_token_targets: dict,  # e.g., {"syntax":50_000, "code":50_000, "math":50_000}
            max_block_size: int = 16,
            weights: dict = None,  # e.g., {"syntax":1.0, "code":1.0, "math":1.0}
            token_max_length: int = 512,
            batch_size: int = None,
            num_workers: int = 0,
    ):
        """
        Per-task angular matrices + normalized aggregation (weighted mean & minimax).
        Saves per-task matrices/best CSVs, aggregate best CSVs, aggregate Z-score matrices, and an 8-subplot figure.
        """
        import json
        import numpy as np
        import pandas as pd
        import torch
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        from dataset_builder import MixtureDataBuilder

        # ---------- outputs root ----------
        outdir = getattr(self, "results_dir", None)
        if not outdir:
            outdir = f"./results/{self.slug}"
            setattr(self, "results_dir", outdir)
        os.makedirs(outdir, exist_ok=True)

        # ---------- ensure model/tokenizer ----------
        if self.model is None:
            print("\nLoading model...")
            self.load_model()
        elif self._tokenizer is None:
            self.load_tokenizer()

        # ---------- normalize weights ----------
        tasks = [t for t in task_token_targets.keys() if task_token_targets[t] > 0]
        if not tasks:
            raise ValueError("task_token_targets is empty or all zeros.")
        if weights is None:
            weights = {t: 1.0 for t in tasks}
        wsum = sum(weights.get(t, 0.0) for t in tasks)
        if wsum <= 0:
            raise ValueError("All task weights are zero.")
        norm_w = {t: (weights.get(t, 0.0) / wsum) for t in tasks}

        # ---------- per-task runs ----------
        task_mats = {}  # task -> (B x N) angular matrix (NaN outside valid)
        task_best = {}  # task -> DataFrame with best per block size
        num_layers = None

        for task in tasks:
            print(f"\nTask: {task}  (target tokens ~ {task_token_targets[task]})")
            builder = MixtureDataBuilder(self._tokenizer, max_length=token_max_length, seed=42, verbose=True)
            texts, _lengths, summary = builder.build_text_mixture({task: int(task_token_targets[task])})
            dl = builder.make_dataloader(
                texts,
                batch_size=(batch_size or self.batch_size),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            # Run this task
            self.extract_layer_representations_from_loader(dl)
            mat, opt = self.analyze_layer_similarities(max_block_size=max_block_size)

            if num_layers is None:
                num_layers = mat.shape[1]

            # mask invalid cells with NaN for clarity
            mat_to_save = mat.copy().astype(float)
            for bi in range(mat_to_save.shape[0]):  # bi: 0..B-1, block size = bi+1
                valid_cols = num_layers - (bi + 1) + 1
                if valid_cols < num_layers:
                    mat_to_save[bi, valid_cols:] = np.nan

            # save full matrix
            df_mat = pd.DataFrame(
                mat_to_save,
                index=[i for i in range(1, mat_to_save.shape[0] + 1)],
                columns=[i for i in range(num_layers)]
            )
            df_mat.index.name = "block_size"
            df_mat.to_csv(os.path.join(outdir, f"{self.slug}_task_{task}_angular_matrix.csv"))

            # save per-m best summary
            rows = []
            for m in range(1, mat.shape[0] + 1):
                entry = opt.get(m)
                if entry:
                    rows.append({
                        "task": task,
                        "block_size": m,
                        "best_start_layer": int(entry["start_layer"]),
                        "best_end_layer_inclusive": int(entry["end_layer_inclusive"]),
                        "best_distance": float(entry["distance"]),
                    })
            df_best = pd.DataFrame(rows)
            df_best.to_csv(os.path.join(outdir, f"{self.slug}_task_{task}_angular_best.csv"), index=False)

            task_mats[task] = mat_to_save
            task_best[task] = df_best

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---------- z-score per task ----------
        def zscore_valid(mat):
            v = mat[~np.isnan(mat)]
            if v.size == 0:
                return np.full_like(mat, np.nan, dtype=float)
            mu, sd = float(np.nanmean(v)), float(np.nanstd(v))
            if sd == 0.0:
                return np.zeros_like(mat, dtype=float)
            return (mat - mu) / sd

        z_mats = {t: zscore_valid(task_mats[t]) for t in tasks}

        # ---------- aggregations ----------
        B = max_block_size
        N = num_layers
        stack = np.stack([z_mats[t] for t in tasks], axis=0)  # T x B x N
        wvec = np.array([norm_w[t] for t in tasks], dtype=float).reshape(-1, 1, 1)

        agg_mean = np.nansum(stack * wvec, axis=0)  # B x N
        # Avoid All-NaN slice warnings for minimax
        weighted = stack * wvec  # T x B x N
        valid = ~np.isnan(weighted)  # T x B x N
        safe = np.where(valid, weighted, -np.inf)  # fill invalid with -inf so max ignores them
        agg_minimax = safe.max(axis=0)  # B x N
        all_nan = ~valid.any(axis=0)  # B x N (true where every task was NaN)
        agg_minimax[all_nan] = np.nan  # put NaN back where nothing was valid

        # save aggregate Z-Score matrices (mask invalid positions first)
        def mask_invalid(mat):
            out = mat.copy()
            for bi in range(out.shape[0]):
                valid_cols = N - (bi + 1) + 1
                if valid_cols < N:
                    out[bi, valid_cols:] = np.nan
            return out

        agg_mean_mat = mask_invalid(agg_mean)
        agg_minimax_mat = mask_invalid(agg_minimax)

        df_agg_mean_mat = pd.DataFrame(
            agg_mean_mat,
            index=[i for i in range(1, B + 1)],
            columns=[i for i in range(N)]
        )
        df_agg_mean_mat.index.name = "block_size"
        df_agg_mean_mat.to_csv(os.path.join(outdir, f"{self.slug}_aggregate_weighted_mean_Z-Score_matrix.csv"))

        df_agg_minimax_mat = pd.DataFrame(
            agg_minimax_mat,
            index=[i for i in range(1, B + 1)],
            columns=[i for i in range(N)]
        )
        df_agg_minimax_mat.index.name = "block_size"
        df_agg_minimax_mat.to_csv(os.path.join(outdir, f"{self.slug}_aggregate_minimax_Z-Score_matrix.csv"))

        # pick best per block size for each aggregation
        def chosen_per_block(agg_mat, method_name):
            rows = []
            for bi in range(B):
                m = bi + 1
                valid_cols = N - m + 1
                if valid_cols <= 0:
                    continue
                row = agg_mat[bi, :valid_cols]
                if np.all(np.isnan(row)):
                    continue
                j = int(np.nanargmin(row))
                start = j
                end_incl = j + m - 1
                score = float(row[j])
                rows.append({
                    "method": method_name,
                    "block_size": m,
                    "weight_json": json.dumps(norm_w),
                    "chosen_start_layer": start,
                    "chosen_end_layer_inclusive": end_incl,
                    "combined_score": score,
                })
            return pd.DataFrame(rows)

        df_agg_mean = chosen_per_block(agg_mean, "weighted_mean_z")
        df_agg_minimax = chosen_per_block(agg_minimax, "minimax_z")

        df_agg_mean.to_csv(os.path.join(outdir, f"{self.slug}_aggregate_weighted_mean.csv"), index=False)
        df_agg_minimax.to_csv(os.path.join(outdir, f"{self.slug}_aggregate_minimax.csv"), index=False)

        # --------- Update pruning report JSON with aggregate recommendations ---------
        try:
            report_path = os.path.join(outdir, f"{self.slug}_pruning_report.json")
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except FileNotFoundError:
            report = {}

        # Ensure sections exist
        report.setdefault("pruning_recommendations_weighted_mean_z", {})
        report.setdefault("pruning_recommendations_minimax_z", {})

        total_layers = N  # number of transformer blocks (columns in matrices)
        for pct in [10, 20, 30, 40, 50]:
            layers_to_remove = int(round(total_layers * pct / 100.0))

            # Weighted mean
            row = df_agg_mean[df_agg_mean["block_size"] == layers_to_remove]
            if not row.empty:
                start = int(row["chosen_start_layer"].iloc[0])
                end_incl = int(row["chosen_end_layer_inclusive"].iloc[0])
                score = float(row["combined_score"].iloc[0])
                report["pruning_recommendations_weighted_mean_z"][f"{pct}%"] = {
                    "layers_to_remove": layers_to_remove,
                    "optimal_start_layer": start,
                    "optimal_end_layer": end_incl,  # inclusive
                    "combined_z_score": score
                }

            # Minimax
            row = df_agg_minimax[df_agg_minimax["block_size"] == layers_to_remove]
            if not row.empty:
                start = int(row["chosen_start_layer"].iloc[0])
                end_incl = int(row["chosen_end_layer_inclusive"].iloc[0])
                score = float(row["combined_score"].iloc[0])
                report["pruning_recommendations_minimax_z"][f"{pct}%"] = {
                    "layers_to_remove": layers_to_remove,
                    "optimal_start_layer": start,
                    "optimal_end_layer": end_incl,  # inclusive
                    "combined_z_score": score
                }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Updated pruning recommendations in {report_path}")

        # print("\nSaved:")
        # for task in tasks:
        #     print(f"  - {os.path.join(outdir, self.slug + f'_task_{task}_angular_matrix.csv')}")
        #     print(f"  - {os.path.join(outdir, self.slug + f'_task_{task}_angular_best.csv')}")
        # print(f"  - {os.path.join(outdir, self.slug + '_aggregate_weighted_mean_Z-Score_matrix.csv')}")
        # print(f"  - {os.path.join(outdir, self.slug + '_aggregate_minimax_Z-Score_matrix.csv')}")
        # print(f"  - {os.path.join(outdir, self.slug + '_aggregate_weighted_mean.csv')}")
        # print(f"  - {os.path.join(outdir, self.slug + '_aggregate_minimax.csv')}")

        # ---------- visualization (8 subplots in 2 rows x 4 cols) ----------
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes = np.array(axes).reshape(2, 4)

        # first 3: per-task angular matrices (up to 3 tasks shown)
        show_tasks = tasks[:3]
        for k, task in enumerate(show_tasks):
            ax = axes[0, k]
            sns.heatmap(task_mats[task], cmap="viridis", ax=ax, cbar=k == 2,
                        cbar_kws={"label": "Angular Distance"} if k == 2 else None)
            ax.set_title(f"{task} – Angular Matrix")
            ax.set_xlabel("Start Layer")
            ax.set_ylabel("Block Size")
            ax.set_yticks(np.arange(0.5, B + 0.5, 1))
            ax.set_yticklabels([str(i) for i in range(1, B + 1)])

        # if fewer than 3 tasks, hide empty slots
        for k in range(len(show_tasks), 3):
            axes[0, k].axis("off")

        # next two: aggregate Z-score matrices
        ax = axes[0, 3]
        sns.heatmap(agg_mean_mat, cmap="coolwarm", ax=ax, cbar=True, cbar_kws={"label": "Z-Score"})
        ax.set_title("Aggregate – Weighted Mean (Z)")
        ax.set_xlabel("Start Layer")
        ax.set_ylabel("Block Size")
        ax.set_yticks(np.arange(0.5, B + 0.5, 1))
        ax.set_yticklabels([str(i) for i in range(1, B + 1)])

        ax = axes[1, 0]
        sns.heatmap(agg_minimax_mat, cmap="coolwarm", ax=ax, cbar=True, cbar_kws={"label": "Z-Score"})
        ax.set_title("Aggregate – Minimax (Z)")
        ax.set_xlabel("Start Layer")
        ax.set_ylabel("Block Size")
        ax.set_yticks(np.arange(0.5, B + 0.5, 1))
        ax.set_yticklabels([str(i) for i in range(1, B + 1)])

        # next: Optimal Distance VS Block size for tasks (line plot)
        ax = axes[1, 1]
        for task in show_tasks:
            dfb = task_best[task]
            if not dfb.empty:
                ax.plot(dfb["block_size"], dfb["best_distance"], marker="o", label=task)
        ax.set_title("Optimal Distance VS Block size (tasks)")
        ax.set_xlabel("Block Size")
        ax.set_ylabel("Minimum Angular Distance")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # next: Optimal Z-Score VS Block size for aggregates
        ax = axes[1, 2]
        if not df_agg_mean.empty:
            ax.plot(df_agg_mean["block_size"], df_agg_mean["combined_score"], marker="s", label="weighted_mean_z")
        if not df_agg_minimax.empty:
            ax.plot(df_agg_minimax["block_size"], df_agg_minimax["combined_score"], marker="^",
                    label="minimax_z")
        ax.set_title("Optimal Z-Score VS Block size (aggregates)")
        ax.set_xlabel("Block Size")
        ax.set_ylabel("Combined Z-Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # last: Best Starting Layer (tasks + aggregates) – 5 lines possible
        ax = axes[1, 3]
        for task in show_tasks:
            dfb = task_best[task]
            if not dfb.empty:
                ax.plot(dfb["block_size"], dfb["best_start_layer"], marker="o", label=f"{task}")
        if not df_agg_mean.empty:
            ax.plot(df_agg_mean["block_size"], df_agg_mean["chosen_start_layer"], marker="s", label="weighted_mean_z")
        if not df_agg_minimax.empty:
            ax.plot(df_agg_minimax["block_size"], df_agg_minimax["chosen_start_layer"], marker="^",
                    label="minimax_z")
        ax.set_title("Best Starting Layer")
        ax.set_xlabel("Block Size")
        ax.set_ylabel("Start Layer")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        fig_path = os.path.join(outdir, f"{self.slug}_multitask_aggregation.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        try:
            # show inline in notebooks
            from IPython.display import display, Image as IPyImage
            display(IPyImage(filename=fig_path))
        except Exception:
            pass

        return {
            "per_task_best": task_best,
            "aggregate": {
                "weighted_mean_matrix_csv": os.path.join(outdir,
                                                         f"{self.slug}_aggregate_weighted_mean_Z-Score_matrix.csv"),
                "minimax_matrix_csv": os.path.join(outdir, f"{self.slug}_aggregate_minimax_Z-Score_matrix.csv"),
                "weighted_mean_csv": os.path.join(outdir, f"{self.slug}_aggregate_weighted_mean.csv"),
                "minimax_csv": os.path.join(outdir, f"{self.slug}_aggregate_minimax.csv"),
                "weights": norm_w,
                "figure": fig_path,
                "results_dir": outdir,
            }
        }

