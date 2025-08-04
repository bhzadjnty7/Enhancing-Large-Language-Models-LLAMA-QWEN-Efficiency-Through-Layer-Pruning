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
        results_dir: str = "results",
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
        self.results_dir = results_dir
        self.slug = _slugify(model_name)
        self.trust_remote_code = trust_remote_code

        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.use_8bit = bool(use_8bit)
        self.use_4bit = bool(use_4bit)
        self.torch_dtype = _pick_torch_dtype(dtype)
        self.batch_size = int(max(1, batch_size))
        self.max_samples = int(max(1, max_samples))

        self.model = None
        self.tokenizer = None
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

    # -----------------------
    # 1) Load model/tokenizer
    # -----------------------
    def load_model(self) -> int:
        """Load tokenizer and model with the requested precision/quantization."""
        print(f"Loading model {self.model_name} ...")

        # Clear caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        # Robust padding setup for causal LMs
        if self.tokenizer.pad_token is None:
            # Many causal LMs lack a pad token; use EOS safely
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or self.tokenizer.unk_token
        # Left padding is usually safer for decoder-only models
        self.tokenizer.padding_side = "left"

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
    # 2) Prepare dataset
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
        enc = self.tokenizer(
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
    # 3) Extract activations
    # -----------------------
    def extract_layer_representations(
            self,
            tokenized_texts,
            batch_size: Optional[int] = None,
            max_batches: Optional[int] = None,
            max_length: int = 512,
    ):
        """
        Register forward hooks and capture per-layer representations at the last non-pad token.
        Supports:
          - NEW PATH: tokenized_texts = List[str] (raw texts)  -> we tokenize per-batch with padding='longest'
          - LEGACY PATH: tokenized_texts = List[Dict[str, Tensor]] -> original behavior (already padded)
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
                try:
                    if is_legacy:
                        # ---- Legacy path (pre-padded items) ----
                        batch = tokenized_texts[i:i + bs]
                        if not batch:
                            break
                        input_ids = torch.cat([b["input_ids"] for b in batch], dim=0).to(self.device)
                        attention_mask = torch.cat([b["attention_mask"] for b in batch], dim=0).to(self.device)
                    else:
                        # ---- New path: per-batch tokenization with padding to longest ----
                        batch_texts = tokenized_texts[i:i + bs]
                        if not batch_texts:
                            break
                        tok = self.tokenizer(
                            batch_texts,
                            truncation=True,
                            max_length=max_length,
                            padding=True,  # <-- pad to the longest in this batch
                            return_tensors="pt"
                        )
                        input_ids = tok["input_ids"].to(self.device)
                        attention_mask = tok["attention_mask"].to(self.device)

                    activations: Dict[str, torch.Tensor] = {}
                    hooks = []

                    def get_activation(name):
                        def hook(_module, _inp, out):
                            hs = out[0] if isinstance(out, tuple) else out
                            last_token_idx = attention_mask.sum(dim=1) - 1
                            bsz = hs.shape[0]
                            activations[name] = hs[range(bsz), last_token_idx].detach().cpu().float()

                        return hook

                    # Register hooks
                    hooks.append(embed_layer.register_forward_hook(get_activation("embedding")))
                    for li, layer in enumerate(layers):
                        hooks.append(layer.register_forward_hook(get_activation(f"layer_{li}")))

                    # Forward pass
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

                    # Collect
                    if "embedding" in activations:
                        layer_representations[0].append(activations["embedding"])
                    for li in range(num_layers):
                        key = f"layer_{li}"
                        if key in activations:
                            layer_representations[li + 1].append(activations[key])

                    # Cleanup
                    for h in hooks:
                        h.remove()
                    del input_ids, attention_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    continue

        # Concatenate lists to tensors
        for k in layer_representations:
            layer_representations[k] = torch.cat(layer_representations[k], dim=0) if layer_representations[k] else torch.empty(0)

        self.layer_outputs = layer_representations
        print(f"Extracted representations for {len(layer_representations)} layers (including embeddings).")
        return layer_representations

    # -----------------------
    # 4) Similarity analysis
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
    # 5) Visualization
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
        img_path = f"results/{self.slug}_analysis_results.png"
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
    # 6) Report
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
            "pruning_recommendations": {},
            "insights": [],
        }

        # Provide recommendations for common percentages (inclusive end)
        for percentage in [10, 20, 30, 40, 50]:
            layers_to_remove = int(round(total_layers * percentage / 100.0))
            if layers_to_remove in self.optimal_layers:
                entry = self.optimal_layers[layers_to_remove]
                report["pruning_recommendations"][f"{percentage}%"] = {
                    "layers_to_remove": layers_to_remove,
                    "optimal_start_layer": int(entry["start_layer"]),
                    "optimal_end_layer": int(entry["end_layer_inclusive"]),  # <-- inclusive fix (6b)
                    "expected_distance": float(entry["distance"]),
                }

        # Light insights
        if self.optimal_layers:
            best_block = min(self.optimal_layers.items(), key=lambda kv: kv[1]["distance"])
            report["insights"] = [
                f"Best block: size {best_block[0]} starting at layer {best_block[1]['start_layer']} "
                f"(end {best_block[1]['end_layer_inclusive']}, distance {best_block[1]['distance']:.4f}).",
                f"Min/Max/Mean distance: {np.min(distances):.4f} / {np.max(distances):.4f} / {np.mean(distances):.4f}.",
            ]

        out_path = f"results/{self.slug}_pruning_report.json"
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
        for pct, info in report["pruning_recommendations"].items():
            print(f"\n{pct} pruning:")
            print(f"  - Layers to remove: {info['layers_to_remove']}")
            print(f"  - Optimal start layer: {info['optimal_start_layer']}")
            print(f"  - Optimal end layer: {info['optimal_end_layer']}")
            print(f"  - Angular distance: {info['expected_distance']:.4f}")

        print(f"\nComplete report saved to {out_path}")
        return report

    # -----------------------
    # 7) One-call pipeline
    # -----------------------
    def run_full_analysis(
        self,
        dataset_samples: Optional[int] = None,
        token_max_length: int = 512,
        max_block_size: Optional[int] = None,
        override_batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
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
        _ = self.load_model()

        # 2) Data
        print("\n2. Preparing dataset...")
        tokenized_texts = self.prepare_dataset(num_samples=dataset_samples, max_length=token_max_length)

        # 3) Reps
        print("\n3. Extracting layer representations...")
        self.extract_layer_representations(
            tokenized_texts,
            batch_size=override_batch_size,
            max_batches=max_batches,
        )

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
