# llm_layer_pruner.py
import os, json, math, gc, inspect
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import contextlib

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import TrainerCallback, TrainerState, TrainerControl

from LayerPruningAnalyzer import _arch_probe
from dataset_builder import TextListMapDataset, collate_with_labels  # <-- uses your builder's single source of truth

# --------- Minimal architecture probe (LLama/Qwen/NeoX-style) ----------
# def _arch_probe(model):
#     if hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "embed_tokens"):
#         return model.model.layers, model.model.embed_tokens
#     if hasattr(model, "layers") and hasattr(model, "embed_tokens"):
#         return model.layers, model.embed_tokens
#     if hasattr(model, "transformer") and hasattr(model.transformer, "h") and hasattr(model.transformer, "wte"):
#         return model.transformer.h, model.transformer.wte
#     if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers") and hasattr(model.gpt_neox, "embed_in"):
#         return model.gpt_neox.layers, model.gpt_neox.embed_in
#     raise ValueError("Unsupported architecture: could not locate decoder layers and embeddings.")


# --------- Dataloader wrapper for Trainer ----------
class _LoaderDataset(torch.utils.data.IterableDataset):
    """
    Wrap an existing DataLoader's underlying iterable to feed Trainer
    without rebuilding your dataset pipeline.
    """
    def __init__(self, dataloader):
        super().__init__()
        self._dl = dataloader
    def __iter__(self):
        for batch in self._dl:
            yield batch


class _UnbatchedIterable(torch.utils.data.IterableDataset):
    """
    Wrap an existing DataLoader that yields dict batches and expose *examples*.
    Each __iter__ yields dicts with tensors shaped [S] (no batch dim).
    """

    def __init__(self, dataloader):
        super().__init__()
        self.dl = dataloader

    def __iter__(self):
        for batch in self.dl:
            # assume batch is a dict of tensors [B, S]
            bsz = batch["input_ids"].size(0)
            for i in range(bsz):
                yield {k: v[i] for k, v in batch.items()}

    def __len__(self):
        try:
            n_batches = len(self.dl)
            bsz = getattr(self.dl, "batch_size", None)
            if bsz is not None:
                return n_batches * int(bsz)
        except TypeError:
            pass
        # fallback if unknown — gives finite epoch and sensible progress
        return 0


def make_leftpad_collator(pad_id: int):
    def _collate(features, return_tensors="pt"):
        import torch
        # features: list of per-example dicts with 1D tensors [S]
        max_len = max(x["input_ids"].shape[0] for x in features)

        def lp(x, L):
            if x.shape[0] == L: return x
            pad = x.new_full((L - x.shape[0],), pad_id)
            return torch.cat([pad, x], dim=0)

        input_ids = torch.stack([lp(f["input_ids"], max_len) for f in features], dim=0)
        attn      = torch.stack([lp(f["attention_mask"], max_len) for f in features], dim=0)
        labels    = input_ids.clone()
        labels[attn == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return _collate

class _CountingCollator:
    def __init__(self, base_collator, counter_dict: dict):
        self.base = base_collator
        self.counter = counter_dict  # expects {"micro_batches": 0, "tokens": 0}
    def __call__(self, features):
        self.counter["micro_batches"] += 1
        # optional: rough token count = sum of lengths pre-padding
        try:
            self.counter["tokens"] += sum(len(f["input_ids"]) for f in features if "input_ids" in f)
        except Exception:
            pass
        return self.base(features)

def _extract_texts_from_loader(dl):
    """
    Expect a MixtureDataBuilder-made DataLoader (dataset holds raw texts).
    Falls back to raising a clear error if we can't access texts without unbatching.
    """
    # Common case: your builder uses TextListDataset with .texts
    ds = getattr(dl, "dataset", None)
    texts = getattr(ds, "texts", None)
    if isinstance(texts, list) and texts and isinstance(texts[0], str):
        return texts
    raise TypeError(
        "heal_*_trainer expects loaders built by MixtureDataBuilder (dataset.texts must exist); "
        "please pass such a loader or switch to builder.as_trainer_dataset()."
    )

# --- Count batches processed in Trainer just to make sure ---
class StepAccountingCallback(TrainerCallback):
    """
    Counts micro-batches and optimizer steps robustly across HF versions.

    - micro_batches: incremented at the end of EACH gradient-accumulation substep
    - optimizer_steps: incremented at the end of EACH optimizer update
    """
    def __init__(self, grad_accum: int):
        self.grad_accum = max(1, int(grad_accum))
        self.micro_batches = 0
        self.optimizer_steps = 0

    # ---- v4.41+ emits this per micro-batch (preferred) ----
    def on_substep_end(self, args, state, control, **kwargs):
        # print("Going through substep_end")
        self.micro_batches += 1

    # ---- fallback for older versions: try train_batch_end ----
    # Some older Trainer versions surface only train-batch events.
    def on_train_batch_end(self, args, state, control, **kwargs):
        # Only use this path if on_substep_end hasn't run in this process
        if not hasattr(self, "_seen_substep") or not self._seen_substep:
            # print("Going through batch_end")
            self.micro_batches += 1

    def on_substep_begin(self, *a, **k):
        # mark that substep callbacks exist in this runtime
        self._seen_substep = True

    # ---- one per optimizer update ----
    def on_step_end(self, args, state, control, **kwargs):
        # print("Finishing a Full Step")
        self.optimizer_steps += 1
        self.micro_batches += 1

    # Print a single, trustworthy line at train end
    def on_train_end(self, args, state, control, **kwargs):
        denom = max(1, self.optimizer_steps)
        ratio = self.micro_batches / denom
        print(f"[Accounting] optimizer_steps={self.optimizer_steps}  "
              f"micro_batches={self.micro_batches}  grad_accum={self.grad_accum}  "
              f"(~{ratio:.2f} batches/opt-step)")


# --------- The pruner class ----------------------------------------------------------
class LLMLayerPruner:
    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[nn.Module] = None,
        max_context_len: int = 2048,
        max_new_tokens: int = 256,
        tokenizer: Optional[AutoTokenizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)),
        quant_config: Optional[BitsAndBytesConfig] = None,
        quant_config_ppl: Optional[BitsAndBytesConfig] = None,
        quant_config_single_heal: Optional[BitsAndBytesConfig] = None,
        quant_config_qlora: Optional[BitsAndBytesConfig] = None,
        results_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        # unified defaults
        default_percentages: Optional[List[int]] = None,
        default_rank: int = 16,
        default_max_heal_steps: int = 500,
        default_max_qlora_steps: int = 1000,
        blank_noise_std: float = 0.0
    ):
        """
        Either provide (model_name) or an already-loaded (model, tokenizer).
        Results default to: results/pruning_sweep/single-shot/<model_slug>
        """
        self.model_name = model_name
        self.model = model
        self._tokenizer = tokenizer
        self.device = device
        self.torch_dtype = dtype
        self.quant_config = quant_config
        self.trust_remote_code = trust_remote_code

        slug = self._slugify(model_name) if model_name else "model"
        self.results_dir = results_dir or os.path.join("results", "pruning_sweep", "single-shot", slug)
        os.makedirs(self.results_dir, exist_ok=True)

        # Phase-specific quantization configs (override self.quant_config if set)
        self.quant_cfg_for = {
            "ppl": quant_config_ppl,  # e.g., None => full bf16/fp16
            "single_heal": quant_config_single_heal,  # e.g., None => full bf16/fp16
            "qlora": quant_config_qlora,  # e.g., BitsAndBytesConfig(load_in_4bit=True, ...)
        }

        # Optional caps
        self.max_eval_seq_len = max_context_len
        self.gen_max_new_tokens = max_new_tokens

        # unified defaults
        self.default_percentages = default_percentages or [10, 20, 30, 40, 50]
        self.default_rank = default_rank
        self.default_max_heal_steps = default_max_heal_steps
        self.default_max_qlora_steps = default_max_qlora_steps
        self.blank_noise_std = blank_noise_std

    # --- lazy tokenizer property ---
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if not self.model_name:
                raise ValueError("Tokenizer requested but model_name is None and no tokenizer provided.")
            tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            if tok.pad_token is None:
                tok.pad_token = getattr(tok, "eos_token", None) or tok.unk_token
            tok.padding_side = "left"
            self._tokenizer = tok
        return self._tokenizer

    # --- utils ---
    def _slugify(self, name: Optional[str]) -> str:
        if not name:
            return "model"
        import re
        slug = name.strip().lower().replace("/", "-").replace(" ", "-")
        slug = re.sub(r"[^a-z0-9\-_.]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug

    def _model_uses_bnb(self, model) -> bool:
        # Robust check for 4/8bit
        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            return True
        for m in model.modules():
            n = m.__class__.__name__.lower()
            if "linear8bit" in n or "linear4bit" in n:
                return True
        return False

    def _pick_eval_loader(self, train_dataloader, eval_dataloader):
        """
        Prefer eval_dataloader if it's provided and non-empty, otherwise fall back to train_dataloader.
        """
        if eval_dataloader is not None:
            try:
                if len(eval_dataloader) > 0:
                    return eval_dataloader
            except TypeError:
                # If __len__ isn't defined, assume it's usable
                return eval_dataloader
        print("Warning: no valid eval dataloader provided, Falling back to Train Dataloader!")
        return train_dataloader

    def _infer_compute_dtype(self, model) -> torch.dtype:
        """
        Choose the dtype we should compute in for *trainable* pieces and activations:
        - 4-bit: use bnb_4bit_compute_dtype if provided, else bf16/fp16
        - 8-bit: force bf16 if supported, else fp16 (never fp32)
        - otherwise: prefer any floating param dtype (bf16/fp16), else sane default
        """
        # Prefer an explicit 4-bit compute dtype from the quant config
        qc = getattr(self, "quant_config", None)
        if qc is not None and getattr(qc, "load_in_4bit", False):
            cd = getattr(qc, "bnb_4bit_compute_dtype", None)
            if cd is not None:
                return cd
            return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        # If model is in 8-bit, bnb expects fp16/bf16 inputs (avoid fp32 here)
        if getattr(model, "is_loaded_in_8bit", False) or (qc is not None and getattr(qc, "load_in_8bit", False)):
            return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        # Otherwise pick any floating param dtype present (bf16/fp16/fp32)
        for p in model.parameters():
            if p.is_floating_point():
                return p.dtype

        # Final fallback
        return (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else (torch.float16 if torch.cuda.is_available() else torch.float32))

    def _effective_compute_dtype(self, model) -> torch.dtype:
        # Prefer any floating param’s dtype (skips int8/uint8 quant buffers)
        for p in model.parameters():
            if p.is_floating_point():
                return p.dtype
        # Fallback to BF16 if nothing found
        return torch.bfloat16

    # --- model loading / cloning ---
    def load_model(self):
        """
        Load (and cache) the base/PEFT-composed model. Always returns a model object.
        """
        if self.model is not None:
            return self.model

        if not self.model_name:
            raise ValueError("Provide model_name or a loaded model.")

        # Try composed base+adapters layout first
        compose_json = os.path.join(self.model_name, "PEFT_COMPOSE.json")
        if os.path.isdir(self.model_name) and os.path.exists(compose_json):
            try:
                import json
                from peft import PeftModel
                with open(compose_json, "r") as f:
                    comp = json.load(f)

                base_dir = os.path.join(self.model_name, comp.get("base", "base"))
                adapters_dir = os.path.join(self.model_name, comp.get("adapters", "adapters"))

                # Tokenizer aligned with base
                if self._tokenizer is None:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        base_dir, trust_remote_code=self.trust_remote_code
                    )
                    if self._tokenizer.pad_token is None:
                        self._tokenizer.pad_token = getattr(self._tokenizer, "eos_token",
                                                            None) or self._tokenizer.unk_token
                    self._tokenizer.padding_side = "left"

                model_kwargs = {
                    "trust_remote_code": self.trust_remote_code,
                    # Prefer no device_map so we hard-crash on OOM per your policy:
                    # "device_map": "auto" if self.device.startswith("cuda") else None,
                }
                if getattr(self, "bnb_config", None) is not None:
                    model_kwargs["quantization_config"] = self.bnb_config
                elif getattr(self, "quant_config", None) is not None:
                    model_kwargs["quantization_config"] = self.quant_config
                else:
                    model_kwargs["torch_dtype"] = self.torch_dtype

                base = AutoModelForCausalLM.from_pretrained(base_dir, **model_kwargs).to(self.device)
                self.model = PeftModel.from_pretrained(base, adapters_dir)
                self.model.eval()
                return self.model
            except Exception as e:
                print(f"[Pruner] PEFT compose load failed ({e}); falling back to plain load.")

        # Plain load path
        kwargs = {"trust_remote_code": self.trust_remote_code}
        if getattr(self, "bnb_config", None) is not None:
            kwargs["quantization_config"] = self.bnb_config
        elif getattr(self, "quant_config", None) is not None:
            kwargs["quantization_config"] = self.quant_config
        else:
            kwargs["torch_dtype"] = self.torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs).to(self.device)
        self.model.eval()
        return self.model

    def get_layer_count(self) -> int:
        """Return the current model's decoder-layer count."""
        m = self.load_model()
        layers, _ = _arch_probe(m)
        return len(layers)

    def configure_profiles(self, *, ppl=None, single_heal=None, qlora=None,
                           max_eval_seq_len=None, gen_max_new_tokens=None):
        if ppl is not None: self.quant_cfg_for["ppl"] = ppl
        if single_heal is not None: self.quant_cfg_for["single_heal"] = single_heal
        if qlora is not None: self.quant_cfg_for["qlora"] = qlora
        if max_eval_seq_len is not None: self.max_eval_seq_len = max_eval_seq_len
        if gen_max_new_tokens is not None: self.gen_max_new_tokens = gen_max_new_tokens

    def clone_fresh_model(self):
        """Reload a fresh copy. Safer than deep-copy esp. for quantized models."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        return self.load_model()

    def _fresh_model_for_pruning(
            self,
            *,
            quantized: bool = False,
            on_cpu: bool = False,
            dtype: Optional[torch.dtype] = None,
    ):
        """
        Fresh model loader dedicated to the pruning_sweep/save step.
        - quantized=False => standard torch_dtype; CPU path forces float32.
        - quantized=True  => use self.quant_config (bnb) if provided.
        """
        kwargs = {"trust_remote_code": self.trust_remote_code}
        # if on_cpu:
        #     kwargs["device_map"] = None  # force CPU load
        # else:
        #     kwargs["device_map"] = "auto" if self.device.startswith("cuda") else None

        if quantized and self.quant_config is not None:
            kwargs["quantization_config"] = self.quant_config
        else:
            if dtype is None:
                if on_cpu:
                    dtype = torch.float32
                else:
                    dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                             else (torch.float16 if torch.cuda.is_available() else torch.float32))
            kwargs["torch_dtype"] = dtype

        m = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs).to(self.device)
        m.eval()
        return m

    def _fresh_model_for_phase(self, phase: str, *, on_cpu: bool = False, dtype: Optional[torch.dtype] = None):
        qcfg = self.quant_cfg_for.get(phase, self.quant_config)
        kwargs = {"trust_remote_code": self.trust_remote_code}
        # kwargs["device_map"] = None if on_cpu else ("auto" if self.device.startswith("cuda") else None)

        if qcfg is not None:
            kwargs["quantization_config"] = qcfg
        else:
            if dtype is None:
                if on_cpu:
                    dtype = torch.float32
                else:
                    dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                             else (torch.float16 if torch.cuda.is_available() else torch.float32))
            kwargs["torch_dtype"] = dtype

        try:
            m = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs).to(self.device)
            m.eval()
            return m
        except Exception as e:
            raise RuntimeError(f"Failed to (re)load model for phase='{phase}': {e}") from e

    # --- basic evals ---
    def sanity_check(self, dataloader, max_batches: int = 1, model: Optional[nn.Module] = None) -> Dict:
        model = model if model is not None else self.load_model()
        layers, _ = _arch_probe(model)
        H = model.config.hidden_size
        L = len(layers)

        model.eval()
        n_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits  # [B,S,V]
                assert logits.ndim == 3 and logits.size(0) == input_ids.size(0)
                n_tokens += int(attention_mask.sum().item())
                if i + 1 >= max_batches:
                    break
        return {"num_layers": L, "hidden_size": H, "tokens_seen": n_tokens}

    def compute_perplexity(self, dataloader, max_batches: Optional[int] = None,
                           model: Optional[nn.Module] = None) -> float:
        """
        Cross-entropy perplexity on padded batches.
        - Masks out padding positions (labels = -100 where attention_mask==0).
        - Warns & returns inf if the dataloader is empty.
        """
        # Empty-dataloader guard
        try:
            if len(dataloader) == 0:
                print("[WARN] compute_perplexity: EMPTY loader; returning inf.")
                return float("inf")
        except TypeError:
            pass

        model = model if model is not None else self.load_model()
        model.eval()

        # Temporarily force the leanest forward
        prev_cache = getattr(model.config, "use_cache", None)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        prev_attn = getattr(model.config, "output_attentions", None)
        if hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False
        prev_hid = getattr(model.config, "output_hidden_states", None)
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False

        want_dtype = self._infer_compute_dtype(model)
        amp_dtype = want_dtype if want_dtype in (torch.float16, torch.bfloat16) else None

        losses, steps = [], 0
        autocast_cm = torch.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext

        # Inference-mode is stricter than no_grad and can reduce peak usage
        with torch.inference_mode():
            for i, batch in enumerate(tqdm(dataloader, desc="Perplexity")):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

                # Optional hard cap to keep memory predictable
                if self.max_eval_seq_len is not None and input_ids.size(1) > self.max_eval_seq_len:
                    input_ids = input_ids[:, -self.max_eval_seq_len:]
                    attention_mask = attention_mask[:, -self.max_eval_seq_len:]

                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                with autocast_cm("cuda", dtype=amp_dtype):
                    # Ask the model to avoid extras and return a tuple to reduce Python object overhead
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,  # => (loss, logits) usually
                    )
                    loss_val = out[0] if isinstance(out, (tuple, list)) else out.loss
                    loss_val = loss_val.detach().float()

                if torch.isfinite(loss_val):
                    losses.append(loss_val.item())

                steps += 1
                if max_batches is not None and steps >= max_batches:
                    break

                # # help the allocator between long sequences
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                #     torch.cuda.ipc_collect()

        # restore config
        if prev_cache is not None: model.config.use_cache = prev_cache
        if prev_attn is not None:  model.config.output_attentions = prev_attn
        if prev_hid is not None:   model.config.output_hidden_states = prev_hid

        if steps == 0 or len(losses) == 0:
            print("[WARN] compute_perplexity: no valid losses; returning inf.")
            return float("inf")

        mean_loss = float(np.mean(losses))
        return float(math.exp(mean_loss)) if np.isfinite(mean_loss) else float("inf")

    # --- pruning_sweep mechanics ---
    def _get_layer_container_and_list(self, model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model, model.model.layers
        if hasattr(model, "layers"):
            return model, model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer, model.transformer.h
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox, model.gpt_neox.layers
        raise ValueError("Unsupported architecture for pruning_sweep.")

    def prune_without_replacement(self, model, start: int, end_inclusive: int):
        parent, layers = self._get_layer_container_and_list(model)
        keep_idx = list(range(0, start)) + list(range(end_inclusive + 1, len(layers)))
        parent.layers = nn.ModuleList([layers[i] for i in keep_idx])
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(parent.layers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return model

    def _insert_replacement(self, model, pos: int, block: nn.Module):
        parent, layers = self._get_layer_container_and_list(model)
        new_layers = []
        for i in range(len(layers) + 1):
            if i < pos:
                new_layers.append(layers[i])
            elif i == pos:
                new_layers.append(block)
            else:
                new_layers.append(layers[i - 1])
        parent.layers = nn.ModuleList(new_layers)
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(parent.layers)
        return model

    # --- replacement layer builders (training-free) ---
    def _gather_removed_layers_state_dicts_fp_shadow(
            self, start: int, end_inclusive: int, dtype: torch.dtype = torch.float32
    ):
        """Load a CPU, full-precision copy of the model just to fetch FP state_dicts."""
        kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": dtype,  # CPU float32 for numerically stable averaging
            # device_map=None -> load on CPU
        }
        shadow = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        layers, _ = _arch_probe(shadow)
        sds = [layers[i].state_dict() for i in range(start, end_inclusive + 1)]
        del shadow
        gc.collect()
        return sds

    def _new_decoder_layer(self, model, layer_idx: Optional[int] = None):
        """
        Instantiate a new decoder layer that seamlessly bridges dtypes:
        - casts incoming hidden_states to the model's compute dtype (comp_dtype)
        - runs the native layer forward
        - casts the primary output back to comp_dtype
        This avoids any matmul Float vs BF16/FP16 mismatches without hooks/wrappers.
        """
        parent, layers = self._get_layer_container_and_list(model)
        base_cls = layers[0].__class__

        # Build kwargs for the native constructor
        sig = inspect.signature(base_cls.__init__)
        kwargs = {}
        if "config" in sig.parameters:
            kwargs["config"] = model.config
        else:
            raise TypeError(f"{base_cls.__name__} constructor does not accept 'config'")
        if "layer_idx" in sig.parameters:
            kwargs["layer_idx"] = 0 if layer_idx is None else int(layer_idx)

        comp_dtype = self._infer_compute_dtype(model)
        target_device = next(model.parameters()).device

        # Dynamic subclass that only overrides forward to bridge dtype
        class _DTypeBridgedLayer(base_cls):  # type: ignore[misc]
            def __init__(self, *args, _comp_dtype=None, **kw):
                self._comp_dtype = _comp_dtype
                super().__init__(*args, **kw)

            def forward(self, *args, **kw):
                # Extract hidden_states from args/kwargs in a model-agnostic way
                if len(args) > 0:
                    x = args[0]
                    rest = args[1:]
                    used_args = True
                else:
                    x = kw.get("hidden_states", None)
                    used_args = False
                    rest = ()

                # Cast input to comp_dtype if needed (fixes mat1 vs mat2 dtype)
                if torch.is_tensor(x) and x.is_floating_point() and x.dtype != self._comp_dtype:
                    x = x.to(self._comp_dtype)

                # Call the native forward
                if used_args:
                    out = super().forward(x, *rest, **kw)
                else:
                    kw["hidden_states"] = x
                    out = super().forward(**kw)

                # Ensure the primary output tensor matches comp_dtype
                if isinstance(out, tuple):
                    h = out[0]
                    if torch.is_tensor(h) and h.is_floating_point() and h.dtype != self._comp_dtype:
                        h = h.to(self._comp_dtype)
                    out = (h,) + out[1:]
                elif torch.is_tensor(out) and out.is_floating_point() and out.dtype != self._comp_dtype:
                    out = out.to(self._comp_dtype)

                return out

        # Instantiate bridged layer with the same constructor args as the native class
        layer = _DTypeBridgedLayer(*(), _comp_dtype=comp_dtype, **kwargs)
        # Place on the right device/dtype from the start
        layer.to(device=target_device, dtype=comp_dtype)
        return layer

    @torch.no_grad()
    def _init_decoder_layer_as_identity(self, layer: nn.Module, noise_std: float = 0.0):
        # LayerNorms to identity
        for m in layer.modules():
            if isinstance(m, nn.LayerNorm):
                if getattr(m, "weight", None) is not None:
                    m.weight.fill_(1.0)
                if getattr(m, "bias", None) is not None:
                    m.bias.zero_()
        # Linear weights/biases: zero (identity via residual), then optional small noise
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if noise_std and noise_std > 0:
                    # fan-in scaled Gaussian: N(0, (noise_std/sqrt(fan_in))^2)
                    fan_in = m.weight.size(1)
                    scale = float(noise_std) / math.sqrt(fan_in if fan_in > 0 else 1)
                    m.weight.add_(torch.randn_like(m.weight) * scale)
        # Disable any dropout present
        for name in ["dropout", "attn_dropout"]:
            if hasattr(layer, name):
                getattr(layer, name).p = 0.0

    def _gather_removed_layers_state_dicts(self, model, start: int, end_inclusive: int):
        parent, layers = self._get_layer_container_and_list(model)
        return [layers[i].state_dict() for i in range(start, end_inclusive + 1)]

    def _load_layer_state_dict_safe(self, layer: nn.Module, sd: dict):
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        return layer

    def _avg_state_dicts(self, sds: List[Dict[str, torch.Tensor]], target_dtype: Optional[torch.dtype] = None):
        avg = {}
        K = len(sds)
        mid = sds[K // 2]
        for k in mid.keys():
            v0 = mid[k]
            if torch.is_floating_point(v0):
                # average in fp32 for numerical stability
                stack = torch.stack([sd[k].to(torch.float32) for sd in sds], dim=0).mean(dim=0)
                if target_dtype is not None and stack.dtype != target_dtype:
                    stack = stack.to(target_dtype)
                avg[k] = stack
            else:
                avg[k] = v0  # copy non-float buffers as-is
        return avg

    def _svd_truncate(self, W: torch.Tensor, rank: int) -> torch.Tensor:
        if rank <= 0 or rank >= min(W.shape):
            return W
        W_cpu = W.detach().to("cpu", torch.float32)
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        r = min(rank, S.shape[0])
        recon = (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]).to(W.dtype)
        return recon

    def _consensus_project(self, Ws: List[torch.Tensor], rank: int, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W0 = Ws[0].detach().to("cpu", torch.float32)
        out, in_ = W0.shape
        C_row = torch.zeros((out, out), dtype=torch.float32, device="cpu")
        C_col = torch.zeros((in_, in_), dtype=torch.float32, device="cpu")
        for W in Ws:
            Wf = W.detach().to("cpu", torch.float32)
            C_row += Wf @ Wf.T
            C_col += Wf.T @ Wf
        C_row += eps * torch.eye(out, dtype=torch.float32, device="cpu")
        C_col += eps * torch.eye(in_, dtype=torch.float32, device="cpu")

        _, P = torch.linalg.eigh(C_row)  # ascending
        _, Q = torch.linalg.eigh(C_col)
        P = P[:, -rank:] if rank < P.shape[1] else P
        Q = Q[:, -rank:] if rank < Q.shape[1] else Q

        cores = []
        for W in Ws:
            Wf = W.detach().to("cpu", torch.float32)
            Ci = P.T @ Wf @ Q
            cores.append(Ci)
        C_bar = torch.stack(cores, dim=0).mean(dim=0)
        Wtilde = (P @ C_bar @ Q.T).to(dtype=W0.dtype)
        return Wtilde, P, Q

    # --- build replacement layer from heuristics ---
    def _build_replacement_layer(
        self,
        model,
        start: int,
        end_inclusive: int,
        strategy: str,
        rank: int = 16,
    ) -> Tuple[nn.Module, Dict]:
        """
        Returns (new_layer, info_dict) without mutating the model.
        strategy ∈ {'tblock_blank','tblock_avg','tblock_lr_mean','tblock_consensus'}
        """
        device = next(model.parameters()).device
        # choose the source sds:
        needs_fp = (strategy in ("tblock_avg", "tblock_lr_mean", "tblock_consensus"))
        if needs_fp and self._model_uses_bnb(model):
            sds = self._gather_removed_layers_state_dicts_fp_shadow(start, end_inclusive, dtype=torch.float32)
        else:
            sds = self._gather_removed_layers_state_dicts(model, start, end_inclusive)

        K = len(sds)
        mid_sd = sds[K // 2]

        new_layer = self._new_decoder_layer(model, layer_idx=start)
        # pick a true compute dtype from the host model
        target_dtype = self._effective_compute_dtype(model)
        # ensure the fresh layer is already at the correct dtype/device
        new_layer.to(dtype=target_dtype, device=next(model.parameters()).device)

        info = {"strategy": strategy, "rank": rank, "removed_layers": [start, end_inclusive]}
        out_sd = {}

        if strategy == "tblock_blank":
            self._init_decoder_layer_as_identity(new_layer, noise_std=getattr(self, "blank_noise_std", 0.0))
            # keep it consistent
            new_layer.to(dtype=target_dtype, device=next(model.parameters()).device)
            return new_layer, info

        if strategy == "tblock_avg":
            out_sd = self._avg_state_dicts(sds, target_dtype=target_dtype)
            self._load_layer_state_dict_safe(new_layer, out_sd)
            # ensure final dtype/device exactly match model compute
            new_layer.to(dtype=target_dtype, device=next(model.parameters()).device)
            return new_layer, info

        # low-rank strategies
        keys = mid_sd.keys()
        if strategy in ("tblock_lr_mean", "tblock_consensus"):
            perkey_list: Dict[str, List[torch.Tensor]] = {}
            for k in keys:
                if not torch.is_floating_point(mid_sd[k]):
                    continue
                perkey_list[k] = [sd[k].to(device) for sd in sds]

            for k in keys:
                t = mid_sd[k]
                if not torch.is_floating_point(t):
                    out_sd[k] = t
                    continue
                tensors = perkey_list[k]
                if t.ndim == 2:
                    if strategy == "tblock_lr_mean":
                        W_mean = torch.stack(tensors, dim=0).mean(dim=0)
                        out_sd[k] = self._svd_truncate(W_mean, rank)
                    else:
                        Wtilde, _, _ = self._consensus_project(tensors, rank=rank)
                        out_sd[k] = Wtilde
                elif t.ndim == 1:
                    if "norm" in k or "ln" in k:
                        out_sd[k] = t
                    else:
                        out_sd[k] = torch.stack(tensors, dim=0).mean(dim=0)
                else:
                    out_sd[k] = t
            for kk, vv in list(out_sd.items()):
                if torch.is_tensor(vv) and vv.dtype.is_floating_point and vv.dtype != target_dtype:
                    out_sd[kk] = vv.to(target_dtype)
            self._load_layer_state_dict_safe(new_layer, out_sd)
            new_layer.to(dtype=target_dtype, device=next(model.parameters()).device)
            return new_layer, info

        raise ValueError(f"Unknown replacement strategy: {strategy}")

    # --- save/load recipes to keep quantized sizes tiny ---
    def _save_inserted_layer_recipe(self, model, inserted_idx: int, path: str, meta: Dict):
        os.makedirs(path, exist_ok=True)
        # save inserted layer only
        parent, layers = self._get_layer_container_and_list(model)
        layer_sd = layers[inserted_idx].state_dict()
        torch.save(layer_sd, os.path.join(path, "inserted_layer.pt"))
        with open(os.path.join(path, "recipe.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _rebuild_from_recipe(self, recipe_dir: str):
        with open(os.path.join(recipe_dir, "recipe.json"), "r") as f:
            meta = json.load(f)
        m = self.clone_fresh_model()
        s, e = int(meta["selected_start"]), int(meta["selected_end_inclusive"])
        strat = meta["strategy"]
        rank = int(meta.get("rank", self.default_rank))
        new_layer, _ = self._build_replacement_layer(m, s, e, strategy=strat, rank=rank)
        self.prune_without_replacement(m, s, e)
        self._insert_replacement(m, s, new_layer)
        # load trained inserted layer weights if present
        pt_path = os.path.join(recipe_dir, "inserted_layer.pt")
        if os.path.exists(pt_path):
            parent, layers = self._get_layer_container_and_list(m)
            sd = torch.load(pt_path, map_location="cpu")
            layers[s].load_state_dict(sd, strict=False)
        return m, s

    # --- Single-layer healing (freeze everything except the inserted layer) ---
    def heal_inserted_layer(
            self,
            model,
            inserted_idx: int,
            train_dataloader,
            eval_dataloader=None,
            max_steps: int = 500,
            lr: float = 2e-4,
            grad_accum: int = 1,
    ) -> float:
        """
        Train ONLY the inserted layer with LM loss, masking pads.
        Fast path: avoid logits/materialized outputs; keep extras off; hoist autocast.
        """
        parent, layers = self._get_layer_container_and_list(model)
        train_layer = layers[inserted_idx]

        # Keep compute dtype consistent with base (important w/ k-bit)
        comp_dtype = self._infer_compute_dtype(model)
        dev = next(model.parameters()).device
        train_layer.to(device=dev, dtype=comp_dtype)

        # Freeze everything except the inserted layer
        for p in model.parameters():      p.requires_grad_(False)
        for p in train_layer.parameters(): p.requires_grad_(True)

        # Slim the training graph
        if hasattr(model.config, "use_cache"):             model.config.use_cache = False
        if hasattr(model.config, "output_attentions"):     model.config.output_attentions = False
        if hasattr(model.config, "output_hidden_states"):  model.config.output_hidden_states = False
        model.train()

        # Fused AdamW when available
        fused_ok = torch.cuda.is_available()
        opt = torch.optim.AdamW(
            [p for p in train_layer.parameters() if p.requires_grad],
            lr=lr,
            fused=fused_ok
        )

        # Hoist autocast outside the loop and set explicit dtype if possible
        amp_dtype = comp_dtype if comp_dtype in (torch.float16, torch.bfloat16) else None
        autocast_cm = torch.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext

        step = acc = 0
        with autocast_cm("cuda", dtype=amp_dtype):
            for batch in tqdm(train_dataloader, desc=f"Heal@layer{inserted_idx}"):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

                # label pads as -100
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                # Lean forward: no logits/hidden/attn objects, no cache
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    # -> (loss, logits?) but many heads skip logits computation when labels are given
                )
                loss = out[0] if isinstance(out, (tuple, list)) else out.loss
                loss.backward()

                acc += 1
                if acc % grad_accum == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step >= max_steps:
                        break

        # Eval PPL (keep it lean; your compute_perplexity already is)
        for p in train_layer.parameters(): p.requires_grad_(False)
        model.eval()
        if hasattr(model.config, "use_cache"): model.config.use_cache = True

        eval_loader = self._pick_eval_loader(train_dataloader, eval_dataloader)
        return self.compute_perplexity(eval_loader, model=model)

    # --- Full-model QLoRA healing (PEFT) ---
    def full_heal_qLoRA(
            self,
            model,
            train_dataloader,
            eval_dataloader=None,
            max_steps: int = 1000,
            lr: float = 1e-4,
            r: int = 16,
            alpha: int = 32,
            dropout: float = 0.05,
            target_modules: Optional[List[str]] = None,
            grad_accum: int = 1,
            save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Apply LoRA/QLoRA to attention+MLP modules and fine-tune with LM loss.
        Saves adapters under save_dir and returns {'ok', 'adapter_path', 'ppl'}.
        """
        out = {"ok": False, "error": None, "adapter_path": None, "ppl": None}

        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as e:
            out["error"] = f"Missing peft package for QLoRA: {e}"
            return out

        if target_modules is None:
            # Works across Llama/Qwen/NeoX families
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "w1", "w2", "w3", "dense_h_to_4h", "dense_4h_to_h"
            ]

        # ---- training prep (important for k-bit) ----
        model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        try:
            model = prepare_model_for_kbit_training(model)  # enables gckpt + input grads if available
        except Exception:
            pass
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        lconf = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lconf)
        # Ensure LoRA trainables are in bf16/fp16 (never fp32) for stability + speed
        comp_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                      else torch.float16)
        for n, p in model.named_parameters():
            if p.requires_grad and p.is_floating_point() and p.dtype != comp_dtype:
                p.data = p.data.to(comp_dtype)

        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        step = 0
        acc = 0

        autocast_cm = torch.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext
        for batch in tqdm(train_dataloader, desc="QLoRA-heal"):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            # mask pads to -100 so loss ignores them
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            with autocast_cm("cuda"):
                out_fw = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out_fw.loss

            loss.backward()
            acc += 1
            if acc % grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if step >= max_steps:
                break

        # ---- save adapters BEFORE any merge attempt ----
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Ensure dtype before saving to avoid silent FP32 bloat
            try:
                model.to(self.torch_dtype)
            except Exception:
                pass
            try:
                model.save_pretrained(save_dir)
                out["adapter_path"] = save_dir
            except Exception as e:
                out["error"] = f"Failed to save adapters: {e}"
                return out

        # ---- fast eval: prefer merged weights; otherwise disable training-time overheads ----
        eval_loader = eval_dataloader or train_dataloader
        if eval_loader is not None:
            eval_model = model
            merged = None
            try:
                # Skip merge for 4-bit bases (rounding diff + warning)
                is_4bit = getattr(getattr(model, "base_model", model), "is_loaded_in_4bit", False)
                if hasattr(model, "merge_and_unload") and not is_4bit:
                    merged = model.merge_and_unload()
                    merged.to(self.device)
                    merged.eval()
                    if hasattr(merged.config, "use_cache"):
                        merged.config.use_cache = True
                    if hasattr(merged, "gradient_checkpointing_disable"):
                        merged.gradient_checkpointing_disable()
                    eval_model = merged
            except Exception:
                pass

            if eval_model is model:
                model.eval()
                if hasattr(model, "gradient_checkpointing_disable"):
                    model.gradient_checkpointing_disable()
                for p in model.parameters():
                    p.requires_grad_(False)
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = True

            out["ppl"] = self.compute_perplexity(eval_loader, model=eval_model)

            if merged is not None:
                del merged
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

        out["ok"] = True
        return out

    # --- CSV selection helpers (read aggregate picks) ---
    def _read_aggregate_row(self, csv_path: str, block_size: int) -> Optional[Dict]:
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        row = df[df["block_size"] == block_size]
        if row.empty:
            return None
        return {
            "start": int(row["chosen_start_layer"].iloc[0]),
            "end_incl": int(row["chosen_end_layer_inclusive"].iloc[0]),
            "score": float(row["combined_score"].iloc[0]),
        }

    # --- (b) Single-shot pruning_sweep sweep: no-replacement + 4 heuristics (pre-heal only) ---
    def run_single_shot_pruning_sweep(
        self,
        eval_dataloader,
        agg_method_csvs: Dict[str, str],
        train_dataloader=None,
        percentages: Optional[List[int]] = None,
        strategies: List[str] = ("none", "tblock_blank", "tblock_avg", "tblock_lr_mean", "tblock_consensus"),
        rank: Optional[int] = None,
        save_models: bool = True,
        results_tag: str = "preheal_sweep",
    ) -> pd.DataFrame:
        """
        For each percent and strategy:
          - reload a fresh model
          - prune (with/without replacement)
          - compute perplexity
          - save (quantized: recipe only; non-quantized: full model)
          - log CSV
        """
        percentages = percentages or self.default_percentages
        rank = rank if rank is not None else self.default_rank

        # Baseline sanity + PPL (use same loader as variants; strict GPU; original dtype)
        eval_loader = self._pick_eval_loader(train_dataloader, eval_dataloader)
        base_for_ppl = self._fresh_model_for_pruning(quantized=False, on_cpu=False)

        layers, _ = _arch_probe(base_for_ppl)
        L0 = len(layers)

        # Reuse the already-loaded baseline model for sanity to avoid reloading different dtypes
        sanity = self.sanity_check(eval_loader, max_batches=1, model=base_for_ppl)
        baseline_ppl = self.compute_perplexity(eval_loader, model=base_for_ppl)
        print(f"Baseline PPL: {baseline_ppl:.3f} (layers={sanity['num_layers']}, hidden={sanity['hidden_size']})")

        del base_for_ppl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        records = []
        slug = self._slugify(self.model_name)
        for pct in percentages:
            remove_L = int(round(L0 * pct / 100.0))
            for agg_name, csv_path in agg_method_csvs.items():
                sel_none = self._read_aggregate_row(csv_path, block_size=remove_L)
                sel_repl = self._read_aggregate_row(csv_path, block_size=remove_L + 1)

                for strategy in strategies:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    gc.collect()
                    # load full-precision for pruning_sweep; set on_cpu=True if you want DRAM-only pruning_sweep
                    m = self._fresh_model_for_pruning(quantized=False, on_cpu=False)
                    if strategy == "none":
                        if not sel_none:
                            print(f"[{pct}% {agg_name} none] No selection for block_size={remove_L}")
                            del m; gc.collect(); continue
                        s, e = sel_none["start"], sel_none["end_incl"]
                        self.prune_without_replacement(m, s, e)
                        inserted_idx = None
                    else:
                        if not sel_repl:
                            print(f"[{pct}% {agg_name} {strategy}] No selection for block_size={remove_L+1}")
                            del m; gc.collect(); continue
                        s, e = sel_repl["start"], sel_repl["end_incl"]
                        new_layer, info = self._build_replacement_layer(m, s, e, strategy=strategy, rank=rank)
                        self.prune_without_replacement(m, s, e)
                        self._insert_replacement(m, s, new_layer)
                        inserted_idx = s

                    ppl = self.compute_perplexity(eval_loader, model=m)

                    rec = {
                        "method": agg_name,
                        "strategy": strategy,
                        "percent": pct,
                        "selected_start": s,
                        "selected_end_inclusive": e,
                        "baseline_ppl": baseline_ppl,
                        "post_ppl": ppl,
                        "delta_ppl": ppl - baseline_ppl,
                        "inserted_idx": inserted_idx,
                        "rank": rank,
                    }
                    records.append(rec)

                    # save
                    sub = os.path.join(self.results_dir, f"{slug}_{agg_name}_p{pct}_{strategy}_preheal")
                    os.makedirs(sub, exist_ok=True)
                    try:
                        if strategy == "none":
                            # no replacement to store; save only meta
                            with open(os.path.join(sub, "meta.json"), "w") as f:
                                json.dump(rec, f, indent=2)
                        else:
                            if self._model_uses_bnb(m):
                                # quantized: save recipe only
                                meta = rec.copy()
                                meta["model_name"] = self.model_name
                                self._save_inserted_layer_recipe(m, inserted_idx, sub, meta)
                            else:
                                # Ensure dtype before saving to avoid silent FP32 bloat
                                try:
                                    m.to(self.torch_dtype)
                                except Exception:
                                    pass
                                m.save_pretrained(sub)
                                with open(os.path.join(sub, "meta.json"), "w") as f:
                                    json.dump(rec, f, indent=2)
                    except Exception as ex:
                        print(f"Warning: failed saving model to {sub}: {ex}")

                    del m
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    gc.collect()

        df = pd.DataFrame(records)
        out_csv = os.path.join(self.results_dir, f"{self._slugify(self.model_name)}_{results_tag}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[Pre-heal sweep] Saved: {out_csv}")
        # log
        with open(os.path.join(self.results_dir, "log.json"), "w") as f:
            json.dump({"baseline_ppl": baseline_ppl, "preheal": records}, f, indent=2)
        return df

    # --- (c) Single-layer healing for the 4 replacement strategies ---
    def run_single_layer_heal_sweep(
            self,
            train_dataloader,
            agg_method_csvs: Dict[str, str],
            eval_dataloader=None,
            percentages: Optional[List[int]] = None,
            strategies: List[str] = ("tblock_blank", "tblock_avg", "tblock_lr_mean", "tblock_consensus"),
            rank: Optional[int] = None,
            max_steps: Optional[int] = None,
            lr: float = 2e-4,
            grad_accum: int = 1,
            save_models: bool = True,
            results_tag: str = "single_layer_heal",
    ) -> pd.DataFrame:
        """
        Reuse the pre-heal variants saved by run_single_shot_pruning_sweep when possible:
          - If quantized save: rebuild exactly from recipe.json + inserted_layer.pt
          - If full-precision save: load from the preheal directory
          - Fallback: rebuild from selection CSVs (old behavior)
        """
        percentages = percentages or self.default_percentages
        rank = rank if rank is not None else self.default_rank
        max_steps = max_steps if max_steps is not None else self.default_max_heal_steps

        records, slug = [], self._slugify(self.model_name)

        pre_csv = os.path.join(self.results_dir, f"{slug}_preheal_sweep.csv")
        pre_df = pd.read_csv(pre_csv) if os.path.exists(pre_csv) else None

        for pct in percentages:
            # compute selection once
            L = self.get_layer_count()
            sel_block = int(round(L * pct / 100.0)) + 1  # for replacement
            for agg_name, csv_path in agg_method_csvs.items():
                sel_repl = self._read_aggregate_row(csv_path, block_size=sel_block)

                for strategy in strategies:
                    # --- Prefer exact coords and ppl_before from preheal CSV
                    s_use = e_use = None
                    ppl_before = None
                    if pre_df is not None:
                        row = pre_df[(pre_df["method"] == agg_name) &
                                     (pre_df["strategy"] == strategy) &
                                     (pre_df["percent"] == pct)]
                        if not row.empty:
                            s_use = int(row["selected_start"].iloc[0])
                            e_use = int(row["selected_end_inclusive"].iloc[0])
                            ppl_before = float(row["post_ppl"].iloc[0])

                    if s_use is None or e_use is None:
                        if not sel_repl:
                            print(f"[Heal] Missing selection for {agg_name} {pct}%")
                            continue
                        s_use, e_use = sel_repl["start"], sel_repl["end_incl"]

                    # --- Reuse the preheal model if saved
                    preheal_dir = os.path.join(self.results_dir, f"{slug}_{agg_name}_p{pct}_{strategy}_preheal")
                    m, inserted_idx = None, s_use
                    if os.path.isdir(preheal_dir):
                        recipe_path = os.path.join(preheal_dir, "recipe.json")
                        weights_dir = os.path.join(preheal_dir, "config.json")  # HF marker for full save
                        if os.path.exists(recipe_path):
                            # quantized recipe: deterministic rebuild
                            m, inserted_idx = self._rebuild_from_recipe(preheal_dir)
                        elif os.path.exists(weights_dir):
                            # full-precision save: load directly
                            m = AutoModelForCausalLM.from_pretrained(
                                preheal_dir,
                                torch_dtype=self.torch_dtype,
                                trust_remote_code=True
                            ).to(self.device)
                        else:
                            m = None  # fall back
                    if m is None:
                        # Fallback: rebuild from scratch (old behavior)
                        m = self._fresh_model_for_phase("single_heal")
                        new_layer, _ = self._build_replacement_layer(m, s_use, e_use, strategy=strategy, rank=rank)
                        self.prune_without_replacement(m, s_use, e_use)
                        self._insert_replacement(m, s_use, new_layer)
                        inserted_idx = s_use

                    # ppl_before from preheal CSV if available; otherwise compute now
                    if ppl_before is None:
                        eval_loader = self._pick_eval_loader(train_dataloader, eval_dataloader)
                        ppl_before = self.compute_perplexity(eval_loader, model=m)

                    # Use custom fine-tuning
                    # ppl_after = self.heal_inserted_layer(
                    #     m, inserted_idx,
                    #     train_dataloader=train_dataloader,
                    #     eval_dataloader=eval_dataloader,
                    #     max_steps=max_steps, lr=lr, grad_accum=grad_accum
                    # )

                    # Use Trainer
                    ppl_after = self.heal_inserted_layer_trainer(
                        m, inserted_idx,
                        train_dataloader=train_dataloader,
                        eval_dataloader=eval_dataloader,
                        max_steps=max_steps, lr=lr, grad_accum=grad_accum
                    )

                    rec = {
                        "method": agg_name,
                        "strategy": strategy,
                        "percent": pct,
                        "selected_start": s_use,
                        "selected_end_inclusive": e_use,
                        "ppl_before_heal": ppl_before,
                        "ppl_after_heal": ppl_after,
                        "delta_ppl": ppl_after - ppl_before,
                        "inserted_idx": inserted_idx,
                        "rank": rank,
                    }
                    records.append(rec)

                    # Save compactly (same semantics as before)
                    # save
                    sub = os.path.join(self.results_dir, f"{slug}_{agg_name}_p{pct}_{strategy}_preheal")
                    os.makedirs(sub, exist_ok=True)
                    try:
                        if strategy == "none":
                            with open(os.path.join(sub, "meta.json"), "w") as f:
                                json.dump(rec, f, indent=2)
                        else:
                            if self._model_uses_bnb(m):
                                meta = rec.copy()
                                meta["model_name"] = self.model_name
                                self._save_inserted_layer_recipe(m, inserted_idx, sub, meta)
                            else:
                                # Ensure dtype before saving to avoid silent FP32 bloat
                                try:
                                    m.to(self.torch_dtype)
                                except Exception:
                                    pass
                                m.save_pretrained(sub)
                                with open(os.path.join(sub, "meta.json"), "w") as f:
                                    json.dump(rec, f, indent=2)
                    except Exception as e:
                        print(f"[WARN] Save failed for {sub}: {e}")

                    del m
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    gc.collect()

        df = pd.DataFrame(records)
        out_csv = os.path.join(self.results_dir, f"{slug}_{results_tag}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[Single-layer heal] Saved: {out_csv}")

        # log append
        log_path = os.path.join(self.results_dir, "log.json")
        log = {}
        if os.path.exists(log_path):
            with open(log_path, "r") as f: log = json.load(f)
        log["single_layer_heal"] = records
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        return df

    # --- (d) Full QLoRA healing for three models across percentages ---
    def run_full_heal_experiments(
        self,
        train_dataloader,
        agg_method_csvs: Dict[str, str],
        eval_dataloader=None,
        percentages: Optional[List[int]] = None,
        rank_repl: Optional[int] = None,
        max_qlora_steps: Optional[int] = None,
        qlora_lr: float = 1e-4,
        qlora_r: int = 16,
        qlora_alpha: int = 32,
        qlora_dropout: float = 0.05,
        grad_accum: int = 1,
        target_modules: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        For each percentage runs QLoRA full healing on three models:
          (i)  pruned without replacement,
          (ii) pruned + BLANK replacement (pre single-layer heal),
          (iii) best model from single-layer healing sweep.
        Saves adapters and logs perplexities. Rebuilds quantized variants from recipes.
        """
        percentages = percentages or self.default_percentages
        rank_repl = rank_repl if rank_repl is not None else self.default_rank
        max_qlora_steps = max_qlora_steps if max_qlora_steps is not None else self.default_max_qlora_steps
        eval_loader = self._pick_eval_loader(train_dataloader, eval_dataloader)

        slug = self._slugify(self.model_name)
        variants_all = []

        # cache precomputed PPLs
        preheal_csv = os.path.join(self.results_dir, f"{slug}_preheal_sweep.csv")
        pre_df = pd.read_csv(preheal_csv) if os.path.exists(preheal_csv) else None

        for full_heal_percent in percentages:
            L = self.get_layer_count()
            remove_L = int(round(L * full_heal_percent / 100.0))
            # pick any aggregate (call twice if you want both)
            agg_name, csv_path = list(agg_method_csvs.items())[0]
            sel_none = self._read_aggregate_row(csv_path, block_size=remove_L)
            sel_repl = self._read_aggregate_row(csv_path, block_size=remove_L + 1)
            if not sel_none or not sel_repl:
                print(f"[Full-heal] Missing selections for {full_heal_percent}% at {agg_name}. Skipping.")
                continue
            s_none, e_none = sel_none["start"], sel_none["end_incl"]
            s_repl, e_repl = sel_repl["start"], sel_repl["end_incl"]

            # (i) None
            m1 = self._fresh_model_for_phase("qlora")
            self.prune_without_replacement(m1, s_none, e_none)
            ppl1_before = None
            if pre_df is not None:
                row = pre_df[
                    (pre_df["method"] == agg_name) &
                    (pre_df["strategy"] == "none") &
                    (pre_df["percent"] == full_heal_percent)
                    ]
                if not row.empty:
                    ppl1_before = float(row["post_ppl"].iloc[0])
            if ppl1_before is None:
                ppl1_before = self.compute_perplexity(eval_loader, model=m1)

            adapt_dir1 = os.path.join(self.results_dir, f"{slug}_{agg_name}_p{full_heal_percent}_none_qlora")
            # Use custom finetuning
            # q1 = self.full_heal_qLoRA(m1, train_dataloader, eval_dataloader=eval_loader,
            #                           max_steps=max_qlora_steps, lr=qlora_lr, r=qlora_r, alpha=qlora_alpha,
            #                           dropout=qlora_dropout, target_modules=target_modules, grad_accum=grad_accum,
            #                           save_dir=adapt_dir1)
            # Use Trainer
            q1 = self.full_heal_qLoRA_trainer(m1, train_dataloader, eval_dataloader=eval_loader,
                                               max_steps=max_qlora_steps, lr=qlora_lr, r=qlora_r, alpha=qlora_alpha,
                                               dropout=qlora_dropout, target_modules=target_modules,
                                               grad_accum=grad_accum, save_dir=adapt_dir1)

            ppl1_after = q1.get("ppl", None) or self.compute_perplexity(eval_loader or train_dataloader, model=m1)
            variants_all.append({
                "variant": "none",
                "percent": full_heal_percent,
                "method": agg_name,
                "ppl_pre_qlora": ppl1_before,
                "ppl_post_qlora": ppl1_after,
                "adapters": q1.get("adapter_path"),
                "ok": q1.get("ok", False),
                "error": q1.get("error"),
            })
            del m1; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # (ii) Blank replacement pre single-layer heal
            m2 = self._fresh_model_for_phase("qlora")
            new_layer2, _ = self._build_replacement_layer(m2, s_repl, e_repl, strategy="tblock_blank", rank=rank_repl)
            self.prune_without_replacement(m2, s_repl, e_repl)
            self._insert_replacement(m2, s_repl, new_layer2)
            ppl2_before = None
            if pre_df is not None:
                row = pre_df[
                    (pre_df["method"] == agg_name) &
                    (pre_df["strategy"] == "tblock_blank") &
                    (pre_df["percent"] == full_heal_percent)
                    ]
                if not row.empty:
                    ppl2_before = float(row["post_ppl"].iloc[0])
            if ppl2_before is None:
                ppl2_before = self.compute_perplexity(eval_loader, model=m2)

            adapt_dir2 = os.path.join(self.results_dir, f"{slug}_{agg_name}_p{full_heal_percent}_tblock_blank_preheal_qlora")
            # Use custom finetuning
            # q2 = self.full_heal_qLoRA(m2, train_dataloader, eval_dataloader=eval_loader,
            #                           max_steps=max_qlora_steps, lr=qlora_lr, r=qlora_r, alpha=qlora_alpha,
            #                           dropout=qlora_dropout, target_modules=target_modules, grad_accum=grad_accum,
            #                           save_dir=adapt_dir2)
            # Use Trainer
            q2 = self.full_heal_qLoRA_trainer(m2, train_dataloader, eval_dataloader=eval_loader,
                                              max_steps=max_qlora_steps, lr=qlora_lr, r=qlora_r, alpha=qlora_alpha,
                                              dropout=qlora_dropout, target_modules=target_modules,
                                              grad_accum=grad_accum, save_dir=adapt_dir2)
            ppl2_after = q2.get("ppl", None) or self.compute_perplexity(eval_loader or train_dataloader, model=m2)
            variants_all.append({
                "variant": "tblock_blank_preheal",
                "percent": full_heal_percent,
                "method": agg_name,
                "ppl_pre_qlora": ppl2_before,
                "ppl_post_qlora": ppl2_after,
                "adapters": q2.get("adapter_path"),
                "ok": q2.get("ok", False),
                "error": q2.get("error"),
            })
            del m2; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # (iii) Best-of-(c) single-layer healed
            heal_csv = os.path.join(self.results_dir, f"{slug}_single_layer_heal.csv")
            if not os.path.exists(heal_csv):
                raise RuntimeError("Single-layer heal CSV not found. Run run_single_layer_heal_sweep first.")
            df = pd.read_csv(heal_csv)
            df_pct = df[df["percent"] == full_heal_percent]
            if df_pct.empty:
                print(f"[Full-heal] No single-layer heal results for {full_heal_percent}%. Skipping.")
                continue
            best_row = df_pct.sort_values(by="ppl_after_heal").iloc[0]
            strat_best = best_row["strategy"]

            # Try to reconstruct from recipe if quantized / no full weights
            m3_dir = os.path.join(self.results_dir, f"{slug}_{best_row['method']}_p{full_heal_percent}_{strat_best}_singlelayer_healed")
            if os.path.isdir(m3_dir) and os.path.exists(os.path.join(m3_dir, "recipe.json")):
                m3, inserted_idx = self._rebuild_from_recipe(m3_dir)
            elif os.path.isdir(m3_dir):
                m3 = AutoModelForCausalLM.from_pretrained(
                    m3_dir,
                    torch_dtype=self.torch_dtype,  # preserve original weight dtype
                    trust_remote_code=True
                ).to(self.device)
            else:
                # fallback: rebuild from CSV selection
                m3 = self._fresh_model_for_phase("qlora")
                new_layer3, _ = self._build_replacement_layer(m3, s_repl, e_repl, strategy=strat_best, rank=rank_repl)
                self.prune_without_replacement(m3, s_repl, e_repl)
                self._insert_replacement(m3, s_repl, new_layer3)

            ppl3_before = float(best_row["ppl_after_heal"])
            adapt_dir3 = os.path.join(self.results_dir, f"{slug}_{best_row['method']}_p{full_heal_percent}_{strat_best}_healed_qlora")
            # Use custom finetuning
            # q3 = self.full_heal_qLoRA(m3, train_dataloader, eval_dataloader=eval_loader,
            #                           max_steps=max_qlora_steps, lr=qlora_lr, r=qlora_r, alpha=qlora_alpha,
            #                           dropout=qlora_dropout, target_modules=target_modules, grad_accum=grad_accum,
            #                           save_dir=adapt_dir3)
            # Use Trainer
            q3 = self.full_heal_qLoRA_trainer(m3, train_dataloader, eval_dataloader=eval_loader,
                                              max_steps=max_qlora_steps, lr=qlora_lr, r=qlora_r, alpha=qlora_alpha,
                                              dropout=qlora_dropout, target_modules=target_modules,
                                              grad_accum=grad_accum, save_dir=adapt_dir3)
            ppl3_after = q3.get("ppl", None) or self.compute_perplexity(eval_loader or train_dataloader, model=m3)
            variants_all.append({
                "variant": f"{strat_best}_post_single_layer_heal",
                "percent": full_heal_percent,
                "method": best_row["method"],
                "ppl_pre_qlora": ppl3_before,
                "ppl_post_qlora": ppl3_after,
                "adapters": q3.get("adapter_path"),
                "ok": q3.get("ok", False),
                "error": q3.get("error"),
            })
            del m3; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        out_csv = os.path.join(self.results_dir, f"{slug}_full_heal_qlora.csv")
        pd.DataFrame(variants_all).to_csv(out_csv, index=False)
        print(f"[Full-heal QLoRA] Saved: {out_csv}")

        # append to log
        log_path = os.path.join(self.results_dir, "log.json")
        log = {}
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
        log.setdefault("full_heal", [])
        log["full_heal"].extend(variants_all)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        return pd.DataFrame(variants_all)

    def run_single_flow_pruning(
        self,
        *,
        train_dataloader,
        eval_dataloader=None,
        percent: int,
        replacement: Optional[str] = None,           # None or one of: "tblock_blank","tblock_avg","tblock_lr_mean","tblock_consensus"
        heal_methods: Union[str, List[str]] = ("qlora",),  # allowed: "single_layer", "qlora" (order matters)
        # How to pick the span:
        agg_method: str = "weighted_mean",           # "weighted_mean" or "minimax"
        agg_method_csvs: Optional[Dict[str, str]] = None,  # if not provided, we'll call Analyzer
        analyzer: Optional["LayerPruningAnalyzer"] = None, # optional: pass a preconfigured analyzer
        task_token_targets: Optional[Dict[str, Dict[str, int]]] = None,  # if we must build CSVs
        analyzer_outdir: Optional[str] = None,
        analyzer_weights: Optional[Dict[str, float]] = None,
        analyzer_max_block_size: int = 15,
        token_max_length: int = 512,                 # used only if we must aggregate
        # Single-layer heal params
        single_lr: float = 2e-4,
        single_max_steps: int = 350,
        single_grad_accum: int = 1,
        # QLoRA params
        qlora_lr: float = 1e-4,
        qlora_max_steps: int = 700,
        qlora_grad_accum: int = 1,
        qlora_r: int = 16,
        qlora_alpha: int = 32,
        qlora_dropout: float = 0.05,
        qlora_target_modules: Optional[List[str]] = None,
        # Results/control
        experiment_dir: Optional[str] = None,
        save_variant: bool = True,
        on_cpu_for_prune: bool = False,              # if True, prune on CPU (shadow) to save VRAM
    ) -> Dict:
        """
        End-to-end: baseline -> analyze/aggregate (if needed) -> prune -> heal(s) -> eval -> save.
        Returns a dictionary with span selection, ppl baselines, post-prune, post-heal metrics and save paths.
        """
        # -------- sanity & setup --------
        if isinstance(heal_methods, str):
            heal_methods = [heal_methods]
        heal_methods = [h.lower() for h in heal_methods]
        assert all(h in ("single_layer", "qlora") for h in heal_methods), f"heal_methods must be from ['single_layer','qlora']"

        eval_loader = self._pick_eval_loader(train_dataloader, eval_dataloader)
        baseline_model = self._fresh_model_for_phase("ppl")
        baseline_ppl = self.compute_perplexity(eval_loader, model=baseline_model)
        del baseline_model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()

        # Layer count for percent->block_size
        tmp_model = self._fresh_model_for_pruning(quantized=False, on_cpu=True)
        layers, _ = _arch_probe(tmp_model)
        L0 = len(layers)
        del tmp_model; gc.collect()

        remove_L = int(round(L0 * percent / 100.0))

        # -------- ensure aggregate CSVs (if needed) --------
        if agg_method_csvs is None:
            if analyzer is None:
                from LayerPruningAnalyzer import LayerPruningAnalyzer  # local import to avoid circulars
                analyzer = LayerPruningAnalyzer(
                    model_name=self.model_name,
                    results_dir=analyzer_outdir or os.path.join(self.results_dir, "analysis"),
                    device=self.device,
                    dtype=("bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else ("fp16" if torch.cuda.is_available() else "fp32")),
                    trust_remote_code=self.trust_remote_code,
                    batch_size=getattr(self, "analysis_batch_size", 4),
                    max_samples=getattr(self, "analysis_max_samples", 2000),
                )
                analyzer.tokenizer  # lazy init

            if task_token_targets is None:
                raise ValueError("run_single_flow_pruning: need 'agg_method_csvs' or 'task_token_targets' to build aggregates.")

            agg_method_csvs = analyzer.ensure_aggregate_csvs_for_tasks(
                task_token_targets=task_token_targets,
                outdir=analyzer_outdir or os.path.join(self.results_dir, "analysis"),
                max_block_size=analyzer_max_block_size,
                weights=analyzer_weights,
                token_max_length=token_max_length,
                batch_size=getattr(self, "analysis_batch_size", 4),
            )

        # pick selection row(s)
        csv_path = agg_method_csvs["weighted_mean"] if agg_method == "weighted_mean" else agg_method_csvs["minimax"]
        sel_none = self._read_aggregate_row(csv_path, block_size=remove_L)              # for 'none'
        sel_repl = self._read_aggregate_row(csv_path, block_size=remove_L + 1)         # for insertion strategies
        if replacement in (None, "none") and not sel_none:
            raise RuntimeError(f"No selection found for block_size={remove_L} in {csv_path}")
        if replacement not in (None, "none") and not sel_repl:
            raise RuntimeError(f"No selection found for block_size={remove_L+1} in {csv_path}")

        # -------- prune (optionally insert replacement) --------
        # Use non-quantized or phase-specific profile for reliable weight surgery
        m = self._fresh_model_for_pruning(quantized=False, on_cpu=on_cpu_for_prune)
        if replacement in (None, "none"):
            s, e = sel_none["start"], sel_none["end_incl"]
            self.prune_without_replacement(m, s, e)                                      # pruning_sweep path unchanged :contentReference[oaicite:5]{index=5}
            inserted_idx = None
        else:
            s, e = sel_repl["start"], sel_repl["end_incl"]
            new_layer, _ = self._build_replacement_layer(m, s, e, strategy=replacement)  # you already support these strategies
            self.prune_without_replacement(m, s, e)
            self._insert_replacement(m, s, new_layer)

            inserted_idx = s

        ppl_post_prune = self.compute_perplexity(eval_loader, model=m)

        # Prepare experiment directory
        slug = self._slugify(self.model_name)
        flow_tag = f"p{percent}_{('none' if replacement in (None,'none') else replacement)}_{'-'.join(heal_methods) or 'noheal'}"
        exp_dir = experiment_dir or os.path.join(self.results_dir, "single_flow", slug, flow_tag)
        os.makedirs(exp_dir, exist_ok=True)

        meta = {
            "model_name": self.model_name,
            "percent": percent,
            "block_size": remove_L,
            "selected_start": s,
            "selected_end_inclusive": e,
            "replacement": (replacement or "none"),
            "heal_methods": heal_methods,
            "agg_method": agg_method,
            "agg_csv": csv_path,
            "baseline_ppl": float(baseline_ppl),
            "post_prune_ppl": float(ppl_post_prune),
        }

        # Persist pre-heal variant if requested
        if save_variant:
            try:
                if self._model_uses_bnb(m):
                    self._save_inserted_layer_recipe(m, inserted_idx, os.path.join(exp_dir, "preheal_recipe"), meta.copy())
                else:
                    try:
                        m.to(self.torch_dtype)
                    except Exception:
                        pass
                    m.save_pretrained(os.path.join(exp_dir, "preheal_model"))
                    with open(os.path.join(exp_dir, "preheal_model", "meta.json"), "w") as f:
                        json.dump(meta, f, indent=2)
            except Exception as ex:
                print(f"[WARN] preheal save failed: {ex}")

        # -------- run healing steps (ordered) --------
        ppl_after_single = None
        ppl_after_qlora  = None
        adapters_path = None

        for step in heal_methods:
            if step == "single_layer":
                if inserted_idx is None:
                    raise RuntimeError("single_layer heal requires a replacement block to be inserted.")
                ppl_after_single = self.heal_inserted_layer_trainer(
                    m, inserted_idx=inserted_idx,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_loader,
                    max_steps=single_max_steps, lr=single_lr, grad_accum=single_grad_accum,
                    log_dir=os.path.join(exp_dir, "single_layer_logs"),
                )
                meta["post_single_layer_ppl"] = float(ppl_after_single)

            elif step == "qlora":
                q = self.full_heal_qLoRA_trainer(
                    m,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_loader,
                    max_steps=qlora_max_steps, lr=qlora_lr,
                    r=qlora_r, alpha=qlora_alpha, dropout=qlora_dropout,
                    target_modules=qlora_target_modules,
                    grad_accum=qlora_grad_accum,
                    save_dir=os.path.join(exp_dir, "qlora_adapters"),
                )
                adapters_path = q.get("adapter_path", adapters_path)
                ppl_after_qlora = q.get("ppl", None) or self.compute_perplexity(eval_loader, model=m)
                meta["post_qlora_ppl"] = float(ppl_after_qlora)
            else:
                raise ValueError(f"Unknown heal step: {step}")

        # Finalize meta and write run.json
        with open(os.path.join(exp_dir, "run.json"), "w") as f:
            json.dump(meta, f, indent=2)

        out = {
            "ok": True,
            "experiment_dir": exp_dir,
            "selection": {"start": s, "end_incl": e, "block_size": remove_L},
            "baseline_ppl": baseline_ppl,
            "post_prune_ppl": ppl_post_prune,
            "post_single_layer_ppl": ppl_after_single,
            "post_qlora_ppl": ppl_after_qlora,
            "adapters_path": adapters_path,
            "replacement": (replacement or "none"),
            "heal_methods": heal_methods,
        }

        # best-effort cleanup
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        gc.collect()

        return out

    # --- Progressive pruning_sweep ---
    def _save_model_for_analysis(self, model, out_dir: str):
        """
        Save `model` in a HF-loadable dir for analysis.
        Uses the same merge/compose logic as rollback.
        """
        self._save_checkpoint_for_rollback(model, out_dir)
        # breadcrumb for humans
        with open(os.path.join(out_dir, "ANALYZE_README.txt"), "w") as f:
            f.write("Checkpoint for per-step analysis (merged if possible).\n")

    def _save_checkpoint_for_rollback(self, model, out_dir: str):
        """
        Save a rollback checkpoint that the Analyzer and HF can reload as a plain model.
        Tries to merge PEFT adapters before saving; falls back to a base+adapters composition.
        Always saves the tokenizer too.
        """
        os.makedirs(out_dir, exist_ok=True)

        def _save_plain(m, tgt):
            try:
                m.to(self.torch_dtype)
            except Exception:
                pass
            m.save_pretrained(tgt)
            self.tokenizer.save_pretrained(tgt)

        # Try fast GPU merge when base isn't 4-bit
        try:
            from peft import PeftModel
            is_peft = isinstance(model, PeftModel) or hasattr(model, "merge_and_unload")
        except Exception:
            is_peft = hasattr(model, "merge_and_unload")

        if is_peft:
            try:
                is_4bit = getattr(getattr(model, "base_model", model), "is_loaded_in_4bit", False)
                if hasattr(model, "merge_and_unload") and not is_4bit:
                    merged = model.merge_and_unload()
                    merged.to(self.device).eval()
                    _save_plain(merged, out_dir)
                    return
            except Exception:
                pass

            # CPU merge fallback (keeps VRAM low)
            try:
                cpu_model = model.to("cpu")
                if hasattr(cpu_model, "merge_and_unload"):
                    merged_cpu = cpu_model.merge_and_unload()
                    _save_plain(merged_cpu, out_dir)
                    return
            except Exception:
                pass

            # Final fallback: compose base + adapters (Analyzer will handle this)
            base_dir = os.path.join(out_dir, "base")
            adapters_dir = os.path.join(out_dir, "adapters")
            try:
                # Save base
                base = getattr(model, "base_model", None) or getattr(model, "model", None)
                if base is None:
                    base = model
                _save_plain(base, base_dir)
                # Save adapters
                model.save_pretrained(adapters_dir)
                with open(os.path.join(out_dir, "PEFT_COMPOSE.json"), "w") as f:
                    json.dump({"base": "base", "adapters": "adapters"}, f)
                return
            except Exception:
                # If even that fails, last-ditch: save whatever we have
                _save_plain(model, out_dir)
                return
        else:
            _save_plain(model, out_dir)

    def run_progressive_pruning(
            self,
            *,
            train_dataloader,
            eval_dataloader=None,
            start_percent: int = 35,
            replacement: str = "tblock_avg",  # or "none"
            heal_methods: List[str] = ("single_layer",),
            # PPL ceiling — use *one* of these (ratio takes precedence if both are given)
            ppl_ratio_limit: Optional[float] = 1.15,  # e.g., allow +15% vs. baseline
            ppl_delta_limit: Optional[float] = None,  # e.g., allow +2.0 absolute
            # selection / analyzer
            agg_method: str = "weighted_mean",
            agg_method_csvs: Optional[Dict[str, str]] = None,
            analyzer: Optional["LayerPruningAnalyzer"] = None,
            task_token_targets: Optional[Dict[str, int]] = None,  # only used if we need to build CSVs
            analyzer_outdir: Optional[str] = None,
            analyzer_weights: Optional[Dict[str, float]] = None,
            analyzer_max_block_size: int = 15,
            token_max_length: int = 512,
            # heal hyperparams (forwarded to single-flow)
            single_lr: float = 2e-4,
            single_max_steps: int = 350,
            single_grad_accum: int = 2,
            qlora_lr: float = 1e-4,
            qlora_max_steps: int = 700,
            qlora_grad_accum: int = 2,
            qlora_r: int = 16,
            qlora_alpha: int = 32,
            qlora_dropout: float = 0.05,
            qlora_target_modules: Optional[List[str]] = None,
            # stopping rules & bookkeeping
            min_block: int = 2,  # stop when next block size < min_block
            experiment_dir: Optional[str] = None,
            save_variant_each_step: bool = True,
            final_qlora: bool = False,
            final_qlora_try_rejected_first: bool = True,
    ) -> Dict:
        """
        Progressive pruning_sweep with a PPL ceiling. At each step, prune+heal by current percent.
        If post-heal PPL > limit, halve percent and retry from the last accepted model.
        Stop when the block to prune falls below `min_block`.

        Returns a dict with baseline, limit, per-step records, and the final model artifact dir.
        """
        import json, os, math, gc
        import pandas as pd
        from copy import deepcopy

        # ---------- Setup / eval loader ----------
        eval_loader = self._pick_eval_loader(train_dataloader, eval_dataloader)

        # Baseline PPL on a fresh model (phase "ppl" quant profile if set)
        base_model = self._fresh_model_for_phase("ppl")
        baseline_ppl = self.compute_perplexity(eval_loader, model=base_model)
        del base_model;
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # PPL limit
        if ppl_ratio_limit is not None:
            ppl_limit = baseline_ppl * float(ppl_ratio_limit)
        elif ppl_delta_limit is not None:
            ppl_limit = baseline_ppl + float(ppl_delta_limit)
        else:
            raise ValueError("Provide ppl_ratio_limit or ppl_delta_limit.")

        # Initial working model for surgery (unquantized by default for reliability)
        working = self._fresh_model_for_pruning(quantized=False, on_cpu=False)
        layers, _ = _arch_probe(working)
        L0 = len(layers)

        if replacement != "none":
            min_block = min_block + 1

        global_stats = {
            "layers_start": int(L0),
            "layers_final": int(L0),  # will update as we accept steps
            "layers_pruned_total": 0,
            "pruned_percent_total": 0.0,
            "accepted_steps": 0,
            "rejected_steps": 0,
        }

        # ---------- experiment dir ----------
        slug = self._slugify(self.model_name)
        prog_dir = experiment_dir or os.path.join(self.results_dir, "progressive", slug)
        os.makedirs(prog_dir, exist_ok=True)

        # ---------- loop state ----------
        records = []
        step = 0
        percent = int(start_percent)

        # Keep a "last accepted" model to roll back to if we exceed the limit
        last_ok_dir = os.path.join(prog_dir, "_last_ok_ckpt")
        self._save_checkpoint_for_rollback(working, last_ok_dir)
        last_ok_ppl = baseline_ppl
        last_rejected_dir = None
        last_rejected_meta = None

        while True:
            L_now = len(_arch_probe(working)[0])
            layers_before = int(L_now)
            # decide block size as a percent of current length
            block_size = int(round(L_now * percent / 100.0))
            if replacement != "none":
                block_size = block_size + 1
            if block_size < min_block:
                print(f"[progressive] stopping: next block_size={block_size} < min_block={min_block}")
                break

            # ---- make new directory for anlysis of this step ----
            step_dir = os.path.join(prog_dir, f"step_{step:03d}_p{percent}_bs{block_size}")
            os.makedirs(step_dir, exist_ok=True)

            # 1) Save the current working model to a temp dir for analysis
            cur_ckpt_dir = os.path.join(step_dir, "analysis_model")
            self._save_model_for_analysis(working, cur_ckpt_dir)

            # 2) Run a fresh analysis on THIS checkpoint to get aggregate CSVs
            from LayerPruningAnalyzer import LayerPruningAnalyzer
            an = LayerPruningAnalyzer(
                model_name=cur_ckpt_dir,  # <-- local path works with HF
                results_dir=os.path.join(step_dir, "analysis"),  # step-local analysis dir
                device=self.device,
                dtype=("bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                       else ("fp16" if torch.cuda.is_available() else "fp32")),
                batch_size=getattr(self, "analysis_batch_size", 4),
                max_samples=getattr(self, "analysis_max_samples", 2000),
                trust_remote_code=self.trust_remote_code,
            )
            an.tokenizer = self.tokenizer  # bypasses AutoTokenizer.from_pretrained(cur_ckpt_dir)
            # Let analyzer default to equal weights if you don't pass any:
            csvs = an.ensure_aggregate_csvs_for_tasks(
                task_token_targets=task_token_targets or {"syntax": 50_000, "code": 50_000, "math": 50_000},
                outdir=os.path.join(step_dir, "analysis"),  # normalized by analyzer ctor
                max_block_size=analyzer_max_block_size,
                weights=analyzer_weights,  # keep None as None
                token_max_length=token_max_length,
                batch_size=getattr(self, "analysis_batch_size", 4),
            )

            csv_path = csvs["weighted_mean"] if agg_method == "weighted_mean" else csvs["minimax"]

            # 3) Read best candidate for THIS model and THIS block_size (no mapping needed now)
            import pandas as pd
            df = pd.read_csv(csv_path)
            if "distance" in df.columns:
                df = df.sort_values("distance", ascending=True)
            cand_df = df[df["block_size"] == int(block_size)]
            if cand_df.empty:
                print(f"[progressive] no candidates in CSV for block_size={block_size}; reducing percent.")
                percent = max(1, percent // 2)
                continue

            sel_row = cand_df.iloc[0].to_dict()
            # indices are already for the *current* model
            s_cur, e_cur = int(sel_row["chosen_start_layer"]), int(sel_row["chosen_end_layer_inclusive"])

            # Net-removed layers this step (replacement keeps 1)
            net_removed = int(block_size if (replacement in (None, "none")) else max(0, block_size - 1))

            # --- perform prune+insert (if replacement) on the *working* model ---
            if replacement in (None, "none"):
                self.prune_without_replacement(working, s_cur, e_cur)
                inserted_idx = None
            else:
                new_layer, _ = self._build_replacement_layer(working, s_cur, e_cur, strategy=replacement)
                self.prune_without_replacement(working, s_cur, e_cur)
                self._insert_replacement(working, s_cur, new_layer)
                inserted_idx = s_cur

            ppl_post_prune = self.compute_perplexity(eval_loader, model=working)

            # heal in the specified order (same trainer-based paths you already use)
            ppl_after_single = None
            ppl_after_qlora = None
            adapters_path = None

            step_dir = os.path.join(prog_dir, f"step_{step:03d}_p{percent}_bs{block_size}")
            os.makedirs(step_dir, exist_ok=True)

            for hm in heal_methods:
                if hm == "single_layer":
                    if inserted_idx is None:
                        # If user requests single_layer heal but used "none" replacement, bail
                        raise RuntimeError("single_layer heal requires a replacement block.")
                    ppl_after_single = self.heal_inserted_layer_trainer(
                        working, inserted_idx=inserted_idx,
                        train_dataloader=train_dataloader, eval_dataloader=eval_loader,
                        max_steps=single_max_steps, lr=single_lr, grad_accum=single_grad_accum,
                        log_dir=os.path.join(step_dir, "single_layer_logs"),
                    )
                elif hm == "qlora":
                    q = self.full_heal_qLoRA_trainer(
                        working, train_dataloader=train_dataloader, eval_dataloader=eval_loader,
                        max_steps=qlora_max_steps, lr=qlora_lr, grad_accum=qlora_grad_accum,
                        r=qlora_r, alpha=qlora_alpha, dropout=qlora_dropout,
                        target_modules=qlora_target_modules, save_dir=os.path.join(step_dir, "qlora_adapters"),
                    )
                    adapters_path = q.get("adapter_path", adapters_path)
                    ppl_after_qlora = q.get("ppl", None) or self.compute_perplexity(eval_loader, model=working)
                else:
                    raise ValueError(f"Unknown heal method: {hm}")

            # decide final ppl for acceptance
            final_ppl = ppl_after_qlora if ppl_after_qlora is not None else (
                ppl_after_single if ppl_after_single is not None else ppl_post_prune
            )

            # save meta & (optional) recipe/model snapshot
            meta = {
                "step": step,
                "percent": int(percent),
                "block_size": int(block_size),
                "selected_original": {
                    "start": int(sel_row["chosen_start_layer"]),
                    "end_incl": int(sel_row["chosen_end_layer_inclusive"]),
                },
                "mapped_current": {"start": int(s_cur), "end_incl": int(e_cur)},
                "replacement": replacement or "none",
                "heal_methods": list(heal_methods),
                "baseline_ppl": float(baseline_ppl),
                "ppl_limit": float(ppl_limit),
                "post_prune_ppl": float(ppl_post_prune),
                "post_single_layer_ppl": float(ppl_after_single) if ppl_after_single is not None else None,
                "post_qlora_ppl": float(ppl_after_qlora) if ppl_after_qlora is not None else None,
                "final_ppl": float(final_ppl),

                # step stats
                "layers_before": layers_before,
                "layers_removed_this_step": net_removed,
                # "layers_after_if_accepted" filled right below, once we know accepted/rejected

                "csv_path": csv_path,
            }

            with open(os.path.join(step_dir, "run.json"), "w") as f:
                json.dump(meta, f, indent=2)

            if save_variant_each_step:
                try:
                    if self._model_uses_bnb(working):
                        self._save_inserted_layer_recipe(working, inserted_idx,
                                                         os.path.join(step_dir, "preheal_recipe"), meta.copy())
                    else:
                        try:
                            working.to(self.torch_dtype)
                        except Exception:
                            pass
                        working.save_pretrained(os.path.join(step_dir, "model"))
                        with open(os.path.join(step_dir, "model", "meta.json"), "w") as f:
                            json.dump(meta, f, indent=2)
                except Exception as ex:
                    print(f"[WARN] save failed at step {step}: {ex}")

            # accept or rollback
            # accept or rollback
            if final_ppl <= ppl_limit:
                # accept
                self._save_checkpoint_for_rollback(working, last_ok_dir)
                last_ok_ppl = final_ppl

                # NEW — update global stats
                global_stats["layers_pruned_total"] += net_removed
                global_stats["layers_final"] = int(layers_before - net_removed)
                # Percent pruned relative to original L0
                global_stats["pruned_percent_total"] = 100.0 * (global_stats["layers_pruned_total"] / max(1, L0))
                global_stats["accepted_steps"] += 1

                # Augment meta and persist
                meta["status"] = "accepted"
                meta["layers_after_if_accepted"] = global_stats["layers_final"]
                meta["layers_pruned_total_so_far"] = global_stats["layers_pruned_total"]
                meta["pruned_percent_total_so_far"] = global_stats["pruned_percent_total"]

                # Print one-line human summary
                print(
                    f"[✓ ACCEPTED] step={step:03d}  p%={percent:02d}  bs={block_size:02d}  "
                    f"Δlayers=-{net_removed}  total_pruned={global_stats['layers_pruned_total']}/{L0} "
                    f"({global_stats['pruned_percent_total']:.1f}%)  ppl={final_ppl:.3f} ≤ limit={ppl_limit:.3f}"
                )

                # Save run.json with status & counters
                with open(os.path.join(step_dir, "run.json"), "w") as f:
                    json.dump(meta, f, indent=2)

                records.append({"status": "accepted", **meta, "dir": step_dir})
                step += 1
                # keep same percent

            else:
                # reject — rollback and halve percent
                print(f"[✗ REJECTED] step={step:03d}  p%={percent:02d}  bs={block_size:02d}  "
                      f"ppl={final_ppl:.3f} > limit={ppl_limit:.3f}  → rollback & halve percent")

                # NEW — rejection counters
                global_stats["rejected_steps"] += 1

                # Mark meta and save it too (so every step has a run.json)
                meta["status"] = "rejected"

                # layers_after_if_accepted stays absent on reject
                meta["layers_pruned_total_so_far"] = global_stats["layers_pruned_total"]
                meta["pruned_percent_total_so_far"] = global_stats["pruned_percent_total"]
                with open(os.path.join(step_dir, "run.json"), "w") as f:
                    json.dump(meta, f, indent=2)

                if save_variant_each_step:
                    # prefer the saved full model dir if present; else fall back to the step dir
                    maybe_model_dir = os.path.join(step_dir, "model")
                    last_rejected_dir = maybe_model_dir if os.path.isdir(maybe_model_dir) else step_dir
                    last_rejected_meta = meta.copy()

                # rollback
                del working
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                working = AutoModelForCausalLM.from_pretrained(
                    last_ok_dir, trust_remote_code=self.trust_remote_code
                ).to(self.device).eval()

                percent = max(1, percent // 2)
                records.append({"status": "rejected", **meta, "dir": step_dir})

            # loop continues; head will test min_block again

        # ---------- Optional final QLoRA rescue / optimization ----------
        final_qlora_result = None
        final_model_origin = None  # "rejected_rescued" | "accepted"
        final_model_dir = None

        if final_qlora:
            eval_loader = self._pick_eval_loader(train_dataloader, eval_dataloader)

            def _run_final_qlora_from_dir(src_dir: str, tag: str):
                # Load the source model, then run your trainer-based QLoRA
                m = AutoModelForCausalLM.from_pretrained(src_dir, trust_remote_code=self.trust_remote_code).to(
                    self.device).eval()
                out = self.full_heal_qLoRA_trainer(
                    m,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_loader,
                    max_steps=qlora_max_steps,
                    lr=qlora_lr,
                    r=qlora_r,
                    alpha=qlora_alpha,
                    dropout=qlora_dropout,
                    target_modules=qlora_target_modules,
                    grad_accum=qlora_grad_accum,
                    save_dir=os.path.join(prog_dir, f"final_qlora_{tag}_adapters"),
                )
                # Prefer merge-and-unload eval when available is already handled inside the trainer
                return out

            # 1) Try to rescue the last rejected model first (if requested and exists)
            if final_qlora_try_rejected_first and last_rejected_dir and os.path.isdir(last_rejected_dir):
                print("[final_qlora] Trying to rescue last rejected model...")
                res = _run_final_qlora_from_dir(last_rejected_dir, "rejected")
                final_qlora_result = res
                if res.get("ok") and res.get("ppl") is not None:
                    if res["ppl"] <= ppl_limit:
                        print(f"[final_qlora] Rescue succeeded: ppl={res['ppl']:.3f} ≤ limit={ppl_limit:.3f}")
                        final_model_origin = "rejected_rescued"
                        final_model_dir = last_rejected_dir
                    else:
                        print(
                            f"[final_qlora] Rescue failed to meet limit: ppl={res['ppl']:.3f} > limit={ppl_limit:.3f}")

            # 2) If rescue didn’t happen or didn’t pass, run QLoRA on the last accepted model
            if final_model_origin is None:
                print("[final_qlora] Running on last accepted model...")
                res = _run_final_qlora_from_dir(last_ok_dir, "accepted")
                final_qlora_result = res
                final_model_origin = "accepted"
                final_model_dir = last_ok_dir

            merged_dir = final_qlora_result.get("merged_dir")
            if merged_dir and os.path.isdir(merged_dir):
                final_model_dir = merged_dir

            # ensure tokenizer is present next to the final model
            if final_model_dir:
                try:
                    self.tokenizer.save_pretrained(final_model_dir)
                except Exception:
                    print("Couldn't save the Tokenizer")
                    pass

            # Persist a tiny summary for the final QLoRA outcome
            with open(os.path.join(prog_dir, "final_qlora_summary.json"), "w") as f:
                json.dump({
                    "origin": final_model_origin,
                    "adapters_path": final_qlora_result.get("adapter_path"),
                    "final_model_dir": final_model_dir,
                    "ppl": final_qlora_result.get("ppl"),
                    "ok": final_qlora_result.get("ok", False),
                }, f, indent=2)

            # If the rescued rejected model met the limit, make that the “final” PPL
            if final_model_origin == "rejected_rescued" and final_qlora_result.get("ppl") is not None:
                last_ok_ppl = float(final_qlora_result["ppl"])

        # write a final manifest
        manifest = {
            "ok": True,
            "baseline_ppl": float(baseline_ppl),
            "ppl_limit": float(ppl_limit),
            "final_ppl": float(last_ok_ppl),
            "layers_start": int(global_stats["layers_start"]),
            "layers_final": int(global_stats["layers_final"]),
            "layers_pruned_total": int(global_stats["layers_pruned_total"]),
            "pruned_percent_total": float(global_stats["pruned_percent_total"]),
            "accepted_steps": int(global_stats["accepted_steps"]),
            "rejected_steps": int(global_stats["rejected_steps"]),
            "steps": records,
            "experiment_dir": prog_dir,
            "final_qlora": bool(final_qlora),
            "final_qlora_ppl": (final_qlora_result or {}).get("ppl"),
            "final_qlora_origin": final_model_origin,  # "rejected_rescued" or "accepted" or None
            "final_qlora_adapters": (final_qlora_result or {}).get("adapter_path"),
            "final_model_dir": final_model_dir,
        }
        with open(os.path.join(prog_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"[DONE] layers: {manifest['layers_start']} → {manifest['layers_final']}  "
            f"(pruned {manifest['layers_pruned_total']} = {manifest['pruned_percent_total']:.1f}% of start)  "
            f"accepted={manifest['accepted_steps']}  rejected={manifest['rejected_steps']}  "
            f"final_ppl={manifest['final_ppl']:.3f}  limit={manifest['ppl_limit']:.3f}"
        )
        if final_qlora and final_qlora_result and final_qlora_result.get("ppl") is not None:
            print(f"[FINAL QLoRA] origin={final_model_origin}  ppl={final_qlora_result['ppl']:.3f}")

        return manifest

    # --- Using Trainer ---
    def _freeze_all_but(self, model, module):
        for p in model.parameters():
            p.requires_grad_(False)
        for p in module.parameters():
            p.requires_grad_(True)

    def _masking_collator(self):
        """
        Returns a collator that:
          - preserves input_ids/attention_mask from your DataLoader batches
          - creates labels with pads masked to -100 (like your current code)
        """

        def collate(features):
            # features already dicts with 'input_ids', 'attention_mask' tensors
            input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
            attention_mask = torch.stack([f["attention_mask"] for f in features], dim=0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        return collate

    def make_trainer(
            self,
            model,
            train_dataset,
            eval_dataset,
            data_collator,
            *,
            output_dir: str,
            learning_rate: float,
            max_steps: int,
            gradient_accumulation_steps: int,
            per_device_train_batch_size: int,
            per_device_eval_batch_size: int,
            logging_steps: int = 10,
            save_steps: int = 0,
            eval_steps: int = 0,
            fp16: bool = False,
            bf16: bool = True,
    ):
        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=max_steps,  # number of *optimizer* steps
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            remove_unused_columns=False,  # don't drop our fields
            label_names=["labels"],
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            fp16=fp16,
            bf16=bf16,
            report_to=[],
        )
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    def heal_inserted_layer_trainer(
            self,
            model,
            inserted_idx: int,
            train_dataloader,
            eval_dataloader=None,
            max_steps: int = 500,
            lr: float = 2e-4,
            grad_accum: int = 1,
            log_dir: Optional[str] = None,
    ) -> float:
        """
        Trainer-based single-layer heal, now using Builder-native dataset+collator:
          - derive truthful per-device batch sizes from incoming loaders
          - tokenize/pad/label exactly once in collator (no unbatch/rebatch)
          - add step accounting to verify max_steps × grad_accum behavior
        """
        from transformers import Trainer, TrainingArguments

        # 1) Freeze everything except the inserted layer (unchanged)
        parent, layers = self._get_layer_container_and_list(model)
        train_layer = layers[inserted_idx]
        comp_dtype = self._infer_compute_dtype(model)
        dev = next(model.parameters()).device
        train_layer.to(device=dev, dtype=comp_dtype)
        for p in model.parameters(): p.requires_grad_(False)
        for p in train_layer.parameters(): p.requires_grad_(True)

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.train()

        # 2) Build Trainer-native dataset+collator from the *texts* behind your loaders
        train_texts = _extract_texts_from_loader(train_dataloader)
        eval_texts = _extract_texts_from_loader(eval_dataloader) if eval_dataloader is not None else None
        train_ds = TextListMapDataset(train_texts)
        eval_ds = TextListMapDataset(eval_texts) if eval_texts is not None else None

        # Respect the batch sizes you used when you built the loaders (truthful values)
        per_dev_train_bs = int(getattr(train_dataloader, "batch_size", 1) or 1)
        per_dev_eval_bs = int(
            getattr(eval_dataloader, "batch_size", 1) or 1) if eval_dataloader is not None else per_dev_train_bs

        # Single source of truth for padding and labels (pad-to-longest, labels=-100 on pad)
        max_len = int(self.max_eval_seq_len or 512)
        collator = collate_with_labels(self.tokenizer, max_len)
        # base_collator = collate_with_labels(self.tokenizer, max_len)
        # acct = {"micro_batches": 0, "tokens": 0}
        # collator = _CountingCollator(base_collator, acct)

        # 3) Trainer args (no fake batch size)
        out_dir = log_dir or os.path.join(self.results_dir, "trainer_heal_tmp")
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=per_dev_train_bs,
            per_device_eval_batch_size=per_dev_eval_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            num_train_epochs=1,  # we drive with max_steps
            max_steps=max_steps,  # counts *optimizer* steps
            logging_steps=max(1, max_steps // 20),
            eval_strategy="no",
            save_strategy="no",
            remove_unused_columns=False,
            report_to=[],
            fp16=(comp_dtype == torch.float16),
            bf16=(comp_dtype == torch.bfloat16),
            dataloader_pin_memory=True,
            label_names=["labels"],
        )

        # 4) Optimizer over the single trainable layer
        opt = torch.optim.AdamW([p for p in train_layer.parameters() if p.requires_grad], lr=lr)
        # 5) Batch accounting
        # acct_callback = StepAccountingCallback(grad_accum=grad_accum)

        # 5) Trainer with step accounting
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            optimizers=(opt, None),
            # callbacks=[acct_callback],
        )

        trainer.train()

        # reliable accounting
        # opt_steps = int(getattr(trainer.state, "global_step", 0))  # optimizer steps
        # mb = int(acct["micro_batches"])
        # ga = int(args.gradient_accumulation_steps)
        # ratio = (mb / opt_steps) if opt_steps else 0.0
        # print(f"[Accounting] optimizer_steps={opt_steps}  micro_batches={mb}  grad_accum={ga}  (~{ratio:.2f} batches/opt-step)")

        # 6) Evaluate perplexity via your existing path
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        eval_loader = self._pick_eval_loader(train_dataloader, eval_dataloader)
        ppl = self.compute_perplexity(eval_loader, model=model)

        # Re-freeze trained layer afterwards
        for p in train_layer.parameters(): p.requires_grad_(False)
        return ppl

    def full_heal_qLoRA_trainer(
            self,
            model,
            train_dataloader,
            eval_dataloader=None,
            max_steps: int = 1000,
            lr: float = 1e-4,
            r: int = 16,
            alpha: int = 32,
            dropout: float = 0.05,
            target_modules: Optional[List[str]] = None,
            grad_accum: int = 1,
            save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Trainer-based QLoRA using Builder-native dataset+collator (no unbatch/rebatch).
        Saves adapters (if save_dir), computes PPL; fast merge-and-eval when safe.
        """
        out = {"ok": False, "error": None, "adapter_path": None, "ppl": None}
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import Trainer, TrainingArguments
        except Exception as e:
            out["error"] = f"Missing deps (peft/transformers): {e}"
            return out

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "w1", "w2", "w3", "dense_h_to_4h", "dense_4h_to_h"
            ]

        # 1) Training prep for k-bit
        model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        lconf = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lconf)

        comp_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                      else torch.float16)
        for _, p in model.named_parameters():
            if p.requires_grad and p.is_floating_point() and p.dtype != comp_dtype:
                p.data = p.data.to(comp_dtype)

        # 2) Builder-native dataset + truthful batch sizes
        train_texts = _extract_texts_from_loader(train_dataloader)
        eval_texts = _extract_texts_from_loader(eval_dataloader) if eval_dataloader is not None else None
        train_ds = TextListMapDataset(train_texts)
        eval_ds = TextListMapDataset(eval_texts) if eval_texts is not None else None

        per_dev_train_bs = int(getattr(train_dataloader, "batch_size", 1) or 1)
        per_dev_eval_bs = int(
            getattr(eval_dataloader, "batch_size", 1) or 1) if eval_dataloader is not None else per_dev_train_bs

        max_len = int(self.max_eval_seq_len or 512)
        collator = collate_with_labels(self.tokenizer, max_len)
        # base_collator = collate_with_labels(self.tokenizer, max_len)
        # acct = {"micro_batches": 0, "tokens": 0}
        # collator = _CountingCollator(base_collator, acct)

        # 3) Trainer args (no hacks)
        out_dir = save_dir or os.path.join(self.results_dir, "qlora_adapters")
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=per_dev_train_bs,
            per_device_eval_batch_size=per_dev_eval_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            max_steps=max_steps,  # optimizer steps
            lr_scheduler_type="cosine",
            warmup_ratio=0.10,
            logging_steps=max(1, max_steps // 20),
            save_strategy="no",  # adapters saved manually after train
            remove_unused_columns=False,
            report_to=[],
            fp16=(comp_dtype == torch.float16),
            bf16=(comp_dtype == torch.bfloat16),
            dataloader_pin_memory=True,
            eval_strategy="no",
            label_names=["labels"],
        )

        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        # acct_callback = StepAccountingCallback(grad_accum=grad_accum)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            optimizers=(opt, None),
            # callbacks=[acct_callback],
        )

        trainer.train()

        # reliable accounting
        # opt_steps = int(getattr(trainer.state, "global_step", 0))  # optimizer steps
        # mb = int(acct["micro_batches"])
        # ga = int(args.gradient_accumulation_steps)
        # ratio = (mb / opt_steps) if opt_steps else 0.0
        # print(f"[Accounting] optimizer_steps={opt_steps}  micro_batches={mb}  grad_accum={ga}  (~{ratio:.2f} batches/opt-step)")

        # 4) Save adapters before any merge
        merged_dir = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            try:
                model.save_pretrained(save_dir)
                out["adapter_path"] = save_dir
            except Exception as e:
                out["error"] = f"Failed to save adapters: {e}"
                return out

        # Choose a target dtype for the merged weights: prefer bf16 on capable systems, else fp16
        preferred_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                           else torch.float16)

        def _set_config_dtype(m, dt):
            # ensure config carries the intended dtype so config.json won't say float32
            try:
                m.config.torch_dtype = dt
            except Exception:
                pass  # non-fatal

        def _cast_module_inplace(m, dt):
            # cast all float params/buffers in-place (safe on CPU)
            for p in m.parameters(recurse=True):
                if p.is_floating_point() and p.dtype != dt:
                    p.data = p.data.to(dt)
            for n, b in m.named_buffers(recurse=True):
                if b.is_floating_point() and b.dtype != dt:
                    setattr(m, n, b.to(dt))

        # 5) Build a merged full model on disk, and force-save in preferred_dtype
        try:
            base_dir_hint = None
            try:
                base_dir_hint = getattr(getattr(model, "config", None), "_name_or_path", None)
            except Exception:
                pass

            if save_dir:
                merged_dir = save_dir.rstrip(os.sep) + "_merged"
                os.makedirs(merged_dir, exist_ok=True)

            is_4bit_base = getattr(getattr(model, "base_model", model), "is_loaded_in_4bit", False)
            merged_obj = None

            if hasattr(model, "merge_and_unload") and not is_4bit_base:
                # Fast path: we can merge directly
                merged_obj = model.merge_and_unload()
            else:
                # Safe compose-then-merge path for 4-bit bases (or when merge isn't available)
                if base_dir_hint is None:
                    # Fallback guess (parent of adapters dir) if we saved adapters
                    base_dir_hint = os.path.dirname(save_dir) if save_dir else None
                if base_dir_hint is None:
                    raise RuntimeError("Cannot determine base directory for composing LoRA adapters.")

                # Reload the float base, compose adapters, then merge
                base_fp = AutoModelForCausalLM.from_pretrained(
                    base_dir_hint,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=preferred_dtype,
                ).to("cpu").eval()

                from peft import PeftModel
                composed = PeftModel.from_pretrained(base_fp, save_dir)
                if hasattr(composed, "merge_and_unload"):
                    merged_obj = composed.merge_and_unload()
                else:
                    merged_obj = composed  # rare fallback

            # Force weights & config to preferred dtype (even on CPU)
            try:
                merged_obj.to("cpu")
            except Exception:
                pass
            _cast_module_inplace(merged_obj, preferred_dtype)
            _set_config_dtype(merged_obj, preferred_dtype)

            # Save merged model + tokenizer
            merged_obj.save_pretrained(merged_dir)
            try:
                self.tokenizer.save_pretrained(merged_dir)
            except Exception:
                pass

            out["merged_dir"] = merged_dir

        except Exception as e:
            out["error"] = f"Adapter merge/save issue: {e}"

        # 6) Evaluate (prefer merged model when available)
        eval_loader = eval_dataloader or train_dataloader
        if eval_loader is not None:
            eval_model = None
            if out.get("merged_dir"):
                try:
                    eval_model = AutoModelForCausalLM.from_pretrained(
                        out["merged_dir"], trust_remote_code=self.trust_remote_code
                    ).to(self.device).eval()
                except Exception:
                    eval_model = None

            if eval_model is None:
                # Fall back to in-memory model (PEFT-wrapped)
                eval_model = model
                try:
                    eval_model.eval()
                    if hasattr(eval_model, "gradient_checkpointing_disable"):
                        eval_model.gradient_checkpointing_disable()
                    for p in eval_model.parameters():
                        p.requires_grad_(False)
                    if hasattr(eval_model.config, "use_cache"):
                        eval_model.config.use_cache = True
                except Exception:
                    pass

            out["ppl"] = self.compute_perplexity(eval_loader, model=eval_model)

        out["ok"] = True
        return out

