import argparse
import os
import pathlib
import inspect
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# --- Small wrapper to normalize AMPLIFY model API differences ---
class AmplifyForwardWrapper(torch.nn.Module):
    """Unifies forward arg names across AMPLIFY variants (src vs input_ids).

    - We only pass token ids (no attention mask) since batches are rectangular
    - Exposes a `get_logits(input_ids)` method returning [B, T, V] tensor.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        # Try to ensure non-packed inference
        try:
            if hasattr(self.model, "pack_sequences"):
                self.model.pack_sequences = False
            if hasattr(getattr(self.model, "module", None), "pack_sequences"):
                self.model.module.pack_sequences = False
            if hasattr(getattr(self.model, "config", None), "pack_sequences"):
                self.model.config.pack_sequences = False
        except Exception:
            pass
        # Inspect forward signature once to decide the expected arg names.
        try:
            sig = inspect.signature(model.forward)
        except (ValueError, AttributeError):
            sig = inspect.signature(model.__call__)

        params = set(sig.parameters.keys())
        if "src" in params and "input_ids" not in params:
            self._ids_kw = "src"
        else:
            self._ids_kw = "input_ids"

        # Discover optional flags we can propagate
        self._supports_hidden = "output_hidden_states" in params
        self._supports_attn = "output_attentions" in params
        self._supports_pos = "position_ids" in params
        self._supports_attn_mask = "attention_mask" in params

        print(
            f"AmplifyForwardWrapper defined with ids_kw='{self._ids_kw}', "
            f"supports_hidden={self._supports_hidden}, supports_attn={self._supports_attn}, "
            f"supports_pos={self._supports_pos}, supports_attn_mask={self._supports_attn_mask}"
            f" for model {type(model).__name__}"
            f"Base model API: {sig}"
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        kwargs = {self._ids_kw: input_ids}
        if self._supports_hidden:
            kwargs["output_hidden_states"] = output_hidden_states
        if self._supports_attn:
            kwargs["output_attentions"] = output_attentions
        if self._supports_pos and position_ids is not None:
            kwargs["position_ids"] = position_ids
        if self._supports_attn_mask and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        return self.model(**kwargs)

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.forward(
            input_ids,
            output_hidden_states=False,
            output_attentions=False,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        if hasattr(out, "logits"):
            out = out.logits
        if isinstance(out, dict) and "logits" in out:
            out = out["logits"]
        if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            out = out[0]
        return out


def create_parser():
    parser = argparse.ArgumentParser(description="Compute mutation fitness using Hugging Face AMPLIFY models.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for masked-marginals computation (default: 16)",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for autocast computations: bf16 (default), fp16, or fp32.",
    )
    # removed --model-dtype; rely on autocast only
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more HF model identifiers (for hub). For local, ignore this and provide "
            "--local-checkpoint (.pt) and --local-config (config.yaml)."
        ),
    )
    parser.add_argument(
        "--load-from",
        type=str,
        choices=["auto", "hub", "local"],
        default="auto",
        help=(
            "Where to load models from: auto (default), hub, or local. For local, provide --local-checkpoint (.pt) "
            "and --local-config (config.yaml)."
        ),
    )
    parser.add_argument("--local-checkpoint", type=str, default=None, help="Path to local .pt checkpoint file")
    parser.add_argument("--local-config", type=str, default=None, help="Path to local Hydra config.yaml file")
    parser.add_argument(
        "--use_auth_token",
        type=str,
        default=None,
        help=("Hugging Face auth token."),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision (branch, tag, or commit) to load models/tokenizers from.",
    )
    parser.add_argument("--sequence", type=str, help="Base WT sequence (overridden by --dms_index mapping if set)")
    parser.add_argument("--target_seq", type=str, default=None, help="Alias for --sequence when not using mapping")
    parser.add_argument("--dms-input", type=pathlib.Path, required=True, help="CSV file or folder (with mapping)")
    parser.add_argument("--dms-output", type=pathlib.Path, required=True, help="Output folder for scored CSV")
    parser.add_argument("--mutation-col", type=str, default="mutant", help="Column with mutation strings")
    parser.add_argument(
        "--dms_index",
        type=str,
        nargs="+",
        default=None,
        help="Indices of DMS in mapping file, or 'all'",
    )
    parser.add_argument("--dms_mapping", type=str, default=None, help="CSV with mapping rows when using --dms_index")
    parser.add_argument("--offset-idx", type=int, default=1, help="Offset of mutation positions in mutation column")
    parser.add_argument(
        "--bos-offset",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Index offset to account for a leading BOS token and trailing EOS in tokenized inputs. "
            "Use 1 when tokenizer adds BOS/EOS (default), 0 when no special tokens are added."
        ),
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "masked-marginals", "pseudo-ppl"],
        help="Scoring strategy using only sequences",
    )
    parser.add_argument(
        "--scoring-window",
        type=str,
        default="optimal",
        choices=["full", "optimal"],
        help="If sequences exceed max length, use an optimal local window around positions",
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument("--overwrite-prior-scores", action="store_true", help="Overwrite existing columns if present")
    return parser


def _window_bounds(center: int, seq_len: int, window_size: int) -> Tuple[int, int]:
    """Simple symmetric window [start, end) around center token index (0-based over raw sequence)."""
    if window_size >= seq_len:
        return 0, seq_len
    half = window_size // 2
    start = max(0, center - half)
    end = start + window_size
    if end > seq_len:
        end = seq_len
        start = max(0, end - window_size)
    return start, end


def _device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_hf_model_and_tokenizer(
    model_name: str, device: torch.device, revision: Optional[str] = None, auth_token: Optional[str] = None
):
    from transformers import AutoTokenizer, AutoModel

    if auth_token == "":
        auth_token = None
    # AMPLIFY models on HF expose a compatible masked LM head for token-level scoring
    kwargs = {"trust_remote_code": True}
    if revision is not None:
        kwargs["revision"] = revision
    if auth_token is not None:
        kwargs["token"] = auth_token

    print(f"Loading HF model {model_name} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(device)
    return tokenizer, model


def _load_local_amplify_model_and_tokenizer(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
):
    """Load an AMPLIFY model/tokenizer from explicit local paths.

    Args:
        checkpoint_path: Path to a .pt checkpoint file.
        config_path: Path to a Hydra config.yaml file.
        device: torch device.
    """
    try:
        from amplify.model import AMPLIFY  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import amplify.model.AMPLIFY. Ensure AMPLIFY is installed (pip install -e .) or on PYTHONPATH."
        ) from e

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Local AMPLIFY checkpoint (.pt) not found at: {checkpoint_path}")
    if not checkpoint_path.endswith(".pt"):
        raise ValueError(f"--local-checkpoint must point to a .pt file, got: {checkpoint_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Local AMPLIFY config.yaml not found at: {config_path}")

    print(f"Loading local AMPLIFY model from checkpoint {checkpoint_path} with config {config_path} on device {device}")
    model, tokenizer = AMPLIFY.load(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def _char_to_id(tokenizer, aa: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(aa)
    if tid is None or tid == tokenizer.unk_token_id:
        raise ValueError(f"Unknown residue token '{aa}' for tokenizer {type(tokenizer).__name__}")
    return tid


def _tokenize_seq(tokenizer, seq: str, device: torch.device, add_special_tokens: bool = True):
    """Tokenize a single sequence. We only use input_ids and drop attention masks. Ensures batch dim."""
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = enc["input_ids"].to(device)
    # Be robust: ensure shape [B, T]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return {"input_ids": input_ids}


# Removed special-token alignment; we assume a single BOS at index 0 and EOS at the end.


def _log_softmax_logits(model_wrapper: AmplifyForwardWrapper, inputs, autocast_dtype: str = "bf16") -> torch.Tensor:
    # Use autocast for computation (default bf16, fp16 if weights are fp16)
    autocast_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    autocast_t = autocast_map.get(autocast_dtype, torch.bfloat16)
    with torch.no_grad():
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=autocast_t, enabled=True):
            input_ids = inputs["input_ids"]
            # Build per-batch position_ids to satisfy RoPE shape (B, T)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            B, T = input_ids.shape[0], input_ids.shape[1]
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            logits = model_wrapper.get_logits(input_ids, position_ids=pos)  # [B, T, V]
            return torch.log_softmax(logits, dim=-1)


def _label_row_hf(
    row_mut_str: str,
    sequence: str,
    token_log_probs: torch.Tensor,
    tokenizer,
    offset_idx: int,
    bos_offset: int,
) -> float:
    # token_log_probs expected shape: [1, T, V] with special tokens at positions 0 and T-1
    score = 0.0
    for mutation in row_mut_str.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_id = _char_to_id(tokenizer, wt)
        mt_id = _char_to_id(tokenizer, mt)
        # Offset by BOS when present
        score += (token_log_probs[0, bos_offset + idx, mt_id] - token_log_probs[0, bos_offset + idx, wt_id]).item()
    return float(score)


def _compute_pseudo_ppl(
    seq: str,
    tokenizer,
    model_wrapper: AmplifyForwardWrapper,
    device: torch.device,
    autocast_dtype: str = "bf16",
    dms_idx=None,
    bos_offset: int = 1,
) -> float:
    # Sum of log p(x_i | x_\i) by masking one token at a time
    # Prefer model config if available, else tokenizer limit
    max_len_cfg = getattr(getattr(model_wrapper, "model", None), "config", None)
    max_len = getattr(max_len_cfg, "max_length", None) or getattr(tokenizer, "model_max_length", 2048) or 2048
    # Reserve space for BOS/EOS when present
    reserved_special = 2 * int(bos_offset == 1)
    window_cap = max_len - reserved_special if max_len and max_len > reserved_special else len(seq)

    log_probs = []
    bar_desc = f"pppl DMS={dms_idx}" if dms_idx is not None else "pppl"
    for pos in tqdm(range(len(seq)), desc=bar_desc):
        if window_cap < len(seq):
            start, end = _window_bounds(pos, len(seq), window_cap)
            subseq = seq[start:end]
            local_pos = pos - start
            inputs = _tokenize_seq(tokenizer, subseq, device, add_special_tokens=(bos_offset == 1))
            # Mask position (offset for BOS if present)
            inputs["input_ids"][0, bos_offset + local_pos] = tokenizer.mask_token_id
        else:
            inputs = _tokenize_seq(tokenizer, seq, device, add_special_tokens=(bos_offset == 1))
            inputs["input_ids"][0, bos_offset + pos] = tokenizer.mask_token_id

        logp = _log_softmax_logits(model_wrapper, inputs, autocast_dtype=autocast_dtype)
        aa_id = _char_to_id(tokenizer, seq[pos])
        if window_cap < len(seq):
            log_probs.append(logp[0, bos_offset + local_pos, aa_id].item())
        else:
            log_probs.append(logp[0, bos_offset + pos, aa_id].item())
    return float(np.sum(log_probs))


def _masked_marginals(
    sequence: str,
    tokenizer,
    model_wrapper: AmplifyForwardWrapper,
    device: torch.device,
    scoring_window: str,
    batch_size: int = 16,
    autocast_dtype: str = "bf16",
    dms_idx=None,
    bos_offset: int = 1,
) -> torch.Tensor:
    # For each position, mask it, run model, and collect the full vocab log-prob vector.
    # Prefer model config if available, else tokenizer limit
    _cfg = getattr(getattr(model_wrapper, "model", None), "config", None)
    max_len = getattr(_cfg, "max_length", None) or getattr(tokenizer, "model_max_length", 2048) or 2048
    reserved_special = 2 * int(bos_offset == 1)
    window_cap = max_len - reserved_special if max_len and max_len > reserved_special else len(sequence)
    use_windows = scoring_window == "optimal" and window_cap < len(sequence)
    all_rows = []  # list of [V] tensors per position
    bar_desc = f"mm DMS={dms_idx}" if dms_idx is not None else "mm"
    positions = list(range(len(sequence)))
    for batch_start in tqdm(range(0, len(positions), batch_size), desc=bar_desc, disable=True):
        batch_positions = positions[batch_start : batch_start + batch_size]
        input_ids = []
        for pos in batch_positions:
            if use_windows:
                start, end = _window_bounds(pos, len(sequence), window_cap)
                subseq = sequence[start:end]
                local_pos = pos - start
                enc = tokenizer(subseq, return_tensors="pt", add_special_tokens=(bos_offset == 1))
                ids = enc["input_ids"].to(device)
                # Ensure ids has a batch dimension for consistent indexing
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                ids[0, bos_offset + local_pos] = tokenizer.mask_token_id
                input_ids.append(ids[0])
            else:
                enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=(bos_offset == 1))
                ids = enc["input_ids"].to(device)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                ids[0, bos_offset + pos] = tokenizer.mask_token_id
                input_ids.append(ids[0])
        # Stack batch
        input_ids = torch.stack(input_ids, dim=0)
        batch_inputs = {"input_ids": input_ids}
        logp = _log_softmax_logits(model_wrapper, batch_inputs, autocast_dtype=autocast_dtype)
        # logp shape: [batch, T, V]
        for i, pos in enumerate(batch_positions):
            if use_windows:
                start, end = _window_bounds(pos, len(sequence), window_cap)
                local_pos = pos - start
                all_rows.append(logp[i, bos_offset + local_pos].detach().cpu())
            else:
                all_rows.append(logp[i, bos_offset + pos].detach().cpu())
    # Shape [1, L + reserved_special, V]
    token_probs = torch.zeros((1, len(sequence) + reserved_special, all_rows[0].size(-1)))
    for i, row in enumerate(all_rows):
        token_probs[0, bos_offset + i] = row
    return token_probs


def main(args):
    if not os.path.exists(args.dms_output):
        os.mkdir(args.dms_output)

    # Resolve device
    device = _device(use_gpu=not args.nogpu)

    # Load single model once
    if args.load_from == "local":
        if not args.local_checkpoint or not args.local_config:
            raise ValueError(
                "For local loading, provide both --local-checkpoint (.pt) and --local-config (config.yaml)"
            )
        tokenizer, model = _load_local_amplify_model_and_tokenizer(
            checkpoint_path=args.local_checkpoint, config_path=args.local_config, device=device
        )
        # Move to device only; avoid global dtype cast to prevent complex->real warnings
        model.to(device)
        model_wrapper = AmplifyForwardWrapper(model)
        model_label = os.path.basename(args.local_checkpoint)
    else:
        # Use the first provided model if a list is passed
        model_name = (
            args.model_name_or_path[0] if isinstance(args.model_name_or_path, list) else args.model_name_or_path
        )
        tokenizer, model = _load_hf_model_and_tokenizer(
            model_name, device, revision=args.revision, auth_token=args.use_auth_token
        )
        # Move to device only; avoid global dtype cast to prevent complex->real warnings
        model.to(device)
        model_wrapper = AmplifyForwardWrapper(model)
        model_label = model_name

    # Handle DMS indices (single, multiple, or 'all')
    if args.dms_index is None:
        dms_indices = [None]
    elif any(str(x).lower() == "all" for x in args.dms_index):
        if args.dms_mapping is None:
            raise ValueError("--dms_mapping is required when using --dms_index all")
        mapping = pd.read_csv(args.dms_mapping)
        dms_indices = list(range(len(mapping)))
    else:
        # Convert all entries to int
        dms_indices = [int(x) for x in args.dms_index]

    for dms_idx in tqdm(dms_indices, desc="DMS"):
        mutant_col = args.mutation_col
        if dms_idx is not None:
            if args.dms_mapping is None:
                raise ValueError("--dms_mapping is required when using --dms_index")
            mapping = pd.read_csv(args.dms_mapping)
            DMS_id = mapping["DMS_id"][dms_idx]
            row = mapping[mapping["DMS_id"] == DMS_id]
            if len(row) != 1:
                raise ValueError(f"Expected exactly one mapping row for DMS_id={DMS_id}, got {len(row)}")
            row = row.iloc[0].replace(np.nan, "")

            sequence = row["target_seq"].upper()
            dms_input_path = str(args.dms_input) + os.sep + row["DMS_filename"]
            mutant_col = (
                row["DMS_mutant_column"]
                if "DMS_mutant_column" in mapping.columns and row["DMS_mutant_column"]
                else mutant_col
            )
            dms_output_file = str(args.dms_output) + os.sep + DMS_id + ".csv"

            target_seq_start_index = (
                row["start_idx"] if "start_idx" in mapping.columns and row["start_idx"] != "" else 1
            )
            offset_idx = int(target_seq_start_index)
        else:
            if args.sequence is None and args.target_seq is None:
                raise ValueError("Provide --sequence/--target_seq or use --dms_index with --dms_mapping")
            sequence = (args.sequence or args.target_seq).upper()
            dms_input_path = str(args.dms_input)
            DMS_id = pathlib.Path(dms_input_path).name.replace(".csv", "")
            dms_output_file = str(args.dms_output) + os.sep + DMS_id + ".csv"
            offset_idx = args.offset_idx

        df = pd.read_csv(dms_input_path)
        if len(df) == 0:
            raise ValueError("No rows found in the dataframe")

        # Score with the single model and append column
        col_name = os.path.basename(str(model_label).rstrip("/"))

        # Skip if already computed and not overwriting
        if os.path.exists(dms_output_file) and not args.overwrite_prior_scores:
            prior_df = pd.read_csv(dms_output_file)
            if col_name in prior_df.columns:
                df = prior_df
                print(f"Skipping {col_name} (already present). Use --overwrite-prior-scores to recompute.")
            else:
                # Fall through to compute and merge below
                pass

        if col_name not in df.columns:
            if args.scoring_strategy == "wt-marginals":
                raise NotImplementedError("Not sure of the implemenation here")
                inputs = _tokenize_seq(tokenizer, sequence, device, add_special_tokens=(args.bos_offset == 1))
                _cfg = getattr(getattr(model_wrapper, "model", None), "config", None)
                max_len = getattr(_cfg, "max_length", None) or getattr(tokenizer, "model_max_length", 2048) or 2048
                bar_desc = f"wtm DMS={dms_idx}" if dms_idx is not None else "wtm"
                if inputs["input_ids"].shape[1] > max_len and args.scoring_window == "optimal":
                    token_probs = _masked_marginals(
                        sequence,
                        tokenizer,
                        model_wrapper,
                        device,
                        scoring_window="optimal",
                        batch_size=args.batch_size,
                        autocast_dtype=args.autocast_dtype,
                        dms_idx=dms_idx,
                        bos_offset=args.bos_offset,
                    )
                else:
                    for pos in tqdm([0], desc=bar_desc):
                        token_probs = (
                            _log_softmax_logits(model_wrapper, inputs, autocast_dtype=args.autocast_dtype)
                            .detach()
                            .cpu()
                        )

                df[col_name] = df.apply(
                    lambda row: _label_row_hf(
                        row[mutant_col], sequence, token_probs, tokenizer, offset_idx, args.bos_offset
                    ),
                    axis=1,
                )

            elif args.scoring_strategy == "masked-marginals":
                token_probs = _masked_marginals(
                    sequence,
                    tokenizer,
                    model_wrapper,
                    device,
                    args.scoring_window,
                    batch_size=args.batch_size,
                    autocast_dtype=args.autocast_dtype,
                    dms_idx=dms_idx,
                    bos_offset=args.bos_offset,
                )
                df[col_name] = df.apply(
                    lambda row: _label_row_hf(
                        row[mutant_col], sequence, token_probs, tokenizer, offset_idx, args.bos_offset
                    ),
                    axis=1,
                )

            elif args.scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                if "mutated_sequence" not in df:
                    df["mutated_sequence"] = df.progress_apply(
                        lambda row: _get_mutated_sequence(row[mutant_col], sequence, offset_idx), axis=1
                    )
                df[col_name] = df.progress_apply(
                    lambda row: _compute_pseudo_ppl(
                        row["mutated_sequence"],
                        tokenizer,
                        model_wrapper,
                        device,
                        autocast_dtype=args.autocast_dtype,
                        dms_idx=dms_idx,
                        bos_offset=args.bos_offset,
                    ),
                    axis=1,
                )

        # Write out incrementally if prior file exists
        if os.path.exists(dms_output_file) and not args.overwrite_prior_scores:
            prior_df = pd.read_csv(dms_output_file)
            if col_name not in prior_df.columns:
                prior_df = prior_df.merge(df[[col_name, mutant_col]], on=mutant_col)
            prior_df.to_csv(dms_output_file, index=False)
            df = prior_df
        else:
            df.to_csv(dms_output_file, index=False)


def _get_mutated_sequence(row: str, wt_sequence: str, offset_idx: int) -> str:
    seq = list(wt_sequence)
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert seq[idx] == wt, "The listed wildtype does not match the provided sequence"
        seq[idx] = mt
    return "".join(seq)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
