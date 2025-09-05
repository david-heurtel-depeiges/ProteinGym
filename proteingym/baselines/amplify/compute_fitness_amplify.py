import argparse
import os
import pathlib
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Minimal, sequence-only fitness computation using Hugging Face AMPLIFY models.
# Mirrors compute_fitness_plm for ESM, adapted for AMPLIFY tokenization and API.


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
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Data type for model weights: fp32 (default), bf16, or fp16. Computations autocasted to bf16 by default.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        nargs="+",
        required=True,
        help="One or more HF model identifiers or local paths (e.g., chandar-lab/AMPLIFY_120M)",
    )
    parser.add_argument(
        "--use_auth_token",
        type=str,
        default=None,
        help=(
            "Hugging Face authentication token (string) or boolean flag passed to from_pretrained() methods. "
            "Useful for accessing private models. If a string is provided, it is used as the token. "
            "If None, no token is used. If set to 'True' (string), the token from the local Hugging Face CLI login is used."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face revision (branch, tag, or commit) to load models/tokenizers from. "
            "Useful when pinning a specific branch like 'main' or a commit SHA."
        ),
    )
    parser.add_argument("--sequence", type=str, help="Base WT sequence (overridden by --dms_index mapping if set)")
    parser.add_argument("--target_seq", type=str, default=None, help="Alias for --sequence when not using mapping")
    parser.add_argument("--dms-input", type=pathlib.Path, required=True, help="CSV file or folder (with mapping)")
    parser.add_argument("--dms-output", type=pathlib.Path, required=True, help="Output folder for scored CSV")
    parser.add_argument(
        "--mutation-col", type=str, default="mutant", help="Column with mutations like 'A42D' or 'A42D:R79G'"
    )

    # Mapping options (optional)
    parser.add_argument(
        "--dms_index",
        type=str,
        nargs="+",
        default=None,
        help="One or more indices of DMS in mapping file (e.g. --dms_index 0 1 2), or 'all' to run all assays.",
    )
    parser.add_argument("--dms_mapping", type=str, default=None, help="CSV with mapping rows when using --dms_index")

    parser.add_argument("--offset-idx", type=int, default=1, help="Offset of mutation positions in mutation column")
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
    if revision is not None:
        print(f"Loading model {model_name} revision {revision} on device {device}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
            token=auth_token,
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
            token=auth_token,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=auth_token,
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=auth_token,
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
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
    # AMPLIFY expects an additive attention mask: 0.0 for tokens to attend, -inf for masked/pad
    if "attention_mask" in inputs:
        am = inputs["attention_mask"]
        # Ensure boolean condition then map True->0.0, False->-inf
        inputs["attention_mask"] = torch.where(
            am.bool(), torch.tensor(0.0, device=am.device), torch.tensor(float("-inf"), device=am.device)
        )
    return inputs


# Removed special-token alignment; we assume a single BOS at index 0 and EOS at the end.


def _log_softmax_logits(model, inputs) -> torch.Tensor:
    # Use autocast for computation (default bf16, fp16 if weights are fp16)
    # Autocast dtype is now CLI specified
    import builtins

    autocast_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    autocast_dtype = autocast_map.get(getattr(builtins, "autocast_dtype", "bf16"), torch.bfloat16)
    with torch.no_grad():
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=True):
            outputs = model(**inputs)
            logits = outputs.logits  # [B, T, V]
            return torch.log_softmax(logits, dim=-1)


def _label_row_hf(row_mut_str: str, sequence: str, token_log_probs: torch.Tensor, tokenizer, offset_idx: int) -> float:
    # token_log_probs expected shape: [1, T, V] with special tokens at positions 0 and T-1
    score = 0.0
    for mutation in row_mut_str.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_id = _char_to_id(tokenizer, wt)
        mt_id = _char_to_id(tokenizer, mt)
        # +1 for BOS (assumes single BOS and single EOS)
        score += (token_log_probs[0, 1 + idx, mt_id] - token_log_probs[0, 1 + idx, wt_id]).item()
    return float(score)


def _compute_pseudo_ppl(seq: str, tokenizer, model, device: torch.device, dms_idx=None) -> float:
    # Sum of log p(x_i | x_\i) by masking one token at a time
    max_len = getattr(model.config, "max_length", 2048)
    # Reserve 2 for special tokens
    window_cap = max_len - 2 if max_len and max_len > 2 else len(seq)

    log_probs = []
    bar_desc = f"pppl DMS={dms_idx}" if dms_idx is not None else "pppl"
    for pos in tqdm(range(len(seq)), desc=bar_desc):
        if window_cap < len(seq):
            start, end = _window_bounds(pos, len(seq), window_cap)
            subseq = seq[start:end]
            local_pos = pos - start
            inputs = _tokenize_seq(tokenizer, subseq, device)
            # Mask position (accounting for BOS at index 0)
            inputs["input_ids"][0, 1 + local_pos] = tokenizer.mask_token_id
        else:
            inputs = _tokenize_seq(tokenizer, seq, device)
            inputs["input_ids"][0, 1 + pos] = tokenizer.mask_token_id

        logp = _log_softmax_logits(model, inputs)
        aa_id = _char_to_id(tokenizer, seq[pos])
        if window_cap < len(seq):
            log_probs.append(logp[0, 1 + local_pos, aa_id].item())
        else:
            log_probs.append(logp[0, 1 + pos, aa_id].item())
    return float(np.sum(log_probs))


def _masked_marginals(
    sequence: str, tokenizer, model, device: torch.device, scoring_window: str, dms_idx=None
) -> torch.Tensor:
    # For each position, mask it, run model, and collect the full vocab log-prob vector.
    max_len = getattr(model.config, "max_length", 2048)
    window_cap = max_len - 2 if max_len and max_len > 2 else len(sequence)
    use_windows = scoring_window == "optimal" and window_cap < len(sequence)

    all_rows = []  # list of [V] tensors per position
    bar_desc = f"mm DMS={dms_idx}" if dms_idx is not None else "mm"
    import builtins

    batch_size = getattr(builtins, "batch_size", 16)
    positions = list(range(len(sequence)))
    for batch_start in tqdm(range(0, len(positions), batch_size), desc=bar_desc, disable=True):
        batch_positions = positions[batch_start : batch_start + batch_size]
        input_ids = []
        attention_masks = []
        for pos in batch_positions:
            if use_windows:
                start, end = _window_bounds(pos, len(sequence), window_cap)
                subseq = sequence[start:end]
                local_pos = pos - start
                inputs = tokenizer(subseq, return_tensors="pt", add_special_tokens=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["input_ids"][0, 1 + local_pos] = tokenizer.mask_token_id
                input_ids.append(inputs["input_ids"][0])
                attention_masks.append(inputs.get("attention_mask", None))
            else:
                inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["input_ids"][0, 1 + pos] = tokenizer.mask_token_id
                input_ids.append(inputs["input_ids"][0])
                attention_masks.append(inputs.get("attention_mask", None))
        # Stack batch
        input_ids = torch.stack(input_ids, dim=0)
        batch_inputs = {"input_ids": input_ids, "attention_mask": None}
        # if attention_masks[0] is not None:
        #     batch_inputs["attention_mask"] = torch.stack(attention_masks, dim=0)
        logp = _log_softmax_logits(model, batch_inputs)
        # logp shape: [batch, T, V]
        for i, pos in enumerate(batch_positions):
            if use_windows:
                start, end = _window_bounds(pos, len(sequence), window_cap)
                local_pos = pos - start
                all_rows.append(logp[i, 1 + local_pos].detach().cpu())
            else:
                all_rows.append(logp[i, 1 + pos].detach().cpu())

    # Shape [1, L+2, V] to keep +1 offset indexing
    token_probs = torch.zeros((1, len(sequence) + 2, all_rows[0].size(-1)))
    for i, row in enumerate(all_rows):
        token_probs[0, 1 + i] = row
    return token_probs


def main(args):
    # Set batch size globally for _masked_marginals
    import builtins

    builtins.batch_size = args.batch_size
    # Set autocast dtype globally for _log_softmax_logits
    import builtins

    builtins.autocast_dtype = args.autocast_dtype
    if not os.path.exists(args.dms_output):
        os.mkdir(args.dms_output)

    # Resolve device
    device = _device(use_gpu=not args.nogpu)

    # Load model(s) once
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()  # keep output clean
    tokenizers_models = []
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    model_dtype = dtype_map.get(args.model_dtype, torch.float32)
    for model_name in args.model_name_or_path:
        tokenizer, model = _load_hf_model_and_tokenizer(
            model_name, device, revision=args.revision, auth_token=args.use_auth_token
        )
        model.to(device, dtype=model_dtype)
        tokenizers_models.append((model_name, tokenizer, model))

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

        # Score with each model and append columns
        for model_name, tokenizer, model in tokenizers_models:
            col_name = os.path.basename(model_name.rstrip("/"))

            # Skip if already computed and not overwriting
            if os.path.exists(dms_output_file) and not args.overwrite_prior_scores:
                prior_df = pd.read_csv(dms_output_file)
                if col_name in prior_df.columns:
                    df = prior_df
                    print(f"Skipping {col_name} (already present). Use --overwrite-prior-scores to recompute.")
                    continue

            if args.scoring_strategy == "wt-marginals":
                inputs = _tokenize_seq(tokenizer, sequence, device)
                max_len = getattr(model.config, "max_length", 2048)
                bar_desc = f"wtm DMS={dms_idx}" if dms_idx is not None else "wtm"
                if inputs["input_ids"].shape[1] > max_len and args.scoring_window == "optimal":
                    token_probs = _masked_marginals(
                        sequence, tokenizer, model, device, scoring_window="optimal", dms_idx=dms_idx
                    )
                else:
                    for pos in tqdm([0], desc=bar_desc):
                        token_probs = _log_softmax_logits(model, inputs).detach().cpu()

                df[col_name] = df.apply(
                    lambda row: _label_row_hf(row[mutant_col], sequence, token_probs, tokenizer, offset_idx),
                    axis=1,
                )

            elif args.scoring_strategy == "masked-marginals":
                token_probs = _masked_marginals(
                    sequence, tokenizer, model, device, args.scoring_window, dms_idx=dms_idx
                )
                df[col_name] = df.apply(
                    lambda row: _label_row_hf(row[mutant_col], sequence, token_probs, tokenizer, offset_idx),
                    axis=1,
                )

            elif args.scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                if "mutated_sequence" not in df:
                    df["mutated_sequence"] = df.progress_apply(
                        lambda row: _get_mutated_sequence(row[mutant_col], sequence, offset_idx), axis=1
                    )
                df[col_name] = df.progress_apply(
                    lambda row: _compute_pseudo_ppl(row["mutated_sequence"], tokenizer, model, device, dms_idx=dms_idx),
                    axis=1,
                )

            # Write out incrementally if prior file exists
            if os.path.exists(dms_output_file) and not args.overwrite_prior_scores:
                prior_df = pd.read_csv(dms_output_file)
                assert col_name not in prior_df.columns, f"Column {col_name} already exists in {dms_output_file}"
                prior_df = prior_df.merge(df[[col_name, mutant_col]], on=mutant_col)
                prior_df.to_csv(dms_output_file, index=False)
                df = prior_df
            else:
                df.to_csv(dms_output_file, index=False)

        # If multiple models provided, add an ensemble average
        if len(args.model_name_or_path) > 1:
            names = [os.path.basename(m.rstrip("/")) for m in args.model_name_or_path]
            present = [n for n in names if n in df.columns]
            if present:
                df["Ensemble_AMPLIFY"] = df[present].mean(axis=1)
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
