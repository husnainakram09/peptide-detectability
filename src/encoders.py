"""Sequence encoders.

Phase 4 — integer encoding for the 1D-CNN: map each residue to its index from
``config.AA_TO_IDX`` (1..20), reserve 0 for padding, pad/truncate to
``config.MAX_LEN``. Embedding layers should be sized ``VOCAB_SIZE + 1``.

Phase 5 — ESM-2 protein-language-model embeddings: run a (small) pretrained ESM-2
model from Hugging Face and MEAN-POOL the last hidden state over the real residue
tokens (excluding the CLS/EOS special tokens and padding). Results are cached to
``embeddings_cache/{model_tag}_{split}_{data_hash}.npy`` so they are computed only
once (extract on a free Colab GPU, download, and reuse locally).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# torch is imported inside the functions that need it, so integer encoding
# (used for NumPy CNN inference) works without PyTorch installed.

PAD_IDX = config.PAD_IDX        # 0
MAX_LEN = config.MAX_LEN        # 50
VOCAB_SIZE = config.VOCAB_SIZE  # 20  (size Embedding as VOCAB_SIZE + 1)
_AA_TO_IDX = config.AA_TO_IDX   # {'A': 1, ..., 'Y': 20}


# =========================================================================== #
# Integer encoding (1D-CNN)
# =========================================================================== #
def encode_sequence(sequence: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Integer-encode one sequence to a fixed-length ``(max_len,)`` int64 array.

    Unknown characters are skipped; the sequence is truncated to ``max_len`` and
    right-padded with the padding index (0).
    """
    seq = str(sequence).strip().upper()
    ids = [_AA_TO_IDX[c] for c in seq if c in _AA_TO_IDX][:max_len]
    out = np.zeros(max_len, dtype=np.int64)
    out[: len(ids)] = ids
    return out


def encode_sequences(sequences, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode an iterable of sequences into an ``(n, max_len)`` int64 array."""
    seqs = list(sequences)
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i] = encode_sequence(s, max_len=max_len)
    return out


def encode_dataframe(df: pd.DataFrame, seq_col: str = "sequence",
                     label_col: str = "label", max_len: int = MAX_LEN):
    """Encode a dataframe into tensors.

    Returns ``(X, y)`` where ``X`` is a ``LongTensor`` of shape ``(n, max_len)``
    and ``y`` is a ``FloatTensor`` of shape ``(n,)`` when ``label_col`` is present
    (otherwise ``y`` is ``None``).
    """
    import torch

    x = torch.from_numpy(encode_sequences(df[seq_col].tolist(), max_len=max_len)).long()
    y = None
    if label_col in df.columns:
        y = torch.tensor(df[label_col].to_numpy(), dtype=torch.float32)
    return x, y


# =========================================================================== #
# ESM-2 embeddings (protein language model)
# =========================================================================== #
DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"

# Known embedding dimensions (also readable from the model config at runtime).
ESM_EMB_DIM = {
    "facebook/esm2_t6_8M_UR50D": 320,   # t6
    "facebook/esm2_t12_35M_UR50D": 480,  # t12
}


def esm_model_tag(model_id: str) -> str:
    """Short tag used in cache filenames, e.g. 'esm2_t6_8M_UR50D'."""
    return model_id.rsplit("/", 1)[-1]


def esm_emb_dim(model_id: str) -> "int | None":
    return ESM_EMB_DIM.get(model_id)


def esm_cache_path(model_id: str, split: str, data_hash: str,
                   cache_dir=None) -> Path:
    cache_dir = Path(cache_dir) if cache_dir is not None else config.EMBEDDINGS_CACHE_DIR
    return cache_dir / f"{esm_model_tag(model_id)}_{split}_{data_hash}.npy"


def extract_esm_embeddings(sequences, model_id: str = DEFAULT_ESM_MODEL,
                           batch_size: int = 16, device=None,
                           max_len: "int | None" = None,
                           progress: bool = True) -> np.ndarray:
    """Mean-pooled ESM-2 embeddings for each peptide → ``(n, emb_dim)`` float32.

    For every peptide the last hidden state is averaged over the real residue
    tokens only (CLS/EOS special tokens and padding are excluded via the
    attention and special-tokens masks). Uses GPU if available, else CPU.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if max_len is None:
        max_len = MAX_LEN + 2  # room for the BOS/EOS special tokens

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    seqs = [str(s).strip().upper() for s in sequences]
    starts = range(0, len(seqs), batch_size)
    if progress:
        from tqdm import tqdm
        starts = tqdm(starts, desc=f"ESM {esm_model_tag(model_id)} [{device}]", unit="batch")

    chunks = []
    with torch.no_grad():
        for start in starts:
            batch = seqs[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_len, return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask").to(device)   # 1 at CLS/EOS(/pad)
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state               # (B, T, H)
            # residue mask = non-pad AND not-special  -> 1 only at real residues
            residue = (enc["attention_mask"] * (1 - special)).unsqueeze(-1).float()
            summed = (hidden * residue).sum(dim=1)                # (B, H)
            counts = residue.sum(dim=1).clamp(min=1.0)            # (B, 1)
            chunks.append((summed / counts).cpu().numpy().astype(np.float32))

    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def cached_esm_embeddings(sequences, split: str, data_hash: str,
                          model_id: str = DEFAULT_ESM_MODEL, cache_dir=None,
                          **extract_kwargs) -> np.ndarray:
    """Load cached embeddings if present (and the right size), else compute+cache.

    Cache file: ``embeddings_cache/{model_tag}_{split}_{data_hash}.npy``.
    """
    sequences = list(sequences)
    path = esm_cache_path(model_id, split, data_hash, cache_dir)
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(sequences):
            print(f"  [cache] {path.name} ({emb.shape[0]}x{emb.shape[1]})")
            return emb
        print(f"  [warn]  cached {path.name} has {emb.shape[0]} rows, expected "
              f"{len(sequences)} — recomputing.")
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = extract_esm_embeddings(sequences, model_id=model_id, **extract_kwargs)
    np.save(path, emb)
    print(f"  [save]  {path.name} ({emb.shape[0]}x{emb.shape[1]})")
    return emb
