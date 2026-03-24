"""
Dataset Explorer Web Server

Usage:
    python explore_data.py

Then open http://localhost:3000 in your browser.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm
from fasthtml.common import (
    H1,
    H2,
    H3,
    A,
    Button,
    Code,
    Div,
    FastHTML,
    Form,
    Group,
    Input,
    Main,
    P,
    Pre,
    Span,
    Style,
    Table,
    Tbody,
    Td,
    Th,
    Thead,
    Title,
    Tr,
    picolink,
    serve,
)

DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp4096")
TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH", "./data/tokenizers/fineweb_4096_bpe.model"
)

custom_css = Style("""
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .stat-card { background: var(--pico-background-color); border: 1px solid var(--pico-muted-border-color); padding: 1rem; border-radius: 0.5rem; }
    .stat-value { font-size: 2rem; font-weight: bold; }
    .stat-label { color: var(--pico-muted-color); }
    .token-sample { font-family: monospace; background: var(--pico-muted-background-color); padding: 1rem; border-radius: 0.5rem; overflow-x: auto; white-space: pre-wrap; word-break: break-word; max-height: 400px; overflow-y: auto; }
    .decoded-text { font-family: monospace; background: #f0f0f0; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }
    .nav-links { margin: 1rem 0; }
    .nav-links a { margin-right: 1rem; }
    pre { font-size: 0.85rem; }
""")

app = FastHTML(hdrs=(picolink, custom_css))

train_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_train_*.bin")))
val_files = sorted(glob.glob(os.path.join(DATA_PATH, "fineweb_val_*.bin")))

sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)


def navbar():
    return Div(
        A("Overview", href="/"),
        Span(" | "),
        A("Browse Shards", href="/shards"),
        Span(" | "),
        A("Token Analysis", href="/tokens"),
        Span(" | "),
        A("Vocabulary", href="/vocab"),
        id="nav",
        cls="nav-links",
    )


def load_data_shard(file: str) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens_np


def get_shard_stats(file: str) -> dict:
    tokens = load_data_shard(file)
    unique, counts = np.unique(tokens, return_counts=True)
    top_tokens = sorted(zip(counts, unique), reverse=True)[:20]
    return {
        "file": os.path.basename(file),
        "size_bytes": os.path.getsize(file),
        "num_tokens": len(tokens),
        "num_unique_tokens": len(unique),
        "min_token": int(tokens.min()),
        "max_token": int(tokens.max()),
        "top_tokens": [(int(count), int(tok)) for count, tok in top_tokens],
    }


def decode_tokens(tokens: list[int], max_len: int = 1000) -> str:
    decoded = sp.DecodeIds(tokens[:max_len])
    return decoded


def get_dataset_stats() -> dict:
    total_train_tokens = 0
    total_train_files = len(train_files)
    total_val_tokens = 0
    total_val_files = len(val_files)

    for f in train_files:
        tokens = load_data_shard(f)
        total_train_tokens += len(tokens)

    for f in val_files:
        tokens = load_data_shard(f)
        total_val_tokens += len(tokens)

    manifest_path = Path(DATA_PATH).parent / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    return {
        "train_files": total_train_files,
        "val_files": total_val_files,
        "total_train_tokens": total_train_tokens,
        "total_val_tokens": total_val_tokens,
        "vocab_size": int(sp.vocab_size()),
        "manifest": manifest,
    }


@app.get("/")
def home():
    stats = get_dataset_stats()
    sample_shard = val_files[0] if val_files else None
    sample_tokens_200 = (
        load_data_shard(sample_shard)[:200].tolist() if sample_shard else []
    )
    sample_tokens_50 = (
        load_data_shard(sample_shard)[:50].tolist() if sample_shard else []
    )

    return Title("FineWeb Dataset Explorer"), Main(
        H1("FineWeb Dataset Explorer"),
        navbar(),
        H2("Dataset Statistics", id="overview"),
        Div(
            Div(
                Div(
                    Code(str(stats["train_files"])),
                    P("Training Shards", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(str(stats["val_files"])),
                    P("Validation Shards", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(f"{stats['total_train_tokens']:,}"),
                    P("Training Tokens", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(f"{stats['total_val_tokens']:,}"),
                    P("Validation Tokens", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(str(stats["vocab_size"])),
                    P("Vocabulary Size", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            cls="stats-grid",
        ),
        H2("Tokenizer Info"),
        Table(
            Thead(Tr(Th("Property"), Th("Value"))),
            Tbody(
                Tr(
                    Td("Model File"),
                    Td(
                        Code(
                            stats.get("manifest", {})
                            .get("tokenizers", [{}])[0]
                            .get("model_path", "N/A")
                        )
                    ),
                ),
                Tr(Td("BOS Token ID"), Td(str(sp.bos_id()))),
                Tr(Td("EOS Token ID"), Td(str(sp.eos_id()))),
                Tr(Td("PAD Token ID"), Td(str(sp.pad_id()))),
                Tr(Td("UNK Token ID"), Td(str(sp.unk_id()))),
            ),
        ),
        H2("Sample Decoded Text"),
        P(
            f"First 200 tokens from {os.path.basename(sample_shard) if sample_shard else 'N/A'}:"
        ),
        Pre(decode_tokens(sample_tokens_200), cls="decoded-text"),
        H3("Sample Token IDs"),
        Pre(str(sample_tokens_50), cls="token-sample"),
        cls="container",
    )


@app.get("/shards")
def shards():
    all_files = [(f, "train") for f in train_files] + [(f, "val") for f in val_files]

    rows = []
    for file, split in all_files:
        basename = os.path.basename(file)
        rows.append(
            Tr(
                Td(A(basename, href=f"/shard/{split}/{basename}")),
                Td(split),
                Td(f"{os.path.getsize(file):,} bytes"),
            )
        )

    return Title("Browse Shards"), Main(
        H1("Browse Dataset Shards"),
        navbar(),
        P(f"{len(train_files)} train, {len(val_files)} val shards"),
        Table(
            Thead(Tr(Th("File"), Th("Split"), Th("Size"))),
            Tbody(*rows),
        ),
        cls="container",
    )


@app.get("/shard/{split}/{filename}")
def shard_detail(split: str, filename: str, page: int = 1):
    if split == "train":
        files = train_files
    else:
        files = val_files

    matching = [f for f in files if os.path.basename(f) == filename]
    if not matching:
        return Title("Error"), Main(H1("Shard not found"), cls="container")

    file = matching[0]
    tokens = load_data_shard(file)
    total_tokens = len(tokens)

    bos_positions = np.where(tokens == sp.bos_id())[0].tolist()
    if not bos_positions:
        bos_positions = [0]
    if bos_positions[0] != 0:
        bos_positions.insert(0, 0)
    doc_starts = bos_positions
    total_docs = len(doc_starts)
    page = max(1, min(page, total_docs))

    start = doc_starts[page - 1]
    end = doc_starts[page] if page < total_docs else total_tokens
    doc_tokens = tokens[start:end].tolist()
    decoded = decode_tokens(doc_tokens, max_len=len(doc_tokens))

    base_url = f"/shard/{split}/{filename}"

    nav_buttons = Div()
    if total_docs > 1:
        nav_buttons = Div(
            A("Prev", href=f"{base_url}?page={page - 1}") if page > 1 else Span("Prev"),
            Span(f"  Doc {page}/{total_docs}  "),
            A("Next", href=f"{base_url}?page={page + 1}")
            if page < total_docs
            else Span("Next"),
            cls="nav-links",
        )

    return Title(f"Shard: {filename}"), Main(
        H1(f"Shard: {filename}"),
        navbar(),
        P(
            f"{total_tokens:,} tokens, {total_docs:,} documents ({os.path.getsize(file):,} bytes)"
        ),
        H2(f"Doc {page}/{total_docs} — tokens {start:,}–{end:,}"),
        nav_buttons,
        Pre(decoded, cls="decoded-text"),
        nav_buttons,
        cls="container",
    )


@app.get("/tokens")
def token_analysis():
    all_tokens = []
    for f in train_files[:5]:
        tokens = load_data_shard(f)
        all_tokens.extend(tokens.tolist())

    all_tokens = np.array(all_tokens)
    total_tokens = len(all_tokens)
    unique, counts = np.unique(all_tokens, return_counts=True)

    sorted_desc = np.argsort(-counts)
    sorted_asc = np.argsort(counts)

    top_50 = [(int(unique[i]), int(counts[i])) for i in sorted_desc[:50]]
    bottom_50 = [(int(unique[i]), int(counts[i])) for i in sorted_asc[:50]]

    sample = all_tokens[:100000].astype(int).tolist()
    decoded_bytes = len(sp.DecodeIds(sample).encode("utf-8"))
    bytes_per_token = decoded_bytes / len(sample)

    top_rows = []
    for i, (tok_id, count) in enumerate(top_50):
        piece = sp.IdToPiece(tok_id)
        piece_type = (
            "control"
            if sp.IsControl(tok_id)
            else "byte"
            if sp.IsByte(tok_id)
            else "normal"
        )
        top_rows.append(
            Tr(
                Td(str(i + 1)),
                Td(Code(str(tok_id))),
                Td(f"{count:,}"),
                Td(f"{count / total_tokens * 100:.3f}%"),
                Td(piece_type),
                Td(Code(piece) if piece_type == "normal" else piece),
            )
        )

    bottom_rows = []
    for i, (tok_id, count) in enumerate(bottom_50):
        piece = sp.IdToPiece(tok_id)
        piece_type = (
            "control"
            if sp.IsControl(tok_id)
            else "byte"
            if sp.IsByte(tok_id)
            else "normal"
        )
        bottom_rows.append(
            Tr(
                Td(str(i + 1)),
                Td(Code(str(tok_id))),
                Td(f"{count:,}"),
                Td(piece_type),
                Td(Code(piece) if piece_type == "normal" else piece),
            )
        )

    control_count = sum(1 for tid, _ in top_50 if sp.IsControl(tid))
    byte_count = sum(1 for tid, _ in top_50 if sp.IsByte(tid))
    normal_count = sum(
        1 for tid, _ in top_50 if not sp.IsControl(tid) and not sp.IsByte(tid)
    )

    return Title("Token Analysis"), Main(
        H1("Token Analysis"),
        navbar(),
        P("Analysis based on first 5 training shards"),
        H2("Compression"),
        Div(
            Div(
                Div(
                    Code(f"{bytes_per_token:.2f}"),
                    P("Bytes / Token", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(f"{len(unique):,}"),
                    P("Unique Tokens Used", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(str(sp.vocab_size())),
                    P("Vocab Size", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            cls="stats-grid",
        ),
        H2("Top 50 Most Frequent Tokens"),
        Table(
            Thead(
                Tr(
                    Th("Rank"),
                    Th("Token ID"),
                    Th("Count"),
                    Th("Frequency"),
                    Th("Type"),
                    Th("Piece"),
                )
            ),
            Tbody(*top_rows),
        ),
        H2("Token Distribution (Top 50)"),
        Table(
            Thead(Tr(Th("Type"), Th("Count in Top 50"))),
            Tbody(
                Tr(Td("Control"), Td(str(control_count))),
                Tr(Td("Byte"), Td(str(byte_count))),
                Tr(Td("Normal"), Td(str(normal_count))),
            ),
        ),
        H2("50 Least Frequent Tokens"),
        Table(
            Thead(Tr(Th("Rank"), Th("Token ID"), Th("Count"), Th("Type"), Th("Piece"))),
            Tbody(*bottom_rows),
        ),
        cls="container",
    )


@app.get("/vocab")
def vocab(q: str = "", page: int = 1):
    page_size = 200
    vocab_size = sp.vocab_size()
    total_pages = (vocab_size + page_size - 1) // page_size

    if q:
        matches = []
        for tok_id in range(vocab_size):
            piece = sp.IdToPiece(tok_id)
            if q.lower() in piece.lower():
                matches.append(tok_id)
        rows = []
        for tok_id in matches:
            piece = sp.IdToPiece(tok_id)
            piece_type = (
                "control"
                if sp.IsControl(tok_id)
                else "byte"
                if sp.IsByte(tok_id)
                else "normal"
            )
            rows.append(
                Tr(
                    Td(Code(str(tok_id))),
                    Td(Code(piece) if piece_type == "normal" else piece),
                    Td(piece_type),
                )
            )
        results_text = P(f"{len(matches)} results for {repr(q)}")
    else:
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = min(start + page_size, vocab_size)

        rows = []
        for tok_id in range(start, end):
            piece = sp.IdToPiece(tok_id)
            piece_type = (
                "control"
                if sp.IsControl(tok_id)
                else "byte"
                if sp.IsByte(tok_id)
                else "normal"
            )
            rows.append(
                Tr(
                    Td(Code(str(tok_id))),
                    Td(Code(piece) if piece_type == "normal" else piece),
                    Td(piece_type),
                )
            )
        results_text = P(f"Tokens {start}–{end - 1} of {vocab_size}")

    base_url = "/vocab"
    search_extra = f"&q={q}" if q else ""

    nav_buttons = Div()
    if not q and total_pages > 1:
        nav_buttons = Div(
            A("Prev", href=f"{base_url}?page={page - 1}{search_extra}")
            if page > 1
            else Span("Prev"),
            Span(f"  Page {page}/{total_pages}  "),
            A("Next", href=f"{base_url}?page={page + 1}{search_extra}")
            if page < total_pages
            else Span("Next"),
            cls="nav-links",
        )

    return Title("Vocabulary"), Main(
        H1("Vocabulary"),
        navbar(),
        Form(
            Group(
                Input(
                    type="text", name="q", value=q, placeholder="Search token pieces..."
                ),
                Button("Search"),
            ),
            method="get",
        ),
        results_text,
        nav_buttons,
        Table(
            Thead(Tr(Th("Token ID"), Th("Piece"), Th("Type"))),
            Tbody(*rows),
        ),
        nav_buttons,
        cls="container",
    )


if __name__ == "__main__":
    print("Starting FineWeb Dataset Explorer...")
    print(f"Data path: {DATA_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Training shards: {len(train_files)}")
    print(f"Validation shards: {len(val_files)}")
    print("\nOpen http://localhost:3000 in your browser")
    serve(port=3000)
