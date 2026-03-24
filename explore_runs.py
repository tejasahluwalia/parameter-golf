"""
Run Explorer Web Server

Usage:
    python explore_runs.py

Then open http://localhost:3000 in your browser.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from fasthtml.common import (
    H1,
    H2,
    A,
    Code,
    Div,
    FastHTML,
    FileResponse,
    Main,
    P,
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

LOGS_DIR = Path(os.environ.get("LOGS_DIR", "./logs"))

custom_css = Style("""
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .stat-card { background: var(--pico-background-color); border: 1px solid var(--pico-muted-border-color); padding: 1rem; border-radius: 0.5rem; }
    .stat-value { font-size: 1.5rem; font-weight: bold; }
    .stat-label { color: var(--pico-muted-color); font-size: 0.85rem; }
    .nav-links { margin: 1rem 0; }
    .nav-links a { margin-right: 1rem; }
    .good { color: #22863a; font-weight: bold; }
    .ok { color: #b08800; font-weight: bold; }
    .bad { color: #cb2431; font-weight: bold; }
    table { font-size: 0.9rem; }
    td, th { padding: 0.4rem 0.6rem !important; white-space: nowrap; }
    .detail-table td:first-child { font-weight: 600; white-space: nowrap; width: 200px; }
    pre.log-block { font-size: 0.8rem; max-height: 600px; overflow-y: auto; background: var(--pico-muted-background-color); padding: 1rem; border-radius: 0.5rem; }
    .tag { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; margin-right: 0.25rem; }
    .tag-vocab { background: #e8d5f5; color: #6b21a8; }
    .tag-early { background: #fee2e2; color: #991b1b; }
    .tag-complete { background: #d1fae5; color: #065f46; }
""")

app = FastHTML(hdrs=(picolink, custom_css))


def parse_log_file(filepath: Path) -> dict | None:
    """Parse a log file and extract run information."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    run_id = filepath.stem

    # Split into code and log sections
    parts = text.split("=" * 100)
    if len(parts) < 3:
        return None

    log_section = parts[-1]  # Last section has the actual training logs

    run = {"id": run_id, "file": filepath.name, "has_loss_html": False}

    # Check for loss html
    loss_html_path = filepath.parent / f"{run_id}_loss.html"
    if loss_html_path.exists():
        run["has_loss_html"] = True

    # Parse configuration lines
    def extract(pattern, text, default=None):
        m = re.search(pattern, text)
        return m.group(1) if m else default

    run["tokenizer_path"] = extract(r"tokenizer_path=(\S+)", log_section, "")
    run["vocab_size"] = int(
        extract(r"fineweb_(\d+)_bpe", run["tokenizer_path"], "0") or "0"
    )
    run["dataset"] = extract(r"dataset:(\S+)", log_section, "")
    run["train_shards"] = int(extract(r"train_shards:(\d+)", log_section, "0") or "0")
    run["val_tokens"] = int(extract(r"tokens:(\d+)", log_section, "0") or "0")
    run["model_params"] = int(extract(r"model_params:(\d+)", log_section, "0") or "0")
    run["world_size"] = int(extract(r"world_size:(\d+)", log_section, "1") or "1")
    run["grad_accum_steps"] = int(
        extract(r"grad_accum_steps:(\d+)", log_section, "1") or "1"
    )
    run["num_heads"] = int(extract(r"num_heads:(\d+)", log_section, "0") or "0")
    run["num_kv_heads"] = int(extract(r"num_kv_heads:(\d+)", log_section, "0") or "0")
    run["tie_embeddings"] = extract(r"tie_embeddings:(\S+)", log_section, "") == "True"
    run["embed_lr"] = float(extract(r"embed_lr:([\d.]+)", log_section, "0") or "0")
    run["head_lr"] = float(extract(r"head_lr:([\d.]+)", log_section, "0") or "0")
    run["matrix_lr"] = float(extract(r"matrix_lr:([\d.]+)", log_section, "0") or "0")
    run["scalar_lr"] = float(extract(r"scalar_lr:([\d.]+)", log_section, "0") or "0")
    run["train_batch_tokens"] = int(
        extract(r"train_batch_tokens:(\d+)", log_section, "0") or "0"
    )
    run["train_seq_len"] = int(extract(r"train_seq_len:(\d+)", log_section, "0") or "0")
    run["iterations"] = int(extract(r"iterations:(\d+)", log_section, "0") or "0")
    run["warmup_steps"] = int(extract(r"warmup_steps:(\d+)", log_section, "0") or "0")
    run["max_wallclock_seconds"] = float(
        extract(r"max_wallclock_seconds:([\d.]+)", log_section, "0") or "0"
    )
    run["seed"] = int(extract(r"seed:(\d+)", log_section, "0") or "0")

    # Parse timestamp (e.g. "Sun Mar 22 17:32:25 2026")
    ts_match = re.search(
        r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\w+\s+\d+\s+[\d:]+\s+\d{4})", text
    )
    if ts_match:
        ts_str = ts_match.group(1).strip()
        run["timestamp"] = datetime.strptime(ts_str, "%a %b %d %H:%M:%S %Y")
        run["timestamp_str"] = ts_str
    else:
        run["timestamp"] = None
        run["timestamp_str"] = ""

    # Parse GPU info
    gpu_match = re.search(r"\|\s+\d+\s+(NVIDIA[^|]+?)\s+Off\s+\|", text)
    run["gpu"] = gpu_match.group(1).strip() if gpu_match else "Unknown"

    # Parse Python/PyTorch versions
    run["python_version"] = extract(r"Running Python ([\d.]+)", log_section, "")
    run["pytorch_version"] = extract(r"Running PyTorch ([\d.+\w]+)", log_section, "")

    # Parse training steps
    train_steps = re.findall(
        r"step:(\d+)/(\d+) train_loss:([\d.]+) train_time:(\d+)ms", log_section
    )
    run["train_steps"] = [
        (int(s), int(t), float(l), int(ms)) for s, t, l, ms in train_steps
    ]

    # Parse validation steps
    val_steps = re.findall(
        r"step:(\d+)/(\d+) val_loss:([\d.]+) val_bpb:([\d.]+) train_time:(\d+)ms",
        log_section,
    )
    run["val_steps"] = [
        (int(s), int(t), float(l), float(b), int(ms)) for s, t, l, b, ms in val_steps
    ]

    # Parse stopping info
    stopping = extract(
        r"stopping_early: wallclock_cap train_time:(\d+)ms step:(\d+)/(\d+)",
        log_section,
    )
    if stopping:
        m = re.search(
            r"stopping_early: wallclock_cap train_time:(\d+)ms step:(\d+)/(\d+)",
            log_section,
        )
        run["stopped_early"] = True
        run["stop_time_ms"] = int(m.group(1))
        run["stop_step"] = int(m.group(2))
    else:
        run["stopped_early"] = False

    # Parse final stats
    run["peak_mem_allocated"] = int(
        extract(r"peak memory allocated: (\d+) MiB", log_section, "0") or "0"
    )
    run["peak_mem_reserved"] = int(
        extract(r"reserved: (\d+) MiB", log_section, "0") or "0"
    )
    run["serialized_model_bytes"] = int(
        extract(r"Serialized model: (\d+) bytes", log_section, "0") or "0"
    )
    run["code_size_bytes"] = int(
        extract(r"Code size: (\d+) bytes", log_section, "0") or "0"
    )
    run["total_submission_bytes"] = int(
        extract(r"Total submission size: (\d+) bytes", log_section, "0") or "0"
    )

    # int8+zlib stats
    m = re.search(
        r"Serialized model int8\+zlib: (\d+) bytes \(payload:(\d+) raw_torch:(\d+) payload_ratio:([\d.]+)x\)",
        log_section,
    )
    if m:
        run["int8_zlib_bytes"] = int(m.group(1))
        run["int8_payload_bytes"] = int(m.group(2))
        run["payload_ratio"] = float(m.group(4))
    else:
        run["int8_zlib_bytes"] = 0
        run["int8_payload_bytes"] = 0
        run["payload_ratio"] = 0.0

    run["total_submission_int8_bytes"] = int(
        extract(r"Total submission size int8\+zlib: (\d+) bytes", log_section, "0")
        or "0"
    )

    # Final roundtrip val
    m = re.search(
        r"final_int8_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+) eval_time:(\d+)ms",
        log_section,
    )
    if m:
        run["final_val_loss"] = float(m.group(1))
        run["final_val_bpb"] = float(m.group(2))
        run["final_eval_time_ms"] = int(m.group(3))
    else:
        run["final_val_loss"] = None
        run["final_val_bpb"] = None

    # Exact roundtrip
    m = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)",
        log_section,
    )
    if m:
        run["final_val_loss_exact"] = float(m.group(1))
        run["final_val_bpb_exact"] = float(m.group(2))
    else:
        run["final_val_loss_exact"] = run["final_val_loss"]
        run["final_val_bpb_exact"] = run["final_val_bpb"]

    # Last training loss
    if run["train_steps"]:
        run["last_train_loss"] = run["train_steps"][-1][2]
        run["last_train_step"] = run["train_steps"][-1][0]
        run["total_train_time_ms"] = run["train_steps"][-1][3]
    else:
        run["last_train_loss"] = None
        run["last_train_step"] = 0
        run["total_train_time_ms"] = 0

    # Best val_bpb
    if run["val_steps"]:
        best = min(run["val_steps"], key=lambda x: x[3])
        run["best_val_bpb"] = best[3]
        run["best_val_step"] = best[0]
    else:
        run["best_val_bpb"] = None
        run["best_val_step"] = 0

    return run


def load_all_runs() -> list[dict]:
    runs = []
    for f in sorted(LOGS_DIR.glob("*.txt")):
        run = parse_log_file(f)
        if run:
            runs.append(run)
    # Sort by timestamp (newest first), None timestamps at end
    runs.sort(
        key=lambda r: (r["timestamp"] is None, r["timestamp"] or datetime.min),
        reverse=True,
    )
    return runs


def fmt_bytes(b: int) -> str:
    if b >= 1_000_000:
        return f"{b / 1_000_000:.2f} MB"
    if b >= 1_000:
        return f"{b / 1_000:.1f} KB"
    return f"{b} B"


def fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_time(ms: int) -> str:
    s = ms / 1000
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.1f}s"


def bpb_class(val: float | None) -> str:
    if val is None:
        return ""
    if val < 2.0:
        return "good"
    if val < 2.5:
        return "ok"
    return "bad"


def navbar():
    return Div(
        A("Runs Table", href="/"),
        id="nav",
        cls="nav-links",
    )


@app.get("/")
def home(ref: str = None):
    runs = load_all_runs()

    # Find reference run if set
    ref_run = None
    if ref:
        for r in runs:
            if r["id"] == ref:
                ref_run = r
                break

    # Helper for delta formatting
    def delta_html(val, ref_val, is_pct=True, lower_is_better=True):
        if (
            ref_run is None
            or val is None
            or ref_val is None
            or val == ref_val
            or ref_val == 0
        ):
            return ""
        diff = ((val - ref_val) / abs(ref_val)) * 100
        sign = "+" if diff > 0 else ""
        color = (
            "var(--pico-primary)"
            if (lower_is_better and diff < 0) or (not lower_is_better and diff > 0)
            else "var(--pico-del-color)"
        )
        return Span(f" ({sign}{diff:.1f}%)", style=f"color:{color};font-size:0.75em")

    def delta_str(val, ref_val):
        if (
            ref_run is None
            or val is None
            or ref_val is None
            or val == ref_val
            or ref_val == 0
        ):
            return ""
        diff = ((val - ref_val) / abs(ref_val)) * 100
        sign = "+" if diff > 0 else ""
        return f" ({sign}{diff:.1f}%)"

    rows = []
    for r in runs:
        status_tag = (
            Span("complete", cls="tag tag-complete")
            if not r["stopped_early"]
            else Span("early stop", cls="tag tag-early")
        )
        vocab_tag = (
            Span(f"sp{r['vocab_size']}", cls="tag tag-vocab") if r["vocab_size"] else ""
        )

        final_bpb = r["final_val_bpb"]
        best_bpb = r["best_val_bpb"]

        # Delta for final bpb (lower is better)
        final_bpb_delta = (
            delta_str(final_bpb, ref_run["final_val_bpb"])
            if ref_run and final_bpb
            else ""
        )
        best_bpb_delta = (
            delta_str(best_bpb, ref_run["best_val_bpb"]) if ref_run and best_bpb else ""
        )
        int8_delta = (
            delta_str(r["int8_zlib_bytes"], ref_run["int8_zlib_bytes"])
            if ref_run and r["int8_zlib_bytes"]
            else ""
        )

        # Is this the reference run?
        is_ref = r["id"] == ref

        # Build cell content with optional delta
        bpb_content = (
            Span(f"{final_bpb:.4f}", cls=bpb_class(final_bpb))
            if final_bpb
            else Span("—")
        )
        if final_bpb_delta:
            bpb_content = Div(
                bpb_content, Span(final_bpb_delta, style="font-size:0.75em")
            )

        best_bpb_content = (
            Span(f"{best_bpb:.4f}", cls=bpb_class(best_bpb)) if best_bpb else Span("—")
        )
        if best_bpb_delta:
            best_bpb_content = Div(
                best_bpb_content, Span(best_bpb_delta, style="font-size:0.75em")
            )

        int8_content = fmt_bytes(r["int8_zlib_bytes"]) if r["int8_zlib_bytes"] else "—"
        if int8_delta:
            int8_content = Div(int8_content, Span(int8_delta, style="font-size:0.75em"))

        ref_btn = A(
            "★",
            href=f"/?ref={r['id']}",
            title="Set as reference",
            style="color:gold" if is_ref else "color:#ccc",
        )

        rows.append(
            Tr(
                Td(ref_btn),
                Td(A(r["id"][:8], href=f"/run/{r['id']}"), title=r["id"]),
                Td(r.get("timestamp_str") or "—"),
                Td(status_tag, " ", vocab_tag),
                Td(Code(fmt_params(r["model_params"])), style="text-align:right"),
                Td(bpb_content, style="text-align:right"),
                Td(best_bpb_content, style="text-align:right"),
                Td(
                    f"{r['stop_step'] if r['stopped_early'] else r['last_train_step']}/{r['iterations']}",
                    style="text-align:right",
                ),
                Td(fmt_time(r["total_train_time_ms"]), style="text-align:right"),
                Td(int8_content, style="text-align:right"),
                Td(
                    fmt_bytes(r["total_submission_int8_bytes"])
                    if r["total_submission_int8_bytes"]
                    else "—",
                    style="text-align:right",
                ),
                Td(
                    f"{r['payload_ratio']:.2f}x" if r["payload_ratio"] else "—",
                    style="text-align:right",
                ),
                Td(
                    f"{r['peak_mem_allocated']} MiB"
                    if r["peak_mem_allocated"]
                    else "—",
                    style="text-align:right",
                ),
            )
        )

    return Title("Run Explorer"), Main(
        H1("Run Explorer"),
        navbar(),
        P(
            f"{len(runs)} runs loaded from {LOGS_DIR}"
            + (f" | Reference: {ref_run['id'][:8]}" if ref_run else "")
        ),
        Table(
            Thead(
                Tr(
                    Th(""),
                    Th("Run"),
                    Th("Timestamp"),
                    Th("Status"),
                    Th("Params", style="text-align:right"),
                    Th("Final val_bpb", style="text-align:right"),
                    Th("Best val_bpb", style="text-align:right"),
                    Th("Steps", style="text-align:right"),
                    Th("Time", style="text-align:right"),
                    Th("int8+zlib", style="text-align:right"),
                    Th("Total Size", style="text-align:right"),
                    Th("Ratio", style="text-align:right"),
                    Th("Peak Mem", style="text-align:right"),
                )
            ),
            Tbody(*rows),
        ),
        cls="container",
    )


@app.get("/run/{run_id}")
def run_detail(run_id: str):
    filepath = LOGS_DIR / f"{run_id}.txt"
    if not filepath.exists():
        return Title("Not Found"), Main(H1("Run not found"), navbar(), cls="container")

    run = parse_log_file(filepath)
    if not run:
        return Title("Parse Error"), Main(
            H1("Could not parse log"), navbar(), cls="container"
        )

    # Training loss progression (sampled)
    train_rows = []
    for step, total, loss, ms in run["train_steps"]:
        train_rows.append(Tr(Td(str(step)), Td(f"{loss:.4f}"), Td(fmt_time(ms))))

    # Validation progression
    val_rows = []
    for step, total, loss, bpb, ms in run["val_steps"]:
        val_rows.append(
            Tr(
                Td(str(step)),
                Td(f"{loss:.4f}"),
                Td(Span(f"{bpb:.4f}", cls=bpb_class(bpb))),
                Td(fmt_time(ms)),
            )
        )

    loss_chart_link = (
        A("View Loss Chart", href=f"/logs/{run['id']}_loss.html")
        if run["has_loss_html"]
        else None
    )

    return Title(f"Run {run_id[:8]}"), Main(
        H1(f"Run {run_id[:8]}"),
        navbar(),
        P(Code(run_id), " ", loss_chart_link or ""),
        H2("Configuration"),
        Div(
            Div(
                Div(
                    Code(fmt_params(run["model_params"])),
                    P("Model Params", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(Code(f"sp{run['vocab_size']}"), P("Vocab Size", cls="stat-label")),
                cls="stat-card",
            ),
            Div(
                Div(Code(str(run["iterations"])), P("Iterations", cls="stat-label")),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(f"{run['max_wallclock_seconds']:.0f}s"),
                    P("Wallclock Cap", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(str(run["train_batch_tokens"])),
                    P("Batch Tokens", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(Code(str(run["train_seq_len"])), P("Seq Length", cls="stat-label")),
                cls="stat-card",
            ),
            cls="stats-grid",
        ),
        Table(
            Thead(Tr(Th("Property"), Th("Value"))),
            Tbody(
                Tr(Td("Dataset"), Td(run["dataset"])),
                Tr(Td("Train Shards"), Td(str(run["train_shards"]))),
                Tr(Td("Val Tokens"), Td(f"{run['val_tokens']:,}")),
                Tr(
                    Td("Heads (Q/KV)"),
                    Td(f"{run['num_heads']} / {run['num_kv_heads']}"),
                ),
                Tr(Td("Tie Embeddings"), Td(str(run["tie_embeddings"]))),
                Tr(Td("Embed LR"), Td(str(run["embed_lr"]))),
                Tr(Td("Head LR"), Td(str(run["head_lr"]))),
                Tr(Td("Matrix LR"), Td(str(run["matrix_lr"]))),
                Tr(Td("Scalar LR"), Td(str(run["scalar_lr"]))),
                Tr(Td("Warmup Steps"), Td(str(run["warmup_steps"]))),
                Tr(Td("Grad Accum"), Td(str(run["grad_accum_steps"]))),
                Tr(Td("World Size"), Td(str(run["world_size"]))),
                Tr(Td("Seed"), Td(str(run["seed"]))),
                Tr(Td("GPU"), Td(run["gpu"])),
                Tr(Td("PyTorch"), Td(run["pytorch_version"])),
                Tr(Td("Python"), Td(run["python_version"])),
            ),
            cls="detail-table",
        ),
        H2("Results"),
        Div(
            Div(
                Div(
                    Code(
                        Span(
                            f"{run['final_val_bpb']:.4f}"
                            if run["final_val_bpb"]
                            else "—",
                            cls=bpb_class(run["final_val_bpb"]),
                        )
                    ),
                    P("Final val_bpb (int8)", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(
                        f"{run['final_val_loss']:.4f}" if run["final_val_loss"] else "—"
                    ),
                    P("Final val_loss", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(
                        Span(
                            f"{run['best_val_bpb']:.4f}"
                            if run["best_val_bpb"]
                            else "—",
                            cls=bpb_class(run["best_val_bpb"]),
                        )
                    ),
                    P("Best val_bpb", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(
                        fmt_bytes(run["int8_zlib_bytes"])
                        if run["int8_zlib_bytes"]
                        else "—"
                    ),
                    P("int8+zlib Size", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(
                        fmt_bytes(run["total_submission_int8_bytes"])
                        if run["total_submission_int8_bytes"]
                        else "—"
                    ),
                    P("Total Submission", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(
                        f"{run['payload_ratio']:.2f}x" if run["payload_ratio"] else "—"
                    ),
                    P("Compression Ratio", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(
                        f"{run['peak_mem_allocated']} MiB"
                        if run["peak_mem_allocated"]
                        else "—"
                    ),
                    P("Peak GPU Memory", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            Div(
                Div(
                    Code(fmt_time(run["total_train_time_ms"])),
                    P("Total Train Time", cls="stat-label"),
                ),
                cls="stat-card",
            ),
            cls="stats-grid",
        ),
        Table(
            Thead(Tr(Th("Metric"), Th("Value"))),
            Tbody(
                Tr(
                    Td("Serialized Model"), Td(fmt_bytes(run["serialized_model_bytes"]))
                ),
                Tr(Td("Code Size"), Td(fmt_bytes(run["code_size_bytes"]))),
                Tr(Td("Total (raw)"), Td(fmt_bytes(run["total_submission_bytes"]))),
                Tr(Td("int8+zlib (file)"), Td(fmt_bytes(run["int8_zlib_bytes"]))),
                Tr(Td("int8 payload"), Td(fmt_bytes(run["int8_payload_bytes"]))),
                Tr(
                    Td("Total (int8+zlib)"),
                    Td(fmt_bytes(run["total_submission_int8_bytes"])),
                ),
                Tr(
                    Td("Final eval time"),
                    Td(
                        fmt_time(run["final_eval_time_ms"])
                        if run.get("final_eval_time_ms")
                        else "—"
                    ),
                ),
                Tr(
                    Td("Early stop"),
                    Td(
                        f"Yes at step {run['stop_step']} ({fmt_time(run['stop_time_ms'])})"
                        if run["stopped_early"]
                        else "No (completed all iterations)"
                    ),
                ),
            ),
            cls="detail-table",
        ),
        H2("Validation Progress"),
        Table(
            Thead(Tr(Th("Step"), Th("val_loss"), Th("val_bpb"), Th("Wall Time"))),
            Tbody(*val_rows)
            if val_rows
            else Tbody(Tr(Td("No validation data", colspan="4"))),
        ),
        H2("Training Loss (sampled)"),
        Table(
            Thead(Tr(Th("Step"), Th("train_loss"), Th("Wall Time"))),
            Tbody(*train_rows)
            if train_rows
            else Tbody(Tr(Td("No training data", colspan="3"))),
        ),
        cls="container",
    )


@app.get("/logs/{filename}")
def serve_log_file(filename: str):
    filepath = LOGS_DIR / filename
    if filepath.exists() and filepath.suffix == ".html":
        return FileResponse(str(filepath))
    return Title("Not Found"), Main(H1("File not found"), navbar(), cls="container")


if __name__ == "__main__":
    print("Starting Run Explorer...")
    print(f"Logs dir: {LOGS_DIR}")
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs")
    print("\nOpen http://localhost:3000 in your browser")
    serve(port=3000)
