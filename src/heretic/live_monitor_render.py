# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Tail-based renderer for the JSONL files produced by `heretic --live-monitor`.

Reads the header record to learn `n_layers` and per-layer thresholds, then
polls the file for new step records and draws a live diverging heatmap of
per-layer refusal-direction projections (Y = layer, X = step). Cells that
exceed their layer threshold are highlighted with a marker overlay.

Two modes:

    heretic-monitor live  path/to/file.jsonl   # poll & animate
    heretic-monitor still path/to/file.jsonl   # render once, save PNG

Both are also accessible via `python -m heretic.live_monitor_render`.

The renderer is fully decoupled from the model code path: it reads records
written by `LiveMonitor` but does not import heretic's heavyweight ML deps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class MonitorState:
    """Cumulative state assembled from a JSONL file.

    Records are absorbed in arrival order; the arrays grow column-wise as
    each step is observed. The renderer reads from this and draws a heatmap.
    """

    header: dict | None = None
    n_layers: int = 0
    thresholds: list[float] = field(default_factory=list)
    # projections[step][layer], shape grows as new steps arrive.
    projections: list[list[float]] = field(default_factory=list)
    spikes: list[list[bool]] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    token_ids: list[int | None] = field(default_factory=list)
    token_texts: list[str | None] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)

    def ingest(self, record: dict) -> bool:
        """Apply one JSONL record. Returns True if a step was appended."""
        rtype = record.get("type")
        if rtype == "header":
            self.header = record
            self.n_layers = int(record.get("n_layers", 0))
            self.thresholds = list(record.get("thresholds", []))
            return False
        if rtype != "step":
            return False
        self.projections.append(list(record["projections"]))
        self.spikes.append(list(record["spikes"]))
        self.steps.append(int(record.get("step", len(self.steps))))
        self.token_ids.append(record.get("token_id"))
        self.token_texts.append(record.get("token_text"))
        self.stages.append(record.get("stage", "generate"))
        self.turns.append(int(record.get("turn", 1)))
        return True


def tail_jsonl(path: Path, poll_interval: float = 0.1) -> Iterator[dict]:
    """Yield records from a JSONL file as they are appended.

    Starts from the beginning of the file (so the header is always seen),
    then sleeps between attempts when the file pointer reaches EOF. The
    iterator never terminates on its own — callers should break out on
    their own condition (e.g., the matplotlib window being closed).
    """
    pending = ""
    with open(path, "r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read()
            if chunk:
                pending += chunk
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Partial line written between our reads; put it
                        # back at the front of `pending`. This shouldn't
                        # actually happen because we split on \n, but be
                        # defensive against half-flushed records.
                        pending = line + "\n" + pending
                        break
            else:
                time.sleep(poll_interval)


def read_jsonl(path: Path) -> Iterator[dict]:
    """Read all records from a JSONL file in arrival order (one-shot)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "Rendering requires matplotlib and numpy. Install heretic with the "
            'research extra: pip install -U heretic-llm[research]',
            file=sys.stderr,
        )
        raise SystemExit(1)
    return plt, np


def _build_figure(state: MonitorState, window: int, title: str):
    plt, np = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    # Empty placeholder image; will be replaced on first update.
    placeholder = np.zeros((state.n_layers + 1, max(window, 1)), dtype=float)
    im = ax.imshow(
        placeholder,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, label="residual · refusal_direction")
    ax.set_xlabel("Step (token index)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    spike_scatter = ax.scatter([], [], s=14, c="yellow", edgecolors="black", linewidths=0.5)
    return fig, ax, im, cbar, spike_scatter


def _refresh(state: MonitorState, window: int, fig, ax, im, spike_scatter):
    _, np = _import_matplotlib()
    if not state.projections:
        return
    n_layers_plus_one = state.n_layers + 1
    # Take the last `window` steps for the live view.
    start = max(0, len(state.projections) - window)
    cols = state.projections[start:]
    spike_cols = state.spikes[start:]
    step_labels = state.steps[start:]

    # data shape: (n_layers+1, n_cols)
    data = np.asarray(cols, dtype=float).T

    # Symmetric color limits so 0 stays at the colormap midpoint.
    max_abs = float(np.nanmax(np.abs(data))) if data.size else 1.0
    if max_abs <= 0:
        max_abs = 1.0
    im.set_data(data)
    im.set_clim(-max_abs, max_abs)
    im.set_extent((-0.5, data.shape[1] - 0.5, -0.5, n_layers_plus_one - 0.5))

    # Tick labels: a few step indices along the X axis.
    n_cols = data.shape[1]
    n_ticks = min(10, n_cols)
    if n_ticks > 0:
        tick_positions = np.linspace(0, n_cols - 1, n_ticks).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(step_labels[i]) for i in tick_positions])

    # Spike overlay: scatter (col, layer) for every flagged cell.
    xs = []
    ys = []
    for col_idx, spike_col in enumerate(spike_cols):
        for layer_idx, flagged in enumerate(spike_col):
            if flagged:
                xs.append(col_idx)
                ys.append(layer_idx)
    spike_scatter.set_offsets(np.column_stack([xs, ys]) if xs else np.empty((0, 2)))


def render_live(
    path: Path,
    window: int = 200,
    poll_interval: float = 0.1,
    update_interval_ms: int = 200,
):
    """Tail the JSONL and animate a live heatmap."""
    plt, _ = _import_matplotlib()

    state = MonitorState()

    # Seed state with whatever's already in the file (header + any prior steps).
    # The header MUST exist before we can build the figure.
    iterator = tail_jsonl(path, poll_interval=poll_interval)
    # Block until we've read the header.
    for record in iterator:
        state.ingest(record)
        if state.header is not None:
            break

    title = f"Refusal-direction projections — {state.header.get('model', path.name)}"
    fig, ax, im, cbar, spike_scatter = _build_figure(state, window, title)

    # Drain any backlog before showing the figure.
    fig.canvas.draw_idle()

    def step_once():
        # Process all currently buffered records, then return so the GUI can
        # repaint. The iterator will block on sleep(poll_interval) when it
        # runs out of data; we therefore use a non-blocking variant.
        pass

    # Use FuncAnimation to drive periodic updates from the matplotlib event
    # loop, and a generator-based pump that pulls as many records as are
    # available without blocking the GUI.
    from matplotlib.animation import FuncAnimation  # type: ignore

    # Build a non-blocking incremental reader that uses file position rather
    # than the generator above (which blocks on sleep).
    fh = open(path, "r", encoding="utf-8")
    # Skip ahead past records already absorbed (header + initial steps).
    absorbed = 1 + len(state.projections)  # +1 for the header record
    seen = 0
    while seen < absorbed:
        if not fh.readline():
            break
        seen += 1
    pending = {"buf": ""}

    def pump() -> bool:
        chunk = fh.read()
        if not chunk:
            return False
        pending["buf"] += chunk
        changed = False
        while "\n" in pending["buf"]:
            line, pending["buf"] = pending["buf"].split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                pending["buf"] = line + "\n" + pending["buf"]
                break
            if state.ingest(record):
                changed = True
        return changed

    def update(_frame):
        if pump():
            _refresh(state, window, fig, ax, im, spike_scatter)
        return (im, spike_scatter)

    # Render the initial state immediately.
    _refresh(state, window, fig, ax, im, spike_scatter)

    anim = FuncAnimation(
        fig,
        update,
        interval=update_interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    # Keep a reference so the animation isn't garbage-collected.
    fig._heretic_anim = anim  # type: ignore[attr-defined]

    print(f"Watching {path}. Close the matplotlib window to exit.", file=sys.stderr)
    plt.show()
    fh.close()


def render_still(path: Path, output: Path | None, window: int | None = None):
    """Read the whole JSONL and produce a single PNG of the heatmap."""
    plt, _ = _import_matplotlib()

    state = MonitorState()
    for record in read_jsonl(path):
        state.ingest(record)

    if state.header is None:
        print(f"No header record found in {path}", file=sys.stderr)
        raise SystemExit(1)
    if not state.projections:
        print(f"No step records found in {path}", file=sys.stderr)
        raise SystemExit(1)

    effective_window = window if window is not None else len(state.projections)
    title = f"Refusal-direction projections — {state.header.get('model', path.name)}"
    fig, ax, im, cbar, spike_scatter = _build_figure(state, effective_window, title)
    _refresh(state, effective_window, fig, ax, im, spike_scatter)

    if output is None:
        output = path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    print(f"Wrote {output}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heretic-monitor",
        description=(
            "Render the JSONL output of `heretic --live-monitor` as a 2D "
            "(layer x step) heatmap of refusal-direction projections."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_live = sub.add_parser(
        "live",
        help="Tail a JSONL file and update an animated matplotlib heatmap.",
    )
    p_live.add_argument("path", type=Path)
    p_live.add_argument(
        "--window",
        type=int,
        default=200,
        help="Number of most recent steps to display in the rolling view.",
    )
    p_live.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Seconds between file polls when no new data is available.",
    )
    p_live.add_argument(
        "--update-interval-ms",
        type=int,
        default=200,
        help="Matplotlib repaint interval in milliseconds.",
    )

    p_still = sub.add_parser(
        "still",
        help="Render a complete JSONL file as a single PNG.",
    )
    p_still.add_argument("path", type=Path)
    p_still.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <path>.png.",
    )
    p_still.add_argument(
        "--window",
        type=int,
        default=None,
        help="If set, only render the last N steps.",
    )

    args = parser.parse_args(argv)

    if args.cmd == "live":
        render_live(
            args.path,
            window=args.window,
            poll_interval=args.poll_interval,
            update_interval_ms=args.update_interval_ms,
        )
    elif args.cmd == "still":
        render_still(args.path, output=args.output, window=args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
