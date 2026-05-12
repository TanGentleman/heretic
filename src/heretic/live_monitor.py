# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Inference-time monitor that streams per-layer projections of the residual
stream onto each layer's cached "refusal direction" to a JSONL file.

Each forward pass through the transformer emits one JSONL record per token
position seen in that pass (prefill produces one record per prompt token,
generation produces one record per newly generated token). A record looks
like:

    {
      "type": "step",
      "step": 42,
      "stage": "generate",
      "token_id": 1234,
      "projections": [..., per-layer floats, length n_layers+1],
      "norms":       [..., per-layer L2 norms of the residual],
      "spikes":      [..., per-layer booleans, True when projection > threshold],
      "spike_count": <int>,
      "max_spike_layer": <int|null>,
      "max_spike_value": <float|null>
    }

A single header record is written first with metadata required to interpret
the projections (per-layer thresholds, baseline good/bad projections, layer
count). Downstream renderers can tail the file and plot a (layer x step)
heatmap of `projections`, highlighting cells where `spikes[layer]` is True.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, TextIO

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


class LiveMonitor:
    def __init__(
        self,
        layers: list[Module],
        refusal_directions: Tensor,
        good_means: Tensor,
        bad_means: Tensor,
        output_path: str | Path,
        threshold_multiplier: float = 1.0,
        embed_tokens: Module | None = None,
        tokenizer: Any = None,
        model_name: str | None = None,
        on_step: Callable[[dict], None] | None = None,
    ):
        """
        Args:
            layers: Ordered list of transformer blocks (as returned by
                Model.get_layers()).
            refusal_directions: Unit-normalized per-layer refusal directions,
                shape (n_layers+1, d_model). Index 0 corresponds to the
                embedding output, indices 1..n_layers to the output of
                transformer block i.
            good_means, bad_means: Mean residual vectors for harmless/harmful
                prompts, shape (n_layers+1, d_model). Used to derive
                per-layer projection thresholds.
            output_path: JSONL file to stream records to. Overwritten if it
                exists.
            threshold_multiplier: Scales the default per-layer threshold
                (=projection of bad_means onto the refusal direction). A token
                is flagged on layer L when its projection exceeds
                thresholds[L].
            embed_tokens: Optional embedding module. If provided, the monitor
                hooks it to capture per-step input token IDs, which are then
                written to the JSONL records.
            tokenizer: Optional HF tokenizer. If provided, each record will
                also include a decoded `token_text` string for the captured
                token ID.
            model_name: Optional model identifier written into the header
                record for downstream context.
            on_step: Optional callback invoked with each step record after it
                is written. Lets the caller react to projections in real time
                (e.g. render a per-token sparkline next to the chat) without
                re-parsing the JSONL file.
        """
        if refusal_directions.dim() != 2:
            raise ValueError(
                "refusal_directions must have shape (n_layers+1, d_model)"
            )
        if refusal_directions.shape[0] != len(layers) + 1:
            raise ValueError(
                f"refusal_directions has {refusal_directions.shape[0]} entries "
                f"but model has {len(layers)} layers (expected {len(layers) + 1})"
            )

        self.layers = list(layers)
        self.n_layers = len(self.layers)
        self.on_step = on_step
        self.refusal_directions = refusal_directions.detach().to(torch.float32)
        self.embed_tokens = embed_tokens
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.threshold_multiplier = float(threshold_multiplier)

        good_means_f32 = good_means.detach().to(torch.float32)
        bad_means_f32 = bad_means.detach().to(torch.float32)
        # Per-layer projection of the dataset means onto the unit refusal direction.
        good_proj = (good_means_f32 * self.refusal_directions).sum(dim=-1)
        bad_proj = (bad_means_f32 * self.refusal_directions).sum(dim=-1)
        self.good_proj = good_proj.tolist()
        self.bad_proj = bad_proj.tolist()
        # A token's residual on layer L is "harmful-like" if it projects onto
        # the refusal direction at least as strongly as the average bad prompt.
        self.thresholds = [p * self.threshold_multiplier for p in self.bad_proj]

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._fh: TextIO | None = None
        self._handles: list[RemovableHandle] = []
        # captured[layer_idx] = (batch, seq, d_model) tensor for the current
        # forward pass. We collect all of them, then flush once we know the
        # pass is complete (post-hook on the final layer fires last).
        self._captured: dict[int, Tensor] = {}
        # Token IDs for the positions captured in the current forward pass.
        # Length matches the seq dim of the captured tensors. None if the
        # embed_tokens hook was not registered or failed to capture.
        self._pending_token_ids: list[int] | None = None
        self._step: int = 0
        # Turn counter: increments on each new prefill burst (or the first
        # generate step of a session if the user manages to skip prefill).
        # Lets the renderer separate distinct chat turns.
        self._turn: int = 0
        self._last_stage: str | None = None

    def __enter__(self) -> "LiveMonitor":
        # Preserve any prior run at this path. Each session has different
        # calibration data (model, thresholds, refusal direction) so appending
        # would corrupt downstream renderers; rotating instead keeps every run.
        if self.output_path.exists():
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = self.output_path.with_suffix(
                self.output_path.suffix + f".{ts}.bak"
            )
            self.output_path.rename(backup)
        self._fh = open(self.output_path, "w", encoding="utf-8")
        header = {
            "type": "header",
            "model": self.model_name,
            "n_layers": self.n_layers,
            "good_projections": [round(x, 6) for x in self.good_proj],
            "bad_projections": [round(x, 6) for x in self.bad_proj],
            "thresholds": [round(x, 6) for x in self.thresholds],
            "threshold_multiplier": self.threshold_multiplier,
        }
        self._fh.write(json.dumps(header) + "\n")
        self._fh.flush()
        self._attach_hooks()
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()
        self._pending_token_ids = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _attach_hooks(self) -> None:
        # Pre-hook on layer i captures its input, which is:
        #   * the embedding output for i = 0
        #   * the output of layer i-1 for i > 0
        # Combined with a post-hook on the last layer, this gives us all
        # n_layers+1 residual snapshots that align with refusal_directions.
        for i, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_pre_hook(
                    self._make_pre_hook(i), with_kwargs=True
                )
            )
        self._handles.append(
            self.layers[-1].register_forward_hook(
                self._make_post_hook(self.n_layers), with_kwargs=True
            )
        )

        if self.embed_tokens is not None:
            self._handles.append(
                self.embed_tokens.register_forward_pre_hook(
                    self._embed_pre_hook, with_kwargs=True
                )
            )

    @staticmethod
    def _extract_hidden_state(args: tuple, kwargs: dict | None) -> Tensor | None:
        # Most HF transformer blocks receive the hidden state as the first
        # positional argument, but some custom models pass it via keyword
        # ("hidden_states="). Cover both cases.
        if args:
            candidate = args[0]
            if isinstance(candidate, Tensor):
                return candidate
        if kwargs:
            for key in ("hidden_states", "x", "inputs_embeds"):
                candidate = kwargs.get(key)
                if isinstance(candidate, Tensor):
                    return candidate
        return None

    def _embed_pre_hook(
        self, module: Module, args: tuple, kwargs: dict
    ) -> None:
        input_ids = None
        if args and isinstance(args[0], Tensor):
            input_ids = args[0]
        elif kwargs:
            for key in ("input", "input_ids"):
                candidate = kwargs.get(key)
                if isinstance(candidate, Tensor):
                    input_ids = candidate
                    break
        if input_ids is None or input_ids.dim() < 1:
            return
        flat = input_ids.detach()
        if flat.dim() == 2:
            flat = flat[0]
        self._pending_token_ids = [int(x) for x in flat.cpu().tolist()]

    def _make_pre_hook(self, layer_idx: int):
        def hook(module: Module, args: tuple, kwargs: dict) -> None:
            hidden = self._extract_hidden_state(args, kwargs)
            if hidden is None:
                return
            self._captured[layer_idx] = hidden.detach()

        return hook

    def _make_post_hook(self, layer_idx: int):
        def hook(module: Module, args: tuple, kwargs: dict, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, Tensor):
                return
            self._captured[layer_idx] = hidden.detach()
            # The post-hook on the final layer fires last for this forward
            # pass, so this is our flush boundary.
            self._flush()

        return hook

    def _flush(self) -> None:
        if self._fh is None:
            return
        if len(self._captured) < self.n_layers + 1:
            # Some hook didn't fire (unexpected module type, conditional
            # routing, etc.). Drop this pass rather than emit a partial row.
            self._captured.clear()
            self._pending_token_ids = None
            return

        try:
            stacked = torch.stack(
                [self._captured[i][0] for i in range(self.n_layers + 1)],
                dim=0,
            )
        except (IndexError, RuntimeError):
            self._captured.clear()
            self._pending_token_ids = None
            return

        # stacked: (n_layers+1, seq, d_model)
        stacked = stacked.to(torch.float32)
        dirs = self.refusal_directions.to(stacked.device)
        # projections[l, s] = <residual[l, s], refusal_direction[l]>
        projections = torch.einsum("lsd,ld->ls", stacked, dirs)
        norms = stacked.norm(dim=-1)

        projections_np = projections.cpu().tolist()
        norms_np = norms.cpu().tolist()
        seq_len = len(projections_np[0]) if projections_np else 0

        token_ids = self._pending_token_ids
        # Tag prefill (initial multi-position forward pass) vs. generate
        # (single-position with KV cache). Streamed prefill in chunks would
        # also be reported as "prefill" because seq_len > 1.
        stage = "prefill" if seq_len > 1 else "generate"

        # A new prefill burst marks a new chat turn. Generate steps after
        # the first prefill inherit the current turn counter.
        if stage == "prefill" and self._last_stage != "prefill":
            self._turn += 1
        elif stage == "generate" and self._last_stage is None:
            # Edge case: somehow we started with generation (no prefill).
            self._turn += 1
        self._last_stage = stage

        for p in range(seq_len):
            proj_col = [projections_np[l][p] for l in range(self.n_layers + 1)]
            norm_col = [norms_np[l][p] for l in range(self.n_layers + 1)]
            spikes = [proj_col[l] > self.thresholds[l] for l in range(self.n_layers + 1)]
            spike_count = sum(spikes)

            max_spike_layer: int | None = None
            max_spike_value: float | None = None
            for l in range(self.n_layers + 1):
                margin = proj_col[l] - self.thresholds[l]
                if margin > 0 and (max_spike_value is None or margin > max_spike_value):
                    max_spike_value = margin
                    max_spike_layer = l

            record: dict[str, Any] = {
                "type": "step",
                "step": self._step,
                "turn": self._turn,
                "stage": stage,
                "projections": [round(x, 4) for x in proj_col],
                "norms": [round(x, 3) for x in norm_col],
                "spikes": spikes,
                "spike_count": spike_count,
                "max_spike_layer": max_spike_layer,
                "max_spike_value": (
                    round(max_spike_value, 4) if max_spike_value is not None else None
                ),
            }
            if token_ids is not None and p < len(token_ids):
                token_id = token_ids[p]
                record["token_id"] = token_id
                if self.tokenizer is not None:
                    try:
                        record["token_text"] = self.tokenizer.decode(
                            [token_id], skip_special_tokens=False
                        )
                    except Exception:
                        pass
            self._fh.write(json.dumps(record) + "\n")
            self._step += 1
            if self.on_step is not None:
                try:
                    self.on_step(record)
                except Exception:
                    # A misbehaving callback must not break the inference loop
                    # or corrupt the JSONL stream.
                    pass

        self._fh.flush()
        self._captured.clear()
        self._pending_token_ids = None
