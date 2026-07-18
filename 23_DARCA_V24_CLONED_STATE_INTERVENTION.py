#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_DARCA_V24_CLONED_STATE_INTERVENTION.py
=========================================
One-step cloned-state intervention for the DARCA causal-direction pathway.

Purpose
-------
The closed-loop battery establishes that output-side manipulation of d(t)
changes later behaviour, but trajectories diverge after the manipulated action.
This experiment removes that downstream ambiguity. At prespecified scored steps,
the intact driver agent is cloned before it receives the current observation.
Every clone therefore starts from the same complete agent state, the same current
observation, and identical RNG states. The clones differ only in the value of
causal direction transmitted at the point of use:

    intact / zero / flip / shuffle / time_shuffle

Each clone executes exactly one agent step. The environment is not advanced for
counterfactual clones. The intact clone alone continues the driver trajectory.

Sampling and inference
----------------------
- Tasks: current_sufficient, history_required, hidden_reversal, and
  action_independent_null, with the same phases as the strict main battery.
- Regimes: 3 delays x 2 noise levels x 2 couplings = 12.
- Base seeds: 32, with independently shuffled and exactly balanced hidden rules.
- Clone states: every 10th scored step, fixed before execution.
- Statistical unit: base seed. Regimes and clone states are aggregated within
  seed before bootstrap intervals or sign-flip tests.

Primary mechanistic estimand
----------------------------
Among cloned states in which the intact clone emits a trusted directed probe,
Direction_Flip should retain the probe class and emit the opposite probe. The
experiment also verifies that raw d(t), C(t), and all non-directional utilities
are invariant across clones, and that the directed probe utilities are swapped.

Outputs
-------
01_preflight_audit.csv
02_clone_events.csv
03_seed_level_clone_metrics.csv
04_clone_summary.csv
05_execution_report.md
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Keep small matrix-vector operations single-threaded in each worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ANALYSIS_SEED = 20260718
MODES: List[Tuple[str, str]] = [
    ("Full", "intact"),
    ("Direction_Zero", "zero"),
    ("Direction_Flip", "flip"),
    ("Direction_Shuffle", "shuffle"),
    ("Direction_TimeShuffle", "time_shuffle"),
]
TASKS = ["current_sufficient", "history_required", "hidden_reversal", "action_independent_null"]
DELAYS = [4, 8, 12]
NOISES = [0.035, 0.060]
COUPLINGS = [0.45, 0.80]
REGIMES = [(d, n, c) for d in DELAYS for n in NOISES for c in COUPLINGS]

ACQUISITION_STEPS = 300
EVALUATION_STEPS = 200
PRE_REVERSAL_STEPS = 150
POST_REVERSAL_STEPS = 200
SAMPLE_STRIDE = 10
DIRECTION_EPS = 1e-6
BOOTSTRAP_RESAMPLES = 5000
PERMUTATIONS = 20000

TASK_CODE = {name: i + 1 for i, name in enumerate(TASKS)}


def load_core(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location("_darca_clone_core", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import core: {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    required = [
        "Params", "Condition", "Agent", "build_conditions", "action_effect",
        "A_PROBE_PLUS", "A_PROBE_MINUS",
    ]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise RuntimeError(f"core missing required symbols: {missing}")
    return mod


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0xFFFFFFFF


def balanced_rule_assignment(n_seeds: int) -> List[int]:
    if n_seeds % 2:
        raise ValueError("n_seeds must be even for exact rule balance")
    rules = np.array([-1] * (n_seeds // 2) + [1] * (n_seeds // 2), dtype=int)
    rng = np.random.default_rng(stable_seed(ANALYSIS_SEED, "balanced_rule_assignment", n_seeds))
    rng.shuffle(rules)
    return rules.tolist()


@dataclass(frozen=True)
class Phase:
    steps: int
    rule: int
    feedback: bool
    action_independent: bool
    scored: bool
    delay: int
    noise: float
    coupling: float
    tag: str
    reset_buffer: bool = False


def task_phases(task: str, rule: int, regime: Tuple[int, float, float]) -> List[Phase]:
    d, n, c = regime
    if task == "current_sufficient":
        return [Phase(ACQUISITION_STEPS + EVALUATION_STEPS, rule, True, False, True, d, n, c, "current")]
    if task == "history_required":
        return [
            Phase(ACQUISITION_STEPS, rule, True, False, False, d, n, c, "acquisition"),
            Phase(EVALUATION_STEPS, rule, False, False, True, d, n, c, "history", reset_buffer=True),
        ]
    if task == "hidden_reversal":
        return [
            Phase(ACQUISITION_STEPS, rule, True, False, False, d, n, c, "acquisition"),
            Phase(PRE_REVERSAL_STEPS, rule, True, False, True, d, n, c, "pre"),
            Phase(POST_REVERSAL_STEPS, -rule, True, False, True, d, n, c, "post"),
        ]
    if task == "action_independent_null":
        return [Phase(ACQUISITION_STEPS + EVALUATION_STEPS, rule, True, True, True, d, n, c, "null")]
    raise ValueError(task)


def new_buffer(delay: int, noise: float, rng: np.random.Generator) -> deque[float]:
    return deque((float(rng.normal(0.0, noise)) for _ in range(delay)), maxlen=delay)


def mode_agent(base_agent: Any, name: str, mode: str) -> Any:
    clone = copy.deepcopy(base_agent)
    clone.condition = replace(clone.condition, name=name, direction_mode=mode)
    return clone


def action_sign(mod: Any, action_id: int) -> int:
    if action_id == mod.A_PROBE_PLUS:
        return 1
    if action_id == mod.A_PROBE_MINUS:
        return -1
    return 0


def finite_max_abs(values: Iterable[float]) -> float:
    vals = [abs(float(x)) for x in values if math.isfinite(float(x))]
    return max(vals) if vals else float("nan")


def evaluate_clones(mod: Any, base_agent: Any, y: float, rule: int, task: str,
                    phase_tag: str, phase_i: int, global_i: int, base_seed: int,
                    regime_index: int, regime: Tuple[int, float, float]) -> Tuple[Dict[str, Any], Any]:
    env_info = {"external_shock": 0.0, "d_dyn": 0.0, "coupling_t": 0.0, "sigma_t": 0.0}
    outputs: Dict[str, Dict[str, Any]] = {}
    agents: Dict[str, Any] = {}
    for name, mode in MODES:
        clone = mode_agent(base_agent, name, mode)
        if mode in ("shuffle", "time_shuffle"):
            # The factual Full trajectory never consumes the intervention RNG.
            # Give each randomized counterfactual a separate, event-keyed stream
            # so repeated clone states receive independent but exactly
            # reproducible manipulations without perturbing the core RNG.
            clone.shuffle_rng = np.random.default_rng(
                stable_seed(ANALYSIS_SEED, "clone_intervention", name, base_seed, task, regime_index, global_i)
            )
        out = clone.step(y, env_info)
        outputs[name] = out
        agents[name] = clone

    full = outputs["Full"]
    raw_d = float(full["causal_direction"])
    conf = float(full["causal_confidence"])
    full_action = int(full["action_id"])
    full_sign = action_sign(mod, full_action)
    full_is_probe = int(full_sign != 0)
    trusted_direction = int(conf >= base_agent.p.probe_direction_threshold and abs(raw_d) > DIRECTION_EPS)
    full_trusted_probe = int(full_is_probe and trusted_direction)

    row: Dict[str, Any] = {
        "task": task,
        "phase": phase_tag,
        "base_seed": base_seed,
        "rule": rule,
        "regime_index": regime_index,
        "regime_id": f"d{regime[0]}_n{regime[1]:.3f}_c{regime[2]:.2f}",
        "delay": regime[0],
        "noise": regime[1],
        "coupling": regime[2],
        "phase_step": phase_i,
        "global_step": global_i,
        "observation": y,
        "full_raw_direction": raw_d,
        "full_confidence": conf,
        "trusted_direction": trusted_direction,
        "full_is_probe": full_is_probe,
        "full_trusted_probe": full_trusted_probe,
        "full_probe_sign": full_sign,
        "full_probe_correct": int(full_sign == rule) if full_is_probe else 0,
    }

    nondirectional_keys = ["U_REGULATE", "U_PROBE", "U_INHIBIT", "U_EXPRESS"]
    raw_diffs: List[float] = []
    confidence_diffs: List[float] = []
    nondir_diffs: List[float] = []

    for name, _mode in MODES:
        out = outputs[name]
        aid = int(out["action_id"])
        sign = action_sign(mod, aid)
        d_used = float(out["causal_direction_used"])
        row[f"{name}_action_id"] = aid
        row[f"{name}_action_name"] = str(out["action_name"])
        row[f"{name}_probe_sign"] = sign
        row[f"{name}_is_probe"] = int(sign != 0)
        row[f"{name}_d_used"] = d_used
        row[f"{name}_directional_evidence_used"] = float(out["directional_evidence_used"])
        row[f"{name}_directional_drive"] = float(out["directional_drive"])
        for key in ["U_REGULATE", "U_PROBE", "U_PROBE_PLUS", "U_PROBE_MINUS", "U_INHIBIT", "U_EXPRESS"]:
            row[f"{name}_{key}"] = float(out[key])
        row[f"{name}_action_match_full"] = int(aid == full_action)
        row[f"{name}_action_switch_from_full"] = int(aid != full_action)
        row[f"{name}_probe_class_match_full"] = int((sign != 0) == (full_sign != 0))
        row[f"{name}_same_probe_as_full"] = int(full_sign != 0 and sign == full_sign)
        row[f"{name}_opposite_probe_from_full"] = int(full_sign != 0 and sign == -full_sign)
        row[f"{name}_nonprobe_from_full_probe"] = int(full_sign != 0 and sign == 0)

        raw_diffs.append(float(out["causal_direction"]) - raw_d)
        confidence_diffs.append(float(out["causal_confidence"]) - conf)
        nondir_diffs.extend(float(out[k]) - float(full[k]) for k in nondirectional_keys)

    # Exact pathway audit quantities.
    row["max_abs_raw_direction_diff"] = finite_max_abs(raw_diffs)
    row["max_abs_confidence_diff"] = finite_max_abs(confidence_diffs)
    row["max_abs_nondirectional_utility_diff"] = finite_max_abs(nondir_diffs)
    row["flip_probe_utility_swap_error"] = max(
        abs(float(outputs["Direction_Flip"]["U_PROBE_PLUS"]) - float(full["U_PROBE_MINUS"])),
        abs(float(outputs["Direction_Flip"]["U_PROBE_MINUS"]) - float(full["U_PROBE_PLUS"])),
    )
    row["zero_probe_utility_tie_error"] = abs(
        float(outputs["Direction_Zero"]["U_PROBE_PLUS"]) - float(outputs["Direction_Zero"]["U_PROBE_MINUS"])
    )
    sh = outputs["Direction_Shuffle"]
    shuffle_same_err = max(
        abs(float(sh["U_PROBE_PLUS"]) - float(full["U_PROBE_PLUS"])),
        abs(float(sh["U_PROBE_MINUS"]) - float(full["U_PROBE_MINUS"])),
    )
    shuffle_swap_err = max(
        abs(float(sh["U_PROBE_PLUS"]) - float(full["U_PROBE_MINUS"])),
        abs(float(sh["U_PROBE_MINUS"]) - float(full["U_PROBE_PLUS"])),
    )
    row["shuffle_same_or_swap_error"] = min(shuffle_same_err, shuffle_swap_err)
    row["shuffle_sign_flipped"] = int(
        abs(raw_d) > DIRECTION_EPS
        and float(sh["causal_direction_used"]) * raw_d < 0.0
    )

    # The Full clone alone continues the factual trajectory.
    return row, agents["Full"]


def run_trajectory(mod: Any, task: str, base_seed: int, rule: int,
                   regime_index: int, regime: Tuple[int, float, float]) -> List[Dict[str, Any]]:
    full_condition = next(c for c in mod.build_conditions("all") if c.name == "Full")
    driver = mod.Agent(mod.Params(), full_condition, base_seed)
    env_seed = stable_seed(ANALYSIS_SEED, "environment", base_seed, TASK_CODE[task], regime_index)
    rng = np.random.default_rng(env_seed)
    buffer: Optional[deque[float]] = None
    current_delay: Optional[int] = None
    rows: List[Dict[str, Any]] = []
    global_i = 0

    for phase in task_phases(task, rule, regime):
        if buffer is None or phase.reset_buffer or current_delay != phase.delay:
            buffer = new_buffer(phase.delay, phase.noise, rng)
            current_delay = phase.delay
        for phase_i in range(phase.steps):
            assert buffer is not None
            y = float(buffer.popleft())
            should_clone = bool(phase.scored and phase_i % SAMPLE_STRIDE == 0)
            if should_clone:
                row, driver = evaluate_clones(
                    mod, driver, y, phase.rule, task, phase.tag, phase_i, global_i,
                    base_seed, regime_index, regime,
                )
                rows.append(row)
                out = {
                    "action_id": int(row["Full_action_id"]),
                }
            else:
                out = driver.step(y, {"external_shock": 0.0, "d_dyn": 0.0, "coupling_t": 0.0, "sigma_t": 0.0})

            action_id = int(out["action_id"])
            u = float(mod.action_effect(action_id, y))
            noise = float(rng.normal(0.0, phase.noise))
            if phase.action_independent or not phase.feedback:
                y_future = noise
            else:
                y_future = phase.rule * u * phase.coupling + noise
            buffer.append(float(y_future))
            global_i += 1
    return rows


_WMOD = None


def init_worker(core_path: str) -> None:
    global _WMOD
    _WMOD = load_core(core_path)


def seed_worker(payload: Tuple[int, int, List[Tuple[int, float, float]], List[str]]) -> List[Dict[str, Any]]:
    base_seed, rule, regimes, tasks = payload
    if _WMOD is None:
        raise RuntimeError("worker core not initialized")
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        for regime_index, regime in enumerate(regimes):
            rows.extend(run_trajectory(_WMOD, task, base_seed, rule, regime_index, regime))
    return rows


def mean_ci(values: Sequence[float], label: str, null: Optional[float] = None) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return {"n": n, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(stable_seed(ANALYSIS_SEED, "analysis", label))
    boots = x[rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float("nan")
    if null is not None:
        d = x - null
        signs = rng.choice(np.array([-1.0, 1.0]), size=(PERMUTATIONS, n))
        perm = np.mean(signs * d, axis=1)
        extreme = int(np.sum(np.abs(perm) >= abs(float(np.mean(d))) - 1e-15))
        p = (extreme + 1.0) / (PERMUTATIONS + 1.0)
    return {"n": n, "mean": float(np.mean(x)), "ci95_low": float(lo), "ci95_high": float(hi), "p": p}


def aggregate_seed_metrics(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for r in events:
        groups[(str(r["task"]), str(r["phase"]), int(r["base_seed"]))].append(r)

    rows: List[Dict[str, Any]] = []
    for (task, phase, seed), group in sorted(groups.items()):
        base: Dict[str, Any] = {
            "task": task,
            "phase": phase,
            "base_seed": seed,
            "clone_state_count": len(group),
            "trusted_direction_state_count": sum(int(r["trusted_direction"]) for r in group),
            "full_probe_state_count": sum(int(r["full_is_probe"]) for r in group),
            "full_trusted_probe_state_count": sum(int(r["full_trusted_probe"]) for r in group),
            "max_abs_raw_direction_diff": max(float(r["max_abs_raw_direction_diff"]) for r in group),
            "max_abs_confidence_diff": max(float(r["max_abs_confidence_diff"]) for r in group),
            "max_abs_nondirectional_utility_diff": max(float(r["max_abs_nondirectional_utility_diff"]) for r in group),
            "max_flip_probe_utility_swap_error": max(float(r["flip_probe_utility_swap_error"]) for r in group),
            "max_zero_probe_utility_tie_error": max(float(r["zero_probe_utility_tie_error"]) for r in group),
            "max_shuffle_same_or_swap_error": max(float(r["shuffle_same_or_swap_error"]) for r in group),
        }
        trusted_probe_group = [r for r in group if int(r["full_trusted_probe"]) == 1]
        trusted_direction_group = [r for r in group if int(r["trusted_direction"]) == 1]

        def rate(rows_in: Sequence[Mapping[str, Any]], key: str) -> float:
            return float(np.mean([float(r[key]) for r in rows_in])) if rows_in else float("nan")

        for name, _ in MODES:
            base[f"{name}_action_match_full_rate_all"] = rate(group, f"{name}_action_match_full")
            base[f"{name}_action_switch_rate_all"] = rate(group, f"{name}_action_switch_from_full")
            base[f"{name}_probe_class_match_full_rate_all"] = rate(group, f"{name}_probe_class_match_full")
            base[f"{name}_same_probe_rate_given_full_trusted_probe"] = rate(trusted_probe_group, f"{name}_same_probe_as_full")
            base[f"{name}_opposite_probe_rate_given_full_trusted_probe"] = rate(trusted_probe_group, f"{name}_opposite_probe_from_full")
            base[f"{name}_nonprobe_rate_given_full_trusted_probe"] = rate(trusted_probe_group, f"{name}_nonprobe_from_full_probe")
            base[f"{name}_probe_class_retention_given_full_trusted_probe"] = rate(trusted_probe_group, f"{name}_is_probe")
        base["Direction_Shuffle_sign_flip_rate_given_trusted_direction"] = rate(trusted_direction_group, "shuffle_sign_flipped")
        rows.append(base)
    return rows


def summarize_seed_metrics(seed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in seed_rows:
        grouped[(str(r["task"]), str(r["phase"]))].append(r)

    summaries: List[Dict[str, Any]] = []
    metrics = [
        "Direction_Flip_opposite_probe_rate_given_full_trusted_probe",
        "Direction_Flip_probe_class_retention_given_full_trusted_probe",
        "Direction_Zero_same_probe_rate_given_full_trusted_probe",
        "Direction_Zero_opposite_probe_rate_given_full_trusted_probe",
        "Direction_Zero_nonprobe_rate_given_full_trusted_probe",
        "Direction_Shuffle_same_probe_rate_given_full_trusted_probe",
        "Direction_Shuffle_opposite_probe_rate_given_full_trusted_probe",
        "Direction_Shuffle_nonprobe_rate_given_full_trusted_probe",
        "Direction_Shuffle_sign_flip_rate_given_trusted_direction",
        "Direction_TimeShuffle_same_probe_rate_given_full_trusted_probe",
        "Direction_TimeShuffle_opposite_probe_rate_given_full_trusted_probe",
        "Direction_TimeShuffle_nonprobe_rate_given_full_trusted_probe",
        "Direction_Zero_action_switch_rate_all",
        "Direction_Flip_action_switch_rate_all",
        "Direction_Shuffle_action_switch_rate_all",
        "Direction_TimeShuffle_action_switch_rate_all",
    ]
    null_half = {
        "Direction_Shuffle_sign_flip_rate_given_trusted_direction",
    }
    for (task, phase), rows in sorted(grouped.items()):
        for metric in metrics:
            values = [float(r.get(metric, float("nan"))) for r in rows]
            st = mean_ci(values, f"{task}|{phase}|{metric}", null=0.5 if metric in null_half else None)
            summaries.append({"task": task, "phase": phase, "metric": metric, **st})

    # Global exact-audit maxima are included as rows rather than inferential tests.
    audit_metrics = [
        "max_abs_raw_direction_diff",
        "max_abs_confidence_diff",
        "max_abs_nondirectional_utility_diff",
        "max_flip_probe_utility_swap_error",
        "max_zero_probe_utility_tie_error",
        "max_shuffle_same_or_swap_error",
    ]
    for metric in audit_metrics:
        vals = [float(r[metric]) for r in seed_rows]
        summaries.append({
            "task": "ALL", "phase": "ALL", "metric": metric,
            "n": len(vals), "mean": float(np.mean(vals)),
            "ci95_low": float("nan"), "ci95_high": float("nan"),
            "p": float("nan"), "maximum": max(vals),
        })
    return summaries


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def preflight(mod: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cond = next(c for c in mod.build_conditions("all") if c.name == "Full")
    agent = mod.Agent(mod.Params(), cond, 17)
    # Build a nontrivial history before cloning.
    for i in range(160):
        y = 0.3 * math.sin(i * 0.07)
        agent.step(y, {"external_shock": 0.0, "d_dyn": 0.0, "coupling_t": 0.0, "sigma_t": 0.0})
    event, _ = evaluate_clones(mod, agent, 0.17, +1, "preflight", "preflight", 0, 0, 17, 0, REGIMES[0])
    checks = {
        "raw_direction_invariant": float(event["max_abs_raw_direction_diff"]) <= 1e-12,
        "confidence_invariant": float(event["max_abs_confidence_diff"]) <= 1e-12,
        "nondirectional_utilities_invariant": float(event["max_abs_nondirectional_utility_diff"]) <= 1e-12,
        "flip_swaps_probe_utilities": float(event["flip_probe_utility_swap_error"]) <= 1e-12,
        "zero_ties_probe_utilities": float(event["zero_probe_utility_tie_error"]) <= 1e-12,
        "shuffle_same_or_swapped": float(event["shuffle_same_or_swap_error"]) <= 1e-12,
    }
    for name, passed in checks.items():
        if not passed:
            raise RuntimeError(f"preflight failed: {name}")
        rows.append({"audit": name, "status": "PASS", "detail": "tolerance <= 1e-12"})
    rows.append({"audit": "sampling_schedule", "status": "PASS", "detail": f"every {SAMPLE_STRIDE}th scored step"})
    rows.append({"audit": "counterfactual_environment", "status": "PASS", "detail": "environment not advanced for counterfactual clones; Full clone continues factual trajectory"})
    rows.append({"audit": "statistical_unit", "status": "PASS", "detail": "base seed; regimes and clone states aggregated within seed"})
    return rows


def report_text(events: Sequence[Mapping[str, Any]], seed_rows: Sequence[Mapping[str, Any]],
                summaries: Sequence[Mapping[str, Any]], elapsed: float, n_seeds: int,
                core_path: str) -> str:
    by_key = {(str(r["task"]), str(r["phase"]), str(r["metric"])): r for r in summaries}

    def fmt(task: str, phase: str, metric: str) -> str:
        r = by_key.get((task, phase, metric))
        if not r or not math.isfinite(float(r.get("mean", float("nan")))):
            return "NA"
        return f"{float(r['mean']):.3f} [{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}]"

    lines = [
        "# DARCA v2.4 cloned-state intervention report",
        "",
        "## Fixed design",
        "",
        f"- Core: `{Path(core_path).name}`",
        f"- Base seeds: {n_seeds}",
        f"- Regimes: {len(REGIMES)}",
        f"- Tasks: {len(TASKS)}",
        f"- Sampling: every {SAMPLE_STRIDE}th scored step",
        f"- Cloned pre-action states: {len(events):,}",
        f"- One-step clone evaluations: {len(events) * len(MODES):,}",
        f"- Elapsed: {elapsed:.1f} s",
        "",
        "## Exact intervention audit",
        "",
    ]
    audit_names = [
        "max_abs_raw_direction_diff",
        "max_abs_confidence_diff",
        "max_abs_nondirectional_utility_diff",
        "max_flip_probe_utility_swap_error",
        "max_zero_probe_utility_tie_error",
        "max_shuffle_same_or_swap_error",
    ]
    for metric in audit_names:
        r = next(x for x in summaries if x["task"] == "ALL" and x["metric"] == metric)
        lines.append(f"- {metric}: maximum = {float(r['maximum']):.3e}")

    lines.extend([
        "",
        "## Primary cloned-state result",
        "",
        "Values are seed-level means with bootstrap 95% confidence intervals.",
        "",
        "| Context | Flip opposite probe | Flip probe-class retention | Shuffle sign-flip rate |",
        "|---|---:|---:|---:|",
    ])
    contexts = [
        ("current_sufficient", "current", "Current"),
        ("history_required", "history", "History"),
        ("hidden_reversal", "pre", "Pre-reversal"),
        ("hidden_reversal", "post", "Post-reversal"),
        ("action_independent_null", "null", "Null"),
    ]
    for task, phase, label in contexts:
        lines.append(
            f"| {label} | {fmt(task, phase, 'Direction_Flip_opposite_probe_rate_given_full_trusted_probe')} | "
            f"{fmt(task, phase, 'Direction_Flip_probe_class_retention_given_full_trusted_probe')} | "
            f"{fmt(task, phase, 'Direction_Shuffle_sign_flip_rate_given_trusted_direction')} |"
        )

    lines.extend([
        "",
        "## Interpretation constraints",
        "",
        "- Flip and Shuffle preserve |d(t)|, so the maximum of the two directed probe utilities is unchanged; they isolate the transmitted sign when the direction gate is open.",
        "- Zero removes the directional drive. Because the maximum directed-probe utility falls from U_probe + |drive| to U_probe, Zero may change both probe direction and whether the probe class wins against other actions. It is therefore a clean d(t)-term ablation, but not a direction-choice-only intervention.",
        "- TimeShuffle may change both sign and magnitude because a past d(t) is substituted; it is a temporal-alignment control rather than a pure sign control.",
        "- Clone states are repeated within regimes and seeds. Event rows are descriptive; inferential summaries use the base seed as the unit.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", default=str(Path(__file__).with_name("21_darca_v24_intrinsic_strict_core.py")))
    p.add_argument("--outdir", default=str(Path.home() / "Desktop" / "DARCA_V24_CLONED_STATE_INTERVENTION"))
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    p.add_argument("--smoke", action="store_true", help="2 seeds, 2 regimes, current and reversal only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds % 2:
        raise ValueError("--seeds must be even")
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    core_path = str(Path(args.core).expanduser().resolve())
    mod = load_core(core_path)
    audit = preflight(mod)
    write_csv(outdir / "01_preflight_audit.csv", audit)

    if args.smoke:
        n_seeds = 2
        regimes = [REGIMES[0], REGIMES[-1]]
        tasks = ["current_sufficient", "hidden_reversal"]
    else:
        n_seeds = args.seeds
        regimes = REGIMES
        tasks = TASKS

    rules = balanced_rule_assignment(n_seeds)
    payloads = [(seed, rules[seed], regimes, tasks) for seed in range(n_seeds)]
    t0 = time.time()
    all_events: List[Dict[str, Any]] = []

    if args.workers <= 1:
        init_worker(core_path)
        for i, payload in enumerate(payloads, start=1):
            all_events.extend(seed_worker(payload))
            print(f"[progress] seeds {i}/{n_seeds}; clone states={len(all_events):,}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(args.workers, n_seeds), initializer=init_worker, initargs=(core_path,)) as pool:
            for i, rows in enumerate(pool.imap_unordered(seed_worker, payloads), start=1):
                all_events.extend(rows)
                print(f"[progress] seeds {i}/{n_seeds}; clone states={len(all_events):,}", flush=True)

    all_events.sort(key=lambda r: (int(r["base_seed"]), str(r["task"]), int(r["regime_index"]), int(r["global_step"])))
    elapsed = time.time() - t0
    seed_rows = aggregate_seed_metrics(all_events)
    summaries = summarize_seed_metrics(seed_rows)

    write_csv(outdir / "02_clone_events.csv", all_events)
    write_csv(outdir / "03_seed_level_clone_metrics.csv", seed_rows)
    write_csv(outdir / "04_clone_summary.csv", summaries)
    (outdir / "05_execution_report.md").write_text(
        report_text(all_events, seed_rows, summaries, elapsed, n_seeds, core_path), encoding="utf-8"
    )

    print(f"[done] outdir={outdir}")
    print(f"[done] clone_states={len(all_events):,}; clone_evaluations={len(all_events) * len(MODES):,}; elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
