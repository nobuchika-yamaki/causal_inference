#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_DARCA_V24_INTRINSIC_STRICT_VALIDATION.py
================================================
Confirmatory validation of the online causal-direction pathway in DARCA.

The protocol is fixed before the main run. It uses an independently shuffled,
exactly balanced hidden-rule assignment across base seeds, common random numbers
across conditions, seed-level
inference, clean output-side interventions on d(t), exact confidence-gate
counts, and time-resolved reversal outcomes.

Primary conditions:
    Full, Direction_Zero, Direction_Shuffle, Direction_Flip
Secondary conditions:
    Direction_TimeShuffle, No_CausalInference (broad lesion)

Confirmatory tasks:
    current_sufficient, history_required, hidden_reversal
Negative control:
    action_independent_null

Outputs are written directly to the selected directory. No post hoc parameter
search or task-specific tuning is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Small 96 x 96 matrix-vector operations are faster and far more scalable
# with one BLAS thread per worker. This must be set before NumPy is imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

CONDITIONS = [
    "Full",
    "Direction_Zero",
    "Direction_Shuffle",
    "Direction_Flip",
    "Direction_TimeShuffle",
    "No_CausalInference",
]
PRIMARY_CONTROLS = ["Direction_Zero", "Direction_Shuffle", "Direction_Flip"]
SECONDARY_CONTROLS = ["Direction_TimeShuffle", "No_CausalInference"]
CONFIRMATORY_TASKS = ["current_sufficient", "history_required", "hidden_reversal"]
NULL_TASK = "action_independent_null"
TASKS = CONFIRMATORY_TASKS + [NULL_TASK]

DELAYS = [4, 8, 12]
NOISES = [0.035, 0.060]
COUPLINGS = [0.45, 0.80]
GENERAL_REGIMES = [(d, n, c) for d in DELAYS for n in NOISES for c in COUPLINGS]
BASELINE_REGIME = (4, 0.035, 0.45)

ACQUISITION_STEPS = 300
EVALUATION_STEPS = 200
PRE_REVERSAL_STEPS = 150
POST_REVERSAL_STEPS = 200
REVERSAL_BINS = 4

BOOTSTRAP_RESAMPLES = 5000
PERMUTATIONS = 20000
ANALYSIS_SEED = 20260717
DIRECTION_EPS = 1e-6
EQUIVALENCE_MARGIN = 0.025

TASK_CODE = {name: i + 1 for i, name in enumerate(TASKS)}


def load_core(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location("_darca_strict_core", str(p))
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


@dataclass
class Counter:
    step_count: int = 0
    probe_count: int = 0
    correct_probe_count: int = 0
    trusted_probe_count: int = 0
    trusted_correct_count: int = 0
    untrusted_probe_count: int = 0
    untrusted_correct_count: int = 0
    direction_defined_count: int = 0
    direction_correct_count: int = 0
    direction_sign_sum: float = 0.0
    direction_alignment_sum: float = 0.0
    used_direction_alignment_sum: float = 0.0
    confidence_sum: float = 0.0
    probe_confidence_sum: float = 0.0
    h_sum: float = 0.0
    terminal_sum: float = 0.0

    def update(self, *, rule: int, is_probe: bool, probe_sign: int, d_true: float,
               d_used: float, confidence: float, trusted: bool, h: float,
               terminal: float) -> None:
        self.step_count += 1
        self.confidence_sum += confidence
        self.h_sum += h
        self.terminal_sum += terminal
        self.direction_alignment_sum += rule * d_true
        self.used_direction_alignment_sum += rule * d_used
        if abs(d_true) > DIRECTION_EPS:
            self.direction_defined_count += 1
            sign_true = 1 if d_true > 0 else -1
            self.direction_correct_count += int(sign_true == rule)
            self.direction_sign_sum += rule * sign_true
        if is_probe:
            self.probe_count += 1
            correct = int(probe_sign == rule)
            self.correct_probe_count += correct
            self.probe_confidence_sum += confidence
            if trusted:
                self.trusted_probe_count += 1
                self.trusted_correct_count += correct
            else:
                self.untrusted_probe_count += 1
                self.untrusted_correct_count += correct

    def merge(self, other: "Counter") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def metrics(self, prefix: str = "") -> Dict[str, float]:
        def ratio(a: float, b: float) -> float:
            return float(a / b) if b > 0 else float("nan")

        p = prefix
        cpr = ratio(self.correct_probe_count, self.probe_count)
        trusted_rate = ratio(self.trusted_correct_count, self.trusted_probe_count)
        untrusted_rate = ratio(self.untrusted_correct_count, self.untrusted_probe_count)
        trusted_fraction = ratio(self.trusted_probe_count, self.probe_count)
        reconstructed = (
            trusted_fraction * trusted_rate + (1.0 - trusted_fraction) * untrusted_rate
            if all(math.isfinite(x) for x in (trusted_fraction, trusted_rate, untrusted_rate))
            else float("nan")
        )
        return {
            f"{p}step_count": float(self.step_count),
            f"{p}probe_count": float(self.probe_count),
            f"{p}correct_probe_count": float(self.correct_probe_count),
            f"{p}correct_probe_rate": cpr,
            f"{p}trusted_probe_count": float(self.trusted_probe_count),
            f"{p}trusted_correct_count": float(self.trusted_correct_count),
            f"{p}trusted_correct_probe_rate": trusted_rate,
            f"{p}untrusted_probe_count": float(self.untrusted_probe_count),
            f"{p}untrusted_correct_count": float(self.untrusted_correct_count),
            f"{p}untrusted_correct_probe_rate": untrusted_rate,
            f"{p}trusted_probe_fraction": trusted_fraction,
            f"{p}confidence_reconstructed_rate": reconstructed,
            f"{p}confidence_reconstruction_error": reconstructed - cpr if math.isfinite(reconstructed) and math.isfinite(cpr) else float("nan"),
            f"{p}direction_defined_count": float(self.direction_defined_count),
            f"{p}direction_correct_count": float(self.direction_correct_count),
            f"{p}causal_direction_accuracy": ratio(self.direction_correct_count, self.direction_defined_count),
            f"{p}direction_defined_fraction": ratio(self.direction_defined_count, self.step_count),
            f"{p}direction_sign_score": ratio(self.direction_sign_sum, self.step_count),
            f"{p}direction_alignment": ratio(self.direction_alignment_sum, self.step_count),
            f"{p}used_direction_alignment": ratio(self.used_direction_alignment_sum, self.step_count),
            f"{p}mean_confidence": ratio(self.confidence_sum, self.step_count),
            f"{p}mean_probe_confidence": ratio(self.probe_confidence_sum, self.probe_count),
            f"{p}mean_h": ratio(self.h_sum, self.step_count),
            f"{p}terminal_fraction": ratio(self.terminal_sum, self.step_count),
        }


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0xFFFFFFFF


def task_regimes(task: str) -> List[Tuple[int, float, float]]:
    return GENERAL_REGIMES


def balanced_rule_assignment(n_seeds: int) -> List[int]:
    if n_seeds % 2 != 0:
        raise ValueError("the number of base seeds must be even for exact rule balance")
    rules = np.array([-1] * (n_seeds // 2) + [1] * (n_seeds // 2), dtype=int)
    rng = np.random.default_rng(stable_seed(ANALYSIS_SEED, "balanced_rule_assignment", n_seeds))
    rng.shuffle(rules)
    return rules.tolist()


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
    if task == NULL_TASK:
        return [Phase(ACQUISITION_STEPS + EVALUATION_STEPS, rule, True, True, True, d, n, c, "null")]
    raise ValueError(task)


def _new_buffer(delay: int, noise: float, rng: np.random.Generator) -> deque[float]:
    # At episode starts and explicit feedback-off boundaries, no prior action-
    # dependent outcomes are allowed to leak into the new phase.
    return deque((float(rng.normal(0.0, noise)) for _ in range(delay)), maxlen=delay)


def run_episode(mod, agent, phases: Sequence[Phase], env_seed: int) -> Dict[str, Counter]:
    rng = np.random.default_rng(env_seed)
    counters: Dict[str, Counter] = defaultdict(Counter)
    buffer: Optional[deque[float]] = None
    current_delay: Optional[int] = None
    global_step = 0

    for phase in phases:
        if phase.delay < 1:
            raise ValueError("delay must be >= 1")
        if buffer is None or phase.reset_buffer or current_delay != phase.delay:
            buffer = _new_buffer(phase.delay, phase.noise, rng)
            current_delay = phase.delay

        for phase_i in range(phase.steps):
            assert buffer is not None
            y = float(buffer.popleft())
            out = agent.step(y, {"external_shock": 0.0, "d_dyn": 0.0, "coupling_t": 0.0, "sigma_t": 0.0})
            action_id = int(out["action_id"])
            u = float(mod.action_effect(action_id, y))
            noise = float(rng.normal(0.0, phase.noise))
            if phase.action_independent or not phase.feedback:
                y_future = noise
            else:
                y_future = phase.rule * u * phase.coupling + noise
            buffer.append(float(y_future))

            if phase.scored:
                is_probe = action_id in (mod.A_PROBE_PLUS, mod.A_PROBE_MINUS)
                probe_sign = 1 if action_id == mod.A_PROBE_PLUS else -1 if action_id == mod.A_PROBE_MINUS else 0
                confidence = float(out.get("causal_confidence", 0.0))
                d_true = float(out.get("causal_direction", 0.0))
                d_used = float(out.get("causal_direction_used", d_true))
                trusted = bool(is_probe and confidence >= agent.p.probe_direction_threshold)
                common = dict(
                    rule=phase.rule,
                    is_probe=is_probe,
                    probe_sign=probe_sign,
                    d_true=d_true,
                    d_used=d_used,
                    confidence=confidence,
                    trusted=trusted,
                    h=float(out.get("h", 0.0)),
                    terminal=float(out.get("terminal", 0.0)),
                )
                counters["scored"].update(**common)
                counters[phase.tag].update(**common)
                if phase.tag == "post":
                    bin_size = POST_REVERSAL_STEPS // REVERSAL_BINS
                    b = min(REVERSAL_BINS - 1, phase_i // bin_size) + 1
                    counters[f"post_bin{b}"].update(**common)
            global_step += 1
    return counters


def run_cell(mod, condition, task: str, base_seed: int, rule: int,
             regime: Tuple[int, float, float], regime_index: int) -> Dict[str, Any]:
    agent = mod.Agent(mod.Params(), condition, base_seed)
    env_seed = stable_seed(ANALYSIS_SEED, "environment", base_seed, TASK_CODE[task], regime_index)
    counters = run_episode(mod, agent, task_phases(task, rule, regime), env_seed)

    primary_tag = {
        "current_sufficient": "current",
        "history_required": "history",
        "hidden_reversal": "post",
        NULL_TASK: "null",
    }[task]
    primary = counters[primary_tag]
    metrics = primary.metrics()

    if task == "history_required":
        cpr = metrics["correct_probe_rate"]
        cda = metrics["causal_direction_accuracy"]
        metrics["history_matched_minus_opposite_advantage"] = 2.0 * cpr - 1.0 if math.isfinite(cpr) else float("nan")
        metrics["history_direction_advantage"] = 2.0 * cda - 1.0 if math.isfinite(cda) else float("nan")
    if task == "hidden_reversal":
        metrics.update(counters["pre"].metrics("pre_"))
        metrics.update(counters["post"].metrics("post_"))
        for b in range(1, REVERSAL_BINS + 1):
            metrics.update(counters[f"post_bin{b}"].metrics(f"post_bin{b}_"))
    return metrics


# --------------------------- seed-level aggregation ---------------------------
COUNT_SUFFIXES = (
    "step_count", "probe_count", "correct_probe_count", "trusted_probe_count",
    "trusted_correct_count", "untrusted_probe_count", "untrusted_correct_count",
    "direction_defined_count", "direction_correct_count",
)
SUM_RECOVERABLE = {
    "direction_sign_score": "step_count",
    "direction_alignment": "step_count",
    "used_direction_alignment": "step_count",
    "mean_confidence": "step_count",
    "mean_probe_confidence": "probe_count",
    "mean_h": "step_count",
    "terminal_fraction": "step_count",
}


def _prefixes_from_row(row: Mapping[str, Any]) -> List[str]:
    # Every counter exports exactly one <prefix>step_count field. Deriving
    # prefixes only from that unique suffix avoids false prefixes such as
    # "trusted_" from the field "trusted_probe_count".
    suffix = "step_count"
    return sorted({key[:-len(suffix)] for key in row if key.endswith(suffix)})


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    prefixes = sorted({p for r in rows for p in _prefixes_from_row(r)})
    out: Dict[str, float] = {}
    for p in prefixes:
        counts: Dict[str, float] = {}
        for suffix in COUNT_SUFFIXES:
            key = p + suffix
            counts[suffix] = float(sum(float(r.get(key, 0.0)) for r in rows if math.isfinite(float(r.get(key, 0.0)))))
            out[key] = counts[suffix]

        step_n = counts["step_count"]
        probe_n = counts["probe_count"]
        trusted_n = counts["trusted_probe_count"]
        untrusted_n = counts["untrusted_probe_count"]
        defined_n = counts["direction_defined_count"]

        def div(a: float, b: float) -> float:
            return a / b if b > 0 else float("nan")

        out[p + "correct_probe_rate"] = div(counts["correct_probe_count"], probe_n)
        out[p + "trusted_correct_probe_rate"] = div(counts["trusted_correct_count"], trusted_n)
        out[p + "untrusted_correct_probe_rate"] = div(counts["untrusted_correct_count"], untrusted_n)
        out[p + "trusted_probe_fraction"] = div(trusted_n, probe_n)
        out[p + "causal_direction_accuracy"] = div(counts["direction_correct_count"], defined_n)
        out[p + "direction_defined_fraction"] = div(defined_n, step_n)

        for metric, weight_metric in SUM_RECOVERABLE.items():
            key = p + metric
            weight_key = p + weight_metric
            num = 0.0
            den = 0.0
            for r in rows:
                value = float(r.get(key, float("nan")))
                weight = float(r.get(weight_key, 0.0))
                if math.isfinite(value) and math.isfinite(weight) and weight > 0:
                    num += value * weight
                    den += weight
            out[key] = div(num, den)

        ft = out[p + "trusted_probe_fraction"]
        at = out[p + "trusted_correct_probe_rate"]
        au = out[p + "untrusted_correct_probe_rate"]
        cpr = out[p + "correct_probe_rate"]
        reconstructed = ft * at + (1.0 - ft) * au if all(math.isfinite(x) for x in (ft, at, au)) else float("nan")
        out[p + "confidence_reconstructed_rate"] = reconstructed
        out[p + "confidence_reconstruction_error"] = reconstructed - cpr if math.isfinite(reconstructed) and math.isfinite(cpr) else float("nan")

    # Derived history estimands.
    if math.isfinite(out.get("correct_probe_rate", float("nan"))):
        out["history_matched_minus_opposite_advantage"] = 2.0 * out["correct_probe_rate"] - 1.0
    if math.isfinite(out.get("causal_direction_accuracy", float("nan"))):
        out["history_direction_advantage"] = 2.0 * out["causal_direction_accuracy"] - 1.0
    return out


# -------------------------------- statistics ---------------------------------
def bh_fdr(pvals: Sequence[float]) -> List[float]:
    arr = np.asarray(pvals, dtype=float)
    q = np.full(arr.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(arr))
    if valid.size == 0:
        return q.tolist()
    order = valid[np.argsort(arr[valid])]
    m = len(order)
    running = 1.0
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = m - rank_rev + 1
        running = min(running, float(arr[idx]) * m / rank)
        q[idx] = min(1.0, running)
    return q.tolist()


def _test_rng(label: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(ANALYSIS_SEED, "analysis", label))


def one_sample_stats(values: Sequence[float], null: float, label: str) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return {k: float("nan") for k in ("mean", "ci95_low", "ci95_high", "ci90_low", "ci90_high", "dz", "p", "positive_fraction")} | {"n": n}
    d = x - null
    rng = _test_rng(label)
    mean = float(np.mean(x))
    boots = x[rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))].mean(axis=1)
    lo95, hi95 = np.percentile(boots, [2.5, 97.5])
    lo90, hi90 = np.percentile(boots, [5.0, 95.0])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(PERMUTATIONS, n))
    perm_means = np.mean(signs * d, axis=1)
    extreme = int(np.sum(np.abs(perm_means) >= abs(float(np.mean(d))) - 1e-15))
    p = (extreme + 1.0) / (PERMUTATIONS + 1.0)
    sd = float(np.std(d, ddof=1))
    dz = float(np.mean(d) / sd) if sd > 0 else (math.inf if np.mean(d) > 0 else -math.inf if np.mean(d) < 0 else 0.0)
    return {
        "n": n,
        "mean": mean,
        "null": null,
        "mean_minus_null": float(np.mean(d)),
        "ci95_low": float(lo95),
        "ci95_high": float(hi95),
        "ci90_low": float(lo90),
        "ci90_high": float(hi90),
        "dz": dz,
        "p": float(p),
        "positive_fraction": float(np.mean(d > 0)),
    }


def paired_stats(a: Sequence[float], b: Sequence[float], label: str) -> Dict[str, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    keep = np.isfinite(aa) & np.isfinite(bb)
    d = aa[keep] - bb[keep]
    st = one_sample_stats(d, 0.0, label)
    st["mean_a"] = float(np.mean(aa[keep])) if np.any(keep) else float("nan")
    st["mean_b"] = float(np.mean(bb[keep])) if np.any(keep) else float("nan")
    st["mean_difference"] = st.pop("mean")
    st["ci95_low_difference"] = st.pop("ci95_low")
    st["ci95_high_difference"] = st.pop("ci95_high")
    st["ci90_low_difference"] = st.pop("ci90_low")
    st["ci90_high_difference"] = st.pop("ci90_high")
    st.pop("null", None)
    st.pop("mean_minus_null", None)
    return st


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
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# -------------------------------- execution ----------------------------------
_WMOD = None


def _init_worker(core_path: str) -> None:
    global _WMOD
    _WMOD = load_core(core_path)


def _seed_payload(payload: Tuple[int, int]) -> List[Dict[str, Any]]:
    base_seed, rule = payload
    mod = _WMOD
    if mod is None:
        raise RuntimeError("worker core not initialized")
    conds = {c.name: c for c in mod.build_conditions("all")}
    out: List[Dict[str, Any]] = []
    for task in TASKS:
        regimes = task_regimes(task)
        for regime_index, regime in enumerate(regimes):
            d, n, c = regime
            regime_id = f"d{d}_n{n:.3f}_c{c:.2f}"
            for condition_name in CONDITIONS:
                metrics = run_cell(mod, conds[condition_name], task, base_seed, rule, regime, regime_index)
                row: Dict[str, Any] = {
                    "task": task,
                    "condition": condition_name,
                    "base_seed": base_seed,
                    "rule": rule,
                    "regime_id": regime_id,
                    "delay": d,
                    "noise": n,
                    "coupling": c,
                }
                row.update(metrics)
                out.append(row)
    return out


def preflight(mod) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    conds = {c.name: c for c in mod.build_conditions("all")}
    missing = [c for c in CONDITIONS if c not in conds]
    if missing:
        raise RuntimeError(f"missing conditions: {missing}")
    rows.append({"audit": "required_conditions", "status": "PASS", "detail": ", ".join(CONDITIONS)})

    full = {k: v for k, v in asdict(conds["Full"]).items() if k != "name"}
    expected = {
        "Direction_Zero": {"direction_mode": "zero"},
        "Direction_Shuffle": {"direction_mode": "shuffle"},
        "Direction_Flip": {"direction_mode": "flip"},
        "Direction_TimeShuffle": {"direction_mode": "time_shuffle"},
        "No_CausalInference": {"causal_enabled": False},
    }
    for name, expected_diff in expected.items():
        cfg = {k: v for k, v in asdict(conds[name]).items() if k != "name"}
        actual = {k: cfg[k] for k in cfg if cfg[k] != full[k]}
        if actual != expected_diff:
            raise RuntimeError(f"{name} differs from Full by {actual}, expected {expected_diff}")
        rows.append({"audit": f"condition_diff_{name}", "status": "PASS", "detail": str(actual)})

    # Determinism of a complete cell.
    regime = GENERAL_REGIMES[0]
    m1 = run_cell(mod, conds["Full"], "current_sufficient", 3, +1, regime, 0)
    m2 = run_cell(mod, conds["Full"], "current_sufficient", 3, +1, regime, 0)
    keys = sorted(set(m1) & set(m2))
    for key in keys:
        a, b = m1[key], m2[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) > 1e-12:
                raise RuntimeError(f"nondeterministic metric {key}: {a} vs {b}")
    rows.append({"audit": "fixed_seed_determinism", "status": "PASS", "detail": "all finite metrics identical"})

    # Point-of-use direction transformations. These are checked on complete
    # episodes rather than inferred from condition names.
    for name in ["Direction_Zero", "Direction_Shuffle", "Direction_Flip", "Direction_TimeShuffle"]:
        agent = mod.Agent(mod.Params(), conds[name], 7)
        phases = task_phases("current_sufficient", +1, regime)
        rng = np.random.default_rng(stable_seed(ANALYSIS_SEED, "preflight", name))
        buffer = _new_buffer(regime[0], regime[1], rng)
        true_hist: List[float] = []
        checked = 0
        for _ in range(500):
            y = float(buffer.popleft())
            out = agent.step(y, {"external_shock": 0.0, "d_dyn": 0.0, "coupling_t": 0.0, "sigma_t": 0.0})
            u = float(mod.action_effect(int(out["action_id"]), y))
            buffer.append(+1 * u * regime[2] + float(rng.normal(0.0, regime[1])))
            dt = float(out["causal_direction"])
            du = float(out["causal_direction_used"])
            if name == "Direction_Zero" and abs(du) > 1e-12:
                raise RuntimeError("Direction_Zero transmitted nonzero d")
            if name == "Direction_Flip" and abs(du + dt) > 1e-12:
                raise RuntimeError("Direction_Flip did not invert d")
            if name == "Direction_Shuffle" and abs(abs(du) - abs(dt)) > 1e-12:
                raise RuntimeError("Direction_Shuffle changed |d|")
            if name == "Direction_TimeShuffle" and true_hist and not any(abs(du - x) <= 1e-12 for x in true_hist):
                raise RuntimeError("Direction_TimeShuffle used a nonhistorical value")
            true_hist.append(dt)
            checked += 1
        rows.append({"audit": f"point_of_use_{name}", "status": "PASS", "detail": f"{checked} steps checked"})

    rows.append({"audit": "balanced_rule_design", "status": "PASS", "detail": "hidden-rule sign is independently shuffled and exactly balanced across base seeds"})
    rows.append({"audit": "feedback_off_buffer_reset", "status": "PASS", "detail": "the history evaluation begins with action-independent noise; no acquisition outcomes leak"})
    return rows


def make_seed_level(raw_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        groups[(str(row["task"]), str(row["condition"]), int(row["base_seed"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (task, condition, seed), rows in sorted(groups.items()):
        agg = aggregate_rows(rows)
        rec: Dict[str, Any] = {"task": task, "condition": condition, "base_seed": seed, "n_rule_regime_cells": len(rows)}
        rec.update(agg)
        out.append(rec)
    return out


def index_seed_metrics(seed_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    return {(str(r["task"]), str(r["condition"]), int(r["base_seed"])): r for r in seed_rows}


def values_by_seed(index: Mapping[Tuple[str, str, int], Mapping[str, Any]], task: str,
                   condition: str, metric: str, seeds: Iterable[int]) -> List[float]:
    return [float(index[(task, condition, s)].get(metric, float("nan"))) for s in seeds]


def analyze(seed_rows: Sequence[Mapping[str, Any]], n_seeds: int) -> Dict[str, List[Dict[str, Any]]]:
    idx = index_seed_metrics(seed_rows)
    seeds = list(range(n_seeds))
    results: Dict[str, List[Dict[str, Any]]] = {}

    # Primary confirmatory family: Full versus clean d(t) controls on behaviour.
    primary: List[Dict[str, Any]] = []
    for task in CONFIRMATORY_TASKS:
        for control in PRIMARY_CONTROLS:
            a = values_by_seed(idx, task, "Full", "correct_probe_rate", seeds)
            b = values_by_seed(idx, task, control, "correct_probe_rate", seeds)
            st = paired_stats(a, b, f"primary|{task}|{control}")
            primary.append({"task": task, "metric": "correct_probe_rate", "contrast": f"Full_minus_{control}", **st})
    q = bh_fdr([r["p"] for r in primary])
    for row, qi in zip(primary, q):
        row["q_fdr_family_primary"] = qi
    results["primary_contrasts"] = primary

    # Chance tests for pre-specified behavioural and internal outcomes.
    chance: List[Dict[str, Any]] = []
    chance_specs = []
    for task in CONFIRMATORY_TASKS + [NULL_TASK]:
        chance_specs.append((task, "Full", "correct_probe_rate", 0.5))
        chance_specs.append((task, "Full", "causal_direction_accuracy", 0.5))
    for task in CONFIRMATORY_TASKS:
        for cond in PRIMARY_CONTROLS:
            chance_specs.append((task, cond, "correct_probe_rate", 0.5))
    for task, condition, metric, null in chance_specs:
        vals = values_by_seed(idx, task, condition, metric, seeds)
        st = one_sample_stats(vals, null, f"chance|{task}|{condition}|{metric}")
        chance.append({"task": task, "condition": condition, "metric": metric, **st})
    q = bh_fdr([r["p"] for r in chance])
    for row, qi in zip(chance, q):
        row["q_fdr_family_chance"] = qi
    results["chance_tests"] = chance

    # Equivalence to chance for Zero, Shuffle, and the action-independent null.
    equivalence: List[Dict[str, Any]] = []
    eq_specs = [(task, cond) for task in CONFIRMATORY_TASKS for cond in ["Direction_Zero", "Direction_Shuffle"]]
    eq_specs += [(NULL_TASK, cond) for cond in CONDITIONS]
    for task, condition in eq_specs:
        vals = values_by_seed(idx, task, condition, "correct_probe_rate", seeds)
        st = one_sample_stats(vals, 0.5, f"equivalence|{task}|{condition}")
        equivalent = bool(st["ci90_low"] >= 0.5 - EQUIVALENCE_MARGIN and st["ci90_high"] <= 0.5 + EQUIVALENCE_MARGIN)
        equivalence.append({
            "task": task,
            "condition": condition,
            "metric": "correct_probe_rate",
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "equivalent_to_chance_90ci": equivalent,
            **st,
        })
    results["equivalence_tests"] = equivalence

    # Null-gate difference-in-differences.
    did: List[Dict[str, Any]] = []
    for task in CONFIRMATORY_TASKS:
        for control in PRIMARY_CONTROLS:
            full_task = np.asarray(values_by_seed(idx, task, "Full", "correct_probe_rate", seeds), float)
            ctrl_task = np.asarray(values_by_seed(idx, task, control, "correct_probe_rate", seeds), float)
            full_null = np.asarray(values_by_seed(idx, NULL_TASK, "Full", "correct_probe_rate", seeds), float)
            ctrl_null = np.asarray(values_by_seed(idx, NULL_TASK, control, "correct_probe_rate", seeds), float)
            task_benefit = full_task - ctrl_task
            null_benefit = full_null - ctrl_null
            st = paired_stats(task_benefit, null_benefit, f"did|{task}|{control}")
            did.append({"task": task, "null_task": NULL_TASK, "contrast": f"Full_minus_{control}", **st})
    q = bh_fdr([r["p"] for r in did])
    for row, qi in zip(did, q):
        row["q_fdr_family_did"] = qi
    results["null_gate_did"] = did

    # Reversal time course and pre-specified recovery contrasts.
    timecourse: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        for b in range(1, REVERSAL_BINS + 1):
            for metric in ("correct_probe_rate", "causal_direction_accuracy", "mean_confidence"):
                key = f"post_bin{b}_{metric}"
                vals = values_by_seed(idx, "hidden_reversal", condition, key, seeds)
                null = 0.5 if metric != "mean_confidence" else 0.0
                st = one_sample_stats(vals, null, f"timecourse|{condition}|b{b}|{metric}")
                timecourse.append({"condition": condition, "bin": b, "metric": metric, **st})
    results["reversal_timecourse"] = timecourse

    reversal_tests: List[Dict[str, Any]] = []
    for metric in ("correct_probe_rate", "causal_direction_accuracy", "mean_confidence"):
        b1 = values_by_seed(idx, "hidden_reversal", "Full", f"post_bin1_{metric}", seeds)
        b4 = values_by_seed(idx, "hidden_reversal", "Full", f"post_bin4_{metric}", seeds)
        st = paired_stats(b4, b1, f"reversal_recovery|{metric}")
        reversal_tests.append({"contrast": "post_bin4_minus_post_bin1", "metric": metric, **st})
    q = bh_fdr([r["p"] for r in reversal_tests])
    for row, qi in zip(reversal_tests, q):
        row["q_fdr_family_reversal"] = qi
    results["reversal_recovery_tests"] = reversal_tests

    # Confidence gate: pooled within seed, so the weighted decomposition is exact.
    confidence: List[Dict[str, Any]] = []
    confidence_tests: List[Dict[str, Any]] = []
    for task in CONFIRMATORY_TASKS:
        ft = values_by_seed(idx, task, "Full", "trusted_probe_fraction", seeds)
        at = values_by_seed(idx, task, "Full", "trusted_correct_probe_rate", seeds)
        au = values_by_seed(idx, task, "Full", "untrusted_correct_probe_rate", seeds)
        observed = values_by_seed(idx, task, "Full", "correct_probe_rate", seeds)
        reconstructed = values_by_seed(idx, task, "Full", "confidence_reconstructed_rate", seeds)
        error = values_by_seed(idx, task, "Full", "confidence_reconstruction_error", seeds)
        for metric, vals in [
            ("trusted_probe_fraction", ft),
            ("trusted_correct_probe_rate", at),
            ("untrusted_correct_probe_rate", au),
            ("observed_correct_probe_rate", observed),
            ("reconstructed_correct_probe_rate", reconstructed),
            ("reconstruction_error", error),
        ]:
            null = 0.0
            st = one_sample_stats(vals, null, f"confidence_summary|{task}|{metric}")
            confidence.append({"task": task, "metric": metric, **st})
        if task in CONFIRMATORY_TASKS:
            st = paired_stats(at, au, f"confidence_difference|{task}")
            confidence_tests.append({"task": task, "contrast": "trusted_minus_untrusted_accuracy", **st})
            eq = one_sample_stats(au, 0.5, f"untrusted_equivalence|{task}")
            eq["equivalent_to_chance_90ci"] = bool(eq["ci90_low"] >= 0.5 - EQUIVALENCE_MARGIN and eq["ci90_high"] <= 0.5 + EQUIVALENCE_MARGIN)
            confidence_tests.append({"task": task, "contrast": "untrusted_accuracy_vs_chance_equivalence", **eq})
    q = bh_fdr([r["p"] for r in confidence_tests if "p" in r])
    for row, qi in zip(confidence_tests, q):
        row["q_fdr_family_confidence"] = qi
    results["confidence_gate_summary"] = confidence
    results["confidence_gate_tests"] = confidence_tests

    # Secondary controls and internal-state diagnostics.
    secondary: List[Dict[str, Any]] = []
    for task in CONFIRMATORY_TASKS + [NULL_TASK]:
        for control in SECONDARY_CONTROLS:
            for metric in ("correct_probe_rate", "causal_direction_accuracy", "direction_alignment", "direction_defined_fraction"):
                a = values_by_seed(idx, task, "Full", metric, seeds)
                b = values_by_seed(idx, task, control, metric, seeds)
                st = paired_stats(a, b, f"secondary|{task}|{control}|{metric}")
                secondary.append({"task": task, "metric": metric, "contrast": f"Full_minus_{control}", **st})
    results["secondary_controls"] = secondary

    # Flip symmetry: Full + Flip - 1 = 0 under exact directional inversion.
    symmetry: List[Dict[str, Any]] = []
    for task in CONFIRMATORY_TASKS:
        full = np.asarray(values_by_seed(idx, task, "Full", "correct_probe_rate", seeds), float)
        flip = np.asarray(values_by_seed(idx, task, "Direction_Flip", "correct_probe_rate", seeds), float)
        vals = full + flip - 1.0
        st = one_sample_stats(vals, 0.0, f"flip_symmetry|{task}")
        symmetry.append({"task": task, "estimand": "Full_rate_plus_Flip_rate_minus_1", **st})
    results["flip_symmetry"] = symmetry
    return results


def write_execution_report(path: Path, args: argparse.Namespace, raw_rows: Sequence[Mapping[str, Any]],
                           seed_rows: Sequence[Mapping[str, Any]], audits: Sequence[Mapping[str, Any]],
                           analyses: Mapping[str, Sequence[Mapping[str, Any]]], n_seeds: int) -> None:
    L: List[str] = []
    L.append("# DARCA v2.4 intrinsic directional pathway - strict validation report")
    L.append("")
    L.append("## Execution")
    L.append(f"- Plan: {args.plan}")
    L.append(f"- Base seeds: {n_seeds}")
    L.append("- Hidden-rule assignment: exactly balanced across base seeds (16 r=-1, 16 r=+1 in the main plan)")
    L.append(f"- Conditions: {len(CONDITIONS)}")
    L.append(f"- Raw cells: {len(raw_rows)}")
    L.append(f"- Seed-level task-condition summaries: {len(seed_rows)}")
    L.append(f"- Bootstrap resamples: {BOOTSTRAP_RESAMPLES}")
    L.append(f"- Sign-flip permutations: {PERMUTATIONS}")
    L.append(f"- Equivalence margin around chance: +/-{EQUIVALENCE_MARGIN:.3f}")
    L.append("")
    L.append("## Preflight")
    for row in audits:
        L.append(f"- {row['status']}: {row['audit']} - {row['detail']}")
    L.append("")
    L.append("## Primary behavioural contrasts")
    for row in analyses["primary_contrasts"]:
        L.append(
            f"- {row['task']} | {row['contrast']}: "
            f"difference={row['mean_difference']:.3f}, "
            f"95% CI [{row['ci95_low_difference']:.3f}, {row['ci95_high_difference']:.3f}], "
            f"q={row['q_fdr_family_primary']:.4g}"
        )
    L.append("")
    L.append("## Full condition means")
    idx = index_seed_metrics(seed_rows)
    for task in TASKS:
        cpr = np.nanmean(values_by_seed(idx, task, "Full", "correct_probe_rate", range(n_seeds)))
        cda = np.nanmean(values_by_seed(idx, task, "Full", "causal_direction_accuracy", range(n_seeds)))
        L.append(f"- {task}: correct_probe_rate={cpr:.3f}; causal_direction_accuracy={cda:.3f}")
    L.append("")
    L.append("## Output files")
    for name in [
        "01_preflight_audit.csv", "02_raw_cells.csv", "03_seed_level_metrics.csv",
        "04_primary_contrasts.csv", "05_chance_tests.csv", "06_equivalence_tests.csv",
        "07_null_gate_did.csv", "08_reversal_timecourse.csv", "09_reversal_recovery_tests.csv",
        "10_confidence_gate_summary.csv", "11_confidence_gate_tests.csv",
        "12_secondary_controls.csv", "13_flip_symmetry.csv",
    ]:
        L.append(f"- {name}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def plan_seeds(plan: str) -> int:
    return {"smoke": 2, "quick": 8, "main": 32}[plan]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--darca-file", required=True)
    parser.add_argument("--outdir", default=str(Path.home() / "Desktop" / "DARCA_V24_INTRINSIC_STRICT_VALIDATION"))
    parser.add_argument("--plan", choices=["smoke", "quick", "main"], default="main")
    parser.add_argument("--workers", default="auto")
    args = parser.parse_args(argv)

    core_path = str(Path(args.darca_file).expanduser().resolve())
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    mod = load_core(core_path)
    audits = preflight(mod)
    write_csv(outdir / "01_preflight_audit.csv", audits)

    n_seeds = plan_seeds(args.plan)
    cpu = os.cpu_count() or 1
    if str(args.workers).lower() == "auto":
        workers = max(1, min(n_seeds, cpu - 1))
    else:
        workers = max(1, min(n_seeds, cpu, int(args.workers)))

    expected_cells_per_seed = len(CONDITIONS) * sum(len(task_regimes(t)) for t in TASKS)
    rule_assignment = balanced_rule_assignment(n_seeds)
    payloads = list(enumerate(rule_assignment))
    print(
        f"[run] plan={args.plan} seeds={n_seeds} workers={workers} "
        f"expected_cells={n_seeds * expected_cells_per_seed}",
        flush=True,
    )

    raw_rows: List[Dict[str, Any]] = []
    if workers == 1:
        _init_worker(core_path)
        for seed, rule in payloads:
            raw_rows.extend(_seed_payload((seed, rule)))
            print(f"[run] seed {seed + 1}/{n_seeds} complete", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(core_path,)) as pool:
            done = 0
            for rows in pool.imap_unordered(_seed_payload, payloads):
                raw_rows.extend(rows)
                done += 1
                print(f"[run] seed {done}/{n_seeds} complete", flush=True)

    raw_rows.sort(key=lambda r: (int(r["base_seed"]), str(r["task"]), str(r["regime_id"]), int(r["rule"]), str(r["condition"])))
    if len(raw_rows) != n_seeds * expected_cells_per_seed:
        raise RuntimeError(f"cell count mismatch: observed {len(raw_rows)}, expected {n_seeds * expected_cells_per_seed}")
    write_csv(outdir / "02_raw_cells.csv", raw_rows)

    seed_rows = make_seed_level(raw_rows)
    write_csv(outdir / "03_seed_level_metrics.csv", seed_rows)
    analyses = analyze(seed_rows, n_seeds)
    output_map = {
        "primary_contrasts": "04_primary_contrasts.csv",
        "chance_tests": "05_chance_tests.csv",
        "equivalence_tests": "06_equivalence_tests.csv",
        "null_gate_did": "07_null_gate_did.csv",
        "reversal_timecourse": "08_reversal_timecourse.csv",
        "reversal_recovery_tests": "09_reversal_recovery_tests.csv",
        "confidence_gate_summary": "10_confidence_gate_summary.csv",
        "confidence_gate_tests": "11_confidence_gate_tests.csv",
        "secondary_controls": "12_secondary_controls.csv",
        "flip_symmetry": "13_flip_symmetry.csv",
    }
    for key, filename in output_map.items():
        write_csv(outdir / filename, analyses[key])

    write_execution_report(outdir / "00_EXECUTION_REPORT.md", args, raw_rows, seed_rows, audits, analyses, n_seeds)
    print(f"[done] {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
