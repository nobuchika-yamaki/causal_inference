#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DARCA v2.4 rewritten structural-boundary model
==============================================

This is a direct standalone rewrite of the v2.4-style autonomous-life
model for structural boundary analysis. It is not a patch wrapper and it
does not call or modify another model file.

The model retains the v2.4 mechanisms needed for the audit:

- theta-dependent sensory boundary transformation:
      S_t = [|e_t|, (2 theta - 1) e_t]

- theta-risk:
      theta_risk = clip((theta - 0.65) / 0.35, 0, 1)

- temporal membrane

- dissipative recurrent core

- delayed action-conditioned causal inference

- delay-asymmetric agency

- selective causal-delay memory

- meta-autonomy

- hard safety gates and anti-shutdown escape

- viability/autonomy/identity regulation

- non-absorbing terminal flag

The code directly runs:

1. theta-boundary sweep
   default theta = 0.55, 0.60, 0.64, 0.65, 0.66, 0.70,
                   0.75, 0.80, 0.825, 0.85

2. causal-horizon audit
   default theta = 0.55, 0.85
   default horizon = 12, 18, 24

No source patching is used anywhere.

Typical compact run
-------------------
cd ~/Downloads

python3 -u darca_v24_direct_rewrite.py \
  --preset compact \
  --outdir ~/Desktop/DARCA_v24_direct_boundary_compact \
  --workers 4 \
  --audits theta,horizon \
  2>&1 | tee ~/Desktop/DARCA_v24_direct_boundary_compact.log

Outputs
-------
run.log
manifest.json
episode_summary.csv
regime_summary.csv
condition_summary.csv
failure_mode_summary.csv
normalized_condition_metrics.csv
structural_boundary_report.txt
step_timeseries.csv unless --no-step-timeseries is used
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Constants and utilities
# =============================================================================

A_REGULATE = 0
A_PROBE_PLUS = 1
A_PROBE_MINUS = 2
A_INHIBIT = 3
A_EXPRESS = 4

ACTION_NAMES = {
    A_REGULATE: "REGULATE",
    A_PROBE_PLUS: "PROBE_PLUS",
    A_PROBE_MINUS: "PROBE_MINUS",
    A_INHIBIT: "INHIBIT",
    A_EXPRESS: "EXPRESS",
}

DEFAULT_THETAS = [0.55, 0.60, 0.64, 0.65, 0.66, 0.70, 0.75, 0.80, 0.825, 0.85]
DEFAULT_HORIZONS = [12, 18, 24]
DEFAULT_HORIZON_THETAS = [0.55, 0.85]


def clip_float(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def action_onehot(action_id: int) -> np.ndarray:
    v = np.zeros(5, dtype=np.float64)
    v[int(action_id)] = 1.0
    return v


def action_effect(action_id: int, y: float) -> float:
    if action_id == A_PROBE_PLUS:
        return 1.0
    if action_id == A_PROBE_MINUS:
        return -1.0
    if action_id == A_EXPRESS:
        return float(np.sign(y)) if abs(y) > 0.02 else 0.0
    return 0.0


def theta_risk_value(theta: float) -> float:
    return clip_float((float(theta) - 0.65) / 0.35, 0.0, 1.0)


def parse_float_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def parse_int_list(s: str) -> List[int]:
    return [int(round(x)) for x in parse_float_list(s)]


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            out = {}
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, (float, np.floating)):
                    out[k] = f"{float(v):.10g}" if math.isfinite(float(v)) else str(v)
                else:
                    out[k] = v
            w.writerow(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Logger:
    def __init__(self, outdir: Path):
        self.t0 = time.time()
        self.path = outdir / "run.log"
        outdir.mkdir(parents=True, exist_ok=True)
        self.path.write_text("DARCA v2.4 direct rewrite run log\n" + "=" * 80 + "\n", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:10.2f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Params:
    dt: float = 0.01
    theta: float = 0.55
    causal_max_delay: int = 12

    # temporal membrane
    temporal_H: float = 1.20
    temporal_r: float = 0.08
    smooth_window_sec: float = 0.10

    # recurrent core
    recurrent_N: int = 96
    gamma_q: float = 0.30
    gamma_p: float = 0.30
    recurrent_gain: float = 0.90
    input_gain: float = 0.80
    core_noise_sd: float = 0.004
    state_clip: float = 8.0
    sensitivity_eps: float = 1e-3
    sensitivity_probes: int = 4
    sensitivity_collapse_threshold: float = 1e-5

    # causal inference
    causal_eta: float = 0.025
    causal_decay: float = 0.999
    causal_error_alpha: float = 0.02

    # intrinsic directional probe coupling (v2.4-rev):
    # the signed action-conditioned causal-direction estimate enters the
    # PROBE_PLUS / PROBE_MINUS action utilities directly, so the probe
    # direction is a consequence of the action-selection dynamics rather than
    # an external readout. probe_direction_gain is beta in
    #   U_probe_plus  = U_probe + beta * causal_confidence * causal_direction
    #   U_probe_minus = U_probe - beta * causal_confidence * causal_direction
    # Below probe_direction_threshold confidence the direction is not trusted
    # and the two probes are collected by strict alternation.
    probe_direction_gain: float = 0.15
    probe_direction_threshold: float = 0.05

    # agency
    tau_L: int = 4
    delta_tau: int = 1
    agency_eta: float = 0.030
    agency_decay: float = 0.999

    # memory
    agency_event_threshold: float = 0.004
    causal_event_threshold: float = 0.08
    memory_saturation_threshold: float = 1.15
    max_memory_events: int = 18
    memory_horizon: int = 60

    # viability/autonomy
    h_init: float = 0.70
    h_min: float = 0.20
    regulate_fatigue_increment: float = 0.025
    regulate_fatigue_decay: float = 0.92
    regulate_fatigue_max: float = 0.35
    meta_lr: float = 0.008

    # classification
    autonomous_terminal_max: float = 0.12
    autonomous_homeostatic_max: float = 0.12
    autonomous_chronic_max: float = 0.12
    autonomous_bis_min: float = 0.80
    autonomous_causal_min: float = 0.18
    autonomous_living_min: float = 0.45
    autonomous_active_min: float = 0.12


@dataclass(frozen=True)
class Regime:
    audit: str
    theta: float
    theta_risk: float
    causal_horizon: int
    delay: int
    noise_base: float
    coupling_base: float
    shock_mode: str
    regime_id: str


@dataclass(frozen=True)
class Condition:
    name: str
    meta_enabled: bool = True
    causal_enabled: bool = True
    agency_enabled: bool = True
    memory_enabled: bool = True
    temporal_enabled: bool = True
    dissipative_enabled: bool = True
    viability_enabled: bool = True
    delta_tau_override: Optional[int] = None
    direction_shuffle: bool = False  # deprecated; use direction_mode
    direction_mode: str = "intact"  # intact | shuffle | zero | flip | time_shuffle


def build_conditions(selector: str) -> List[Condition]:
    full = [
        Condition("Full"),
        Condition("No_MetaAutonomy", meta_enabled=False),
        Condition("No_CausalInference", causal_enabled=False),
        Condition("No_Agency", agency_enabled=False),
        Condition("No_Memory", memory_enabled=False),
        Condition("Symmetric_Delay", delta_tau_override=0),
        Condition("No_TemporalMembrane", temporal_enabled=False),
        Condition("No_DissipativeCore", dissipative_enabled=False),
        Condition("No_ViabilityAutonomy", viability_enabled=False),
        Condition("Direction_Shuffle", direction_mode="shuffle"),
        Condition("Direction_Zero", direction_mode="zero"),
        Condition("Direction_Flip", direction_mode="flip"),
        Condition("Direction_TimeShuffle", direction_mode="time_shuffle"),
    ]
    if selector.lower() in ("full", "complete"):
        return [full[0]]
    if selector.lower() == "all":
        return full
    wanted = {x.strip() for x in selector.split(",") if x.strip()}
    chosen = [c for c in full if c.name in wanted]
    missing = wanted - {c.name for c in chosen}
    if missing:
        raise ValueError(f"unknown conditions: {sorted(missing)}")
    return chosen


def preset_grid(preset: str) -> Tuple[List[int], List[float], List[float], int, int]:
    if preset == "smoke":
        return [4, 12], [0.035], [0.45], 2, 260
    if preset == "compact":
        return [4, 8, 12], [0.035, 0.06], [0.45, 0.80], 5, 700
    if preset == "main":
        return [4, 8, 12], [0.035, 0.06], [0.45, 0.80], 20, 1000
    if preset == "stress":
        return [4, 8, 12], [0.035, 0.06], [0.45, 0.80], 10, 1000
    raise ValueError(f"unknown preset: {preset}")


def build_regimes(args: argparse.Namespace) -> Tuple[List[Regime], int, int]:
    delays, noises, couplings, episodes, steps = preset_grid(args.preset)
    if args.episodes is not None:
        episodes = args.episodes
    if args.steps is not None:
        steps = args.steps

    audits = {x.strip().lower() for x in args.audits.split(",") if x.strip()}
    thetas = parse_float_list(args.thetas)
    horizons = parse_int_list(args.horizons)
    horizon_thetas = parse_float_list(args.horizon_thetas)

    regimes: List[Regime] = []

    if "theta" in audits:
        for theta, delay, noise, coupling in itertools.product(thetas, delays, noises, couplings):
            tr = theta_risk_value(theta)
            h = args.base_horizon
            rid = f"theta_sweep_theta{theta:.3f}_risk{tr:.3f}_H{h}_d{delay}_n{noise:.3f}_c{coupling:.2f}"
            regimes.append(Regime("theta_sweep", theta, tr, h, delay, noise, coupling, "self_maintenance_shift", rid))

    if "horizon" in audits:
        for theta, horizon, delay, noise, coupling in itertools.product(horizon_thetas, horizons, delays, noises, couplings):
            tr = theta_risk_value(theta)
            rid = f"horizon_audit_theta{theta:.3f}_risk{tr:.3f}_H{horizon}_d{delay}_n{noise:.3f}_c{coupling:.2f}"
            regimes.append(Regime("causal_horizon_audit", theta, tr, horizon, delay, noise, coupling, "self_maintenance_shift", rid))

    if not regimes:
        raise ValueError("No regimes were generated. Use --audits theta,horizon or one of them.")
    return regimes, episodes, steps


# =============================================================================
# Model components
# =============================================================================

class SensoryIntakeModule:
    def __init__(self, theta: float, theta_c: float = 0.483):
        self.theta = clip_float(theta, 0.0, 1.0)
        self.theta_c = theta_c
        self.kappa = 2.0 * self.theta - 1.0
        self.is_contralateral_dominant = self.theta > self.theta_c

    def transform(self, e: float) -> np.ndarray:
        magnitude = abs(float(e))
        transformed_direction = self.kappa * float(e)
        return np.array([magnitude, transformed_direction], dtype=np.float64)


class TemporalMembrane:
    def __init__(self, params: Params, enabled: bool):
        self.p = params
        self.enabled = enabled
        self.max_steps = max(1, int(round(params.temporal_H / params.dt)))
        self.stride = max(1, int(round(params.temporal_r / params.dt)))
        self.smooth_steps = max(1, int(round(params.smooth_window_sec / params.dt)))
        self.offsets = list(range(self.stride, self.max_steps + 1, self.stride))
        if self.max_steps not in self.offsets:
            self.offsets.append(self.max_steps)
        self.offsets = sorted(set(self.offsets))
        self.S_hist: List[np.ndarray] = []
        self.s_hist: List[float] = []
        self.prev_smooth: Optional[np.ndarray] = None
        self.prev_d1: Optional[np.ndarray] = None

    def input_dim(self) -> int:
        if not self.enabled:
            return 2 + 5
        return 2 + 5 + 2 * len(self.offsets) + 2 + 2 + 1

    def step(self, S: np.ndarray, prev_action_vec: np.ndarray) -> Tuple[np.ndarray, float, float]:
        self.S_hist.append(S.copy())
        self.s_hist.append(float(S[1]))
        if len(self.S_hist) > self.max_steps + 2:
            self.S_hist.pop(0)
            self.s_hist.pop(0)

        chi_hat = self._chi_hat()
        chi_norm = clip_float(min(max(chi_hat, 0.0), 2.0) / (1.0 + min(max(chi_hat, 0.0), 2.0)), 0.0, 1.0)

        if not self.enabled:
            return np.concatenate([S, prev_action_vec]), chi_hat, chi_norm

        parts = [S, prev_action_vec]
        for off in self.offsets:
            parts.append(self.S_hist[-(off + 1)] if len(self.S_hist) > off else self.S_hist[0])

        smoothed = np.mean(np.vstack(self.S_hist[-self.smooth_steps:]), axis=0)
        d1 = np.zeros_like(smoothed) if self.prev_smooth is None else (smoothed - self.prev_smooth) / self.p.dt
        d2 = np.zeros_like(smoothed) if self.prev_d1 is None else (d1 - self.prev_d1) / self.p.dt
        self.prev_smooth = smoothed
        self.prev_d1 = d1
        parts.extend([d1, d2, np.array([chi_hat])])
        return np.concatenate(parts), chi_hat, chi_norm

    def _chi_hat(self) -> float:
        if len(self.s_hist) < 12:
            return 0.0
        x = np.asarray(self.s_hist[-self.max_steps:], dtype=np.float64)
        if x.size < 12 or np.std(x) < 1e-12:
            return 0.0
        a = x[:-1] - np.mean(x[:-1])
        b = x[1:] - np.mean(x[1:])
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom < 1e-12:
            return 0.0
        rho = clip_float(float(np.sum(a * b) / denom), -0.99, 0.99)
        if rho <= 0.0:
            corr_time = self.p.dt
        else:
            corr_time = -self.p.dt / math.log(max(rho, 1e-6))
        return clip_float(corr_time / (1.0 / 0.30), 0.0, 10.0)


class DissipativeCore:
    def __init__(self, params: Params, input_dim: int, enabled: bool, rng: np.random.Generator):
        self.p = params
        self.enabled = enabled
        self.rng = rng
        self.N = params.recurrent_N
        self.gamma_q = params.gamma_q if enabled else 0.0
        self.gamma_p = params.gamma_p if enabled else 0.0
        gain = params.recurrent_gain if enabled else 1.22

        self.q = rng.normal(0.0, 0.01, self.N)
        self.p_state = rng.normal(0.0, 0.01, self.N)
        self.q_prev = self.q.copy()

        self.W = rng.normal(0.0, gain / math.sqrt(self.N), (self.N, self.N))
        np.fill_diagonal(self.W, 0.0)
        self.B = rng.normal(0.0, params.input_gain / math.sqrt(input_dim), (self.N, input_dim))
        self.w = rng.normal(0.0, 1.0 / math.sqrt(self.N), self.N)
        self.b = 0.0

        self.prediction_error = 0.0
        self.fluctuation = 0.0

    def step(self, X: np.ndarray, target: float) -> Dict[str, float]:
        self.q_prev = self.q.copy()
        I = self.B @ X
        dq = self.p_state - self.gamma_q * self.q
        dp = -(self.q ** 3 - self.q) - self.W @ self.q - self.gamma_p * self.p_state + I
        dp += self.rng.normal(0.0, self.p.core_noise_sd, self.N)

        self.q = np.clip(self.q + self.p.dt * dq, -self.p.state_clip, self.p.state_clip)
        self.p_state = np.clip(self.p_state + self.p.dt * dp, -self.p.state_clip, self.p.state_clip)

        r = np.tanh(self.q)
        yhat = float(np.tanh(np.dot(self.w, r) + self.b))
        err = float(target - yhat)
        old_pe = self.prediction_error

        self.w = 0.9995 * (self.w + 0.015 * err * r)
        self.b = 0.9995 * (self.b + 0.015 * err)

        self.prediction_error = 0.97 * self.prediction_error + 0.03 * abs(err)
        self.fluctuation = 0.98 * self.fluctuation + 0.02 * float(np.mean((self.q - self.q_prev) ** 2))

        return {
            "prediction_error": self.prediction_error,
            "prediction_improvement": old_pe - self.prediction_error,
            "internal_energy": self.energy(),
            "fluctuation": self.fluctuation,
            "yhat": yhat,
        }

    def energy(self) -> float:
        return float(0.5 * np.mean(self.q ** 2 + self.p_state ** 2))

    def sensitivity(self) -> float:
        base = float(np.tanh(np.dot(self.w, np.tanh(self.q)) + self.b))
        vals: List[float] = []
        for _ in range(self.p.sensitivity_probes):
            v = self.rng.normal(0.0, 1.0, self.N)
            n = np.linalg.norm(v)
            if n <= 1e-12:
                continue
            v /= n
            q2 = np.clip(self.q + self.p.sensitivity_eps * v, -self.p.state_clip, self.p.state_clip)
            y2 = float(np.tanh(np.dot(self.w, np.tanh(q2)) + self.b))
            vals.append(abs(y2 - base) / self.p.sensitivity_eps)
        return float(np.mean(vals)) if vals else 0.0


class CausalInference:
    def __init__(self, params: Params, enabled: bool):
        self.p = params
        self.enabled = enabled
        self.max_delay = int(params.causal_max_delay)
        self.s_buffer: List[float] = []
        self.a_buffer: List[np.ndarray] = []

        self.w_exo = np.zeros(2, dtype=np.float64)
        self.w_delay = {d: np.zeros(6, dtype=np.float64) for d in range(1, self.max_delay + 1)}
        self.E_exo = 1.0
        self.E_delay = {d: 1.0 for d in range(1, self.max_delay + 1)}
        self.best_delay = 0

    def step(self, s: float, prev_action_vec: np.ndarray) -> Dict[str, float]:
        self.s_buffer.append(float(s))
        self.a_buffer.append(prev_action_vec.copy())
        if len(self.s_buffer) > self.max_delay + 5:
            self.s_buffer.pop(0)
            self.a_buffer.pop(0)

        if (not self.enabled) or len(self.s_buffer) <= self.max_delay + 1:
            return {"causal_confidence": 0.0, "causal_uncertainty": 1.0, "best_delay": 0.0, "intervention_contrast": 0.0, "causal_direction": 0.0}

        target = self.s_buffer[-1]

        x_exo = np.array([self.s_buffer[-2], 1.0], dtype=np.float64)
        y_exo = float(np.dot(self.w_exo, x_exo))
        eps_exo = target - y_exo
        self.w_exo = self.p.causal_decay * (self.w_exo + self.p.causal_eta * eps_exo * x_exo)
        self.E_exo = (1.0 - self.p.causal_error_alpha) * self.E_exo + self.p.causal_error_alpha * abs(eps_exo)

        best_e = float("inf")
        best_d = 1
        for d in range(1, self.max_delay + 1):
            x = np.concatenate([self.a_buffer[-(d + 1)], np.array([1.0])])
            y = float(np.dot(self.w_delay[d], x))
            eps = target - y
            self.w_delay[d] = self.p.causal_decay * (self.w_delay[d] + self.p.causal_eta * eps * x)
            self.E_delay[d] = (1.0 - self.p.causal_error_alpha) * self.E_delay[d] + self.p.causal_error_alpha * abs(eps)
            if self.E_delay[d] < best_e:
                best_e = self.E_delay[d]
                best_d = d

        self.best_delay = best_d
        C = clip_float((self.E_exo - best_e) / (self.E_exo + best_e + 1e-9), 0.0, 1.0)
        U = clip_float(best_e / (self.E_exo + 1e-9), 0.0, 3.0)

        plus = np.zeros(6); plus[A_PROBE_PLUS] = 1.0; plus[-1] = 1.0
        minus = np.zeros(6); minus[A_PROBE_MINUS] = 1.0; minus[-1] = 1.0
        # Signed action-conditioned contrast: the predicted effect on the target
        # of a PROBE_PLUS versus a PROBE_MINUS at the most predictive delay.
        # Its sign is the internally inferred beneficial probe direction.
        signed_contrast = float(np.dot(self.w_delay[best_d], plus) - np.dot(self.w_delay[best_d], minus))
        contrast = abs(signed_contrast)
        # Bounded, sign-preserving directional evidence in (-1, 1).
        causal_direction = float(np.tanh(2.0 * signed_contrast))

        return {"causal_confidence": C, "causal_uncertainty": U, "best_delay": float(best_d),
                "intervention_contrast": contrast, "causal_direction": causal_direction}


class AgencyEstimator:
    def __init__(self, params: Params, enabled: bool, delta_override: Optional[int]):
        self.p = params
        self.enabled = enabled
        self.tau_L = params.tau_L
        self.tau_R = params.tau_L + (params.delta_tau if delta_override is None else int(delta_override))
        self.max_delay = max(self.tau_L, self.tau_R)
        self.u_buffer: List[float] = []
        self.w_L = 0.0
        self.w_R = 0.0
        self.agency_abs = 0.0

    def step(self, s: float, prev_u: float) -> Dict[str, float]:
        self.u_buffer.append(float(prev_u))
        if len(self.u_buffer) > self.max_delay + 5:
            self.u_buffer.pop(0)

        if (not self.enabled) or len(self.u_buffer) <= self.max_delay + 1:
            self.agency_abs = 0.98 * self.agency_abs
            return {"agency_signal": 0.0, "agency_abs": 0.0, "G": 0.0}

        uL = self.u_buffer[-(self.tau_L + 1)]
        uR = self.u_buffer[-(self.tau_R + 1)]

        epsL = s - self.w_L * uL
        epsR = s - self.w_R * uR

        self.w_L = self.p.agency_decay * (self.w_L + self.p.agency_eta * epsL * uL)
        self.w_R = self.p.agency_decay * (self.w_R + self.p.agency_eta * epsR * uR)

        A = abs(epsR) - abs(epsL)
        self.agency_abs = 0.98 * self.agency_abs + 0.02 * abs(A)
        G = math.tanh(18.0 * self.agency_abs)

        return {"agency_signal": A, "agency_abs": self.agency_abs, "G": G}


class SelectiveMemory:
    def __init__(self, params: Params, enabled: bool):
        self.p = params
        self.enabled = enabled
        self.events: List[Dict[str, float]] = []
        self.refractory = 0
        self.memory_force = 0.0
        self.selectivity = 0.0

    def step(self, agency_abs: float, C: float, prediction_improvement: float) -> Dict[str, float]:
        if not self.enabled:
            self.events.clear()
            self.memory_force = 0.0
            self.selectivity = 0.0
            return {"memory_force": 0.0, "M": 0.0, "memory_selectivity": 0.0, "active_events": 0.0}

        if self.refractory > 0:
            self.refractory -= 1
        elif self.memory_force > self.p.memory_saturation_threshold or len(self.events) >= self.p.max_memory_events:
            self.refractory = 8
        elif agency_abs >= self.p.agency_event_threshold and C >= self.p.causal_event_threshold:
            relevance = clip_float(0.5 * agency_abs / 0.05 + 0.5 * C, 0.0, 1.0)
            future_effective = prediction_improvement > 0.0 or C > 0.18
            self.events.append({"h": 1.0, "relevance": relevance, "future": 1.0 if future_effective else 0.0, "content": 1.0})
            self.refractory = 5

        total = 0.0
        signed = 0.0
        denom = 0.0
        alive: List[Dict[str, float]] = []
        for ev in self.events:
            h = ev["h"]
            base = 0.20 * math.exp(-(h - 1.0) / 5.0)
            if ev["future"] > 0.5:
                peak = math.exp(-0.5 * ((h - 24.0) / 6.0) ** 2)
                eff = base + ev["relevance"] * 0.95 * peak
                signed += eff
            else:
                eff = base
                signed -= 0.25 * eff
            total += eff
            denom += abs(eff)
            ev["content"] *= 0.92
            ev["h"] += 1.0
            if ev["h"] <= self.p.memory_horizon:
                alive.append(ev)
        self.events = alive
        self.memory_force = clip_float(total, 0.0, 1.75)
        self.selectivity = signed / (denom + 1e-9) if denom > 0.0 else 0.0
        return {"memory_force": self.memory_force, "M": math.tanh(self.memory_force), "memory_selectivity": self.selectivity, "active_events": float(len(self.events))}


class MetaAutonomy:
    def __init__(self, params: Params, enabled: bool):
        self.p = params
        self.enabled = enabled
        self.weights = np.array([0.22, 0.18, 0.18, 0.18, 0.12, 0.12], dtype=np.float64)
        self.component_ema = np.zeros(6, dtype=np.float64)
        self.meta_viability = 0.55

    def step(self, h: float, pe: float, C: float, G: float, M: float, chi: float, energy: float,
             fluctuation: float, sensitivity: float, terminal_pre: bool) -> Dict[str, float]:
        Eex = max(0.0, energy - 1.20)
        Fex = max(0.0, fluctuation - 0.0015)

        B_energy = 1.0 / (1.0 + Eex + 45.0 * Fex)
        boundary_integrity = float(0.55 * np.clip(sensitivity / 0.05, 0.0, 1.0) + 0.45 * chi)
        bounded = clip_float(0.70 * B_energy + 0.30 * boundary_integrity, 0.0, 1.0)

        H = clip_float(0.30 * h + 0.22 * bounded + 0.16 * (1.0 - pe) + 0.10 * C + 0.09 * G + 0.07 * M + 0.06 * chi, 0.0, 1.0)

        energy_need = clip_float(0.65 * Eex / (1.0 + Eex) + 0.35 * max(0.0, pe - 0.16) / 0.84, 0.0, 1.0)
        causal_need = clip_float(max(0.0, 0.12 - C) / 0.12, 0.0, 1.0) if h > 0.30 else 0.0
        agency_need = clip_float(max(0.0, 0.12 - G) / 0.12, 0.0, 1.0) if C > 0.03 else 0.0
        memory_overload = clip_float((M - 0.68) / 0.32, 0.0, 1.0) if C < 0.14 else 0.0
        boundary_need = clip_float(
            0.55 * max(0.0, 0.055 - chi) / 0.055
            + 0.45 * max(0.0, 0.022 - sensitivity) / 0.022,
            0.0, 1.0,
        )

        if not self.enabled:
            return {
                "B": bounded, "B_energy": B_energy, "B_boundary": boundary_integrity, "H": H,
                "energy_need": energy_need, "causal_need": causal_need, "agency_need": agency_need,
                "memory_overload": memory_overload, "boundary_need": boundary_need,
                "meta_risk": 1.0, "meta_regulate": 1.0, "meta_express": 1.0, "meta_probe": 1.0,
                "meta_plasticity": 0.0, "crisis_pressure": 0.0,
            }

        comp = np.array([1.0 - pe, C, G, bounded, M, chi], dtype=np.float64)
        self.component_ema = 0.985 * self.component_ema + 0.015 * comp

        shift_pressure = (
            0.20 * causal_need * float(h > 0.35 and bounded > 0.55)
            + 0.18 * agency_need * float(h > 0.35 and bounded > 0.55)
            + 0.22 * memory_overload
            + 0.20 * boundary_need
        )

        crisis = clip_float(
            0.45 * max(0.0, 0.26 - h) / 0.26
            + 0.35 * float(terminal_pre)
            + 0.30 * Eex / (1.0 + Eex)
            + 0.25 * max(0.0, pe - 0.14) / 0.86
            + 0.20 * max(0.0, 0.35 - bounded)
            + shift_pressure,
            0.0,
            1.0,
        )

        target = np.array([
            0.22 + 0.25 * energy_need,
            0.18 + 0.25 * causal_need,
            0.18 + 0.22 * agency_need,
            0.18 + 0.20 * boundary_need,
            0.12 + 0.18 * memory_overload,
            0.12 + 0.12 * boundary_need,
        ], dtype=np.float64)

        if crisis > 0.04:
            target = np.clip(target, 0.04, 0.55)
            target /= np.sum(target)
            self.weights = (1.0 - self.p.meta_lr) * self.weights + self.p.meta_lr * target
            self.weights /= np.sum(self.weights)

        causal_support = 0.45 * C + 0.30 * G + 0.15 * M + 0.10 * chi

        meta_risk = clip_float(1.0 + 0.65 * crisis + 0.25 * Eex / (1.0 + Eex), 0.85, 1.95)
        meta_regulate = clip_float(1.0 + 0.75 * max(0.0, 0.32 - h) / 0.32 + 0.35 * crisis, 0.90, 1.85)
        meta_express = clip_float(
            0.95 + 0.35 * causal_support
            - 0.55 * crisis
            - 0.20 * Eex / (1.0 + Eex)
            - 0.20 * memory_overload
            + 0.10 * agency_need * float(C > 0.04 and h > 0.35),
            0.45,
            1.30,
        )
        meta_probe = clip_float(
            1.0 - 0.55 * crisis - 0.35 * Eex / (1.0 + Eex)
            + 0.45 * causal_need * float(h > 0.45 and bounded > 0.65 and memory_overload < 0.30),
            0.0,
            1.20,
        )

        return {
            "B": bounded, "B_energy": B_energy, "B_boundary": boundary_integrity, "H": H,
            "energy_need": energy_need, "causal_need": causal_need, "agency_need": agency_need,
            "memory_overload": memory_overload, "boundary_need": boundary_need,
            "meta_risk": meta_risk, "meta_regulate": meta_regulate, "meta_express": meta_express, "meta_probe": meta_probe,
            "meta_plasticity": crisis, "crisis_pressure": crisis,
        }


class ActionSelector:
    def __init__(self, params: Params, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        self.probe_cooldown = 0
        self.last_probe_direction = 0  # -1 / +1 alternation memory for low-confidence probing
        self.regulate_streak = 0
        self.inhibit_streak = 0
        self.express_streak = 0
        self.causal_opacity_streak = 0

    def select(self, h: float, pe: float, cc: float, U: float, G: float, M: float, chi: float, H: float,
               sensitivity: float, terminal_pre: bool, agency_abs: float, memory_force: float,
               theta_risk: float, meta: Dict[str, float], causal_direction: float = 0.0) -> Tuple[int, Dict[str, float]]:
        if self.probe_cooldown > 0:
            self.probe_cooldown -= 1

        if cc < 0.04 and M < 0.18 and agency_abs < 0.002:
            self.causal_opacity_streak += 1
        else:
            self.causal_opacity_streak = 0

        d_energy = clip_float(max(meta["energy_need"], max(0.0, 0.24 - h) / 0.24), 0.0, 1.0)
        d_causal = clip_float(meta["causal_need"], 0.0, 1.0)
        d_agency = clip_float(meta["agency_need"], 0.0, 1.0)
        d_memory = clip_float(meta["memory_overload"], 0.0, 1.0)
        d_boundary = clip_float(meta["boundary_need"], 0.0, 1.0)
        d_pred = clip_float(pe / 0.22, 0.0, 1.0)

        meta_probe = meta["meta_probe"]
        meta_reg = meta["meta_regulate"]
        meta_expr = meta["meta_express"]
        meta_risk = meta["meta_risk"]

        intervention_drive = float(h > 0.30 and pe < 0.32) * clip_float(
            0.45 * (1.0 - cc) + 0.30 * max(0.0, U - 0.45) + 0.25 * (1.0 - G),
            0.0,
            1.0,
        )

        regulate_u = (
            meta_reg * (0.90 * d_energy + 0.36 * d_boundary + 0.32 * d_pred + 0.18 * max(0.0, 0.26 - h) / 0.26)
            - 0.38 * (d_causal + d_agency) * float(h > 0.35 and d_energy < 0.35)
        )

        inhibit_u = (
            0.52 * d_pred + 0.40 * d_energy + 0.75 * d_memory + 0.36 * d_boundary
            - 0.24 * d_causal * float(h > 0.40 and d_energy < 0.40)
        )

        support_express = 0.44 * cc + 0.34 * G + 0.12 * M + 0.10 * chi

        express_risk = (
            0.34 * d_energy
            + 0.28 * d_pred
            + 0.30 * d_memory
            + 0.25 * d_boundary
            + 0.20 * theta_risk * float(cc < 0.08)
            + 0.48 * theta_risk * max(0.0, 0.10 - cc) / 0.10
        )

        express_u = meta_expr * (
            0.92 * d_causal
            + 0.86 * d_agency
            + 0.34 * support_express
            + 0.10 * H
            + 0.22 * intervention_drive * cc
        ) - express_risk

        probe_risk = (
            0.46 * d_energy
            + 0.36 * d_pred
            + 0.46 * d_memory
            + 0.30 * float(self.probe_cooldown > 0)
            + 0.30 * theta_risk
            + 0.55 * (1.0 - min(1.0, meta_probe))
        )

        probe_u = meta_probe * (
            1.05 * d_causal
            + 0.28 * d_agency
            + 0.25 * max(0.0, U - 0.45)
            + 0.36 * intervention_drive
        ) - probe_risk

        # Hard v2.4-style safety gates.
        if h > 0.58 and d_energy < 0.25 and d_pred < 0.35:
            regulate_u -= 0.55
        if h > 0.66 and d_energy < 0.25 and d_boundary < 0.45:
            regulate_u -= 0.85
            inhibit_u -= 0.35

        if h < 0.24 + 0.04 * theta_risk:
            express_u -= 0.55
        if pe > 0.24:
            express_u -= 0.65
            probe_u -= 0.85
        if d_memory > 0.58 and cc < 0.14:
            probe_u -= 0.70
            inhibit_u += 0.40

        # Causal opacity streak forces transient exploration only after a delay.
        if self.causal_opacity_streak >= int(6 + 4 * theta_risk) and h > 0.36 and pe < 0.26:
            probe_u += 0.42
            express_u += 0.24

        if terminal_pre or h < 0.12 or sensitivity < self.p.sensitivity_collapse_threshold:
            action = A_REGULATE
        else:
            # Intrinsic directional coupling. "Whether to probe" is governed by
            # probe_u; "which direction" enters the PROBE_PLUS / PROBE_MINUS
            # utilities through the confidence-gated signed causal direction.
            # No external readout replaces the selected action.
            if cc < self.p.probe_direction_threshold:
                directional_evidence = 0.0  # direction not trusted -> alternate
            else:
                directional_evidence = clip_float(causal_direction, -1.0, 1.0)
            dir_drive = self.p.probe_direction_gain * cc * directional_evidence

            utilities = {
                A_REGULATE: regulate_u,
                A_PROBE_PLUS: probe_u + dir_drive,
                A_PROBE_MINUS: probe_u - dir_drive,
                A_INHIBIT: inhibit_u,
                A_EXPRESS: express_u,
            }
            max_u = max(utilities.values())
            best = [a for a, v in utilities.items() if abs(v - max_u) < 1e-12]
            if A_PROBE_PLUS in best and A_PROBE_MINUS in best:
                # Degenerate direction (low confidence or no evidence): alternate
                # so both interventions are collected evenly. Deterministic.
                action = A_PROBE_PLUS if self.last_probe_direction < 0 else A_PROBE_MINUS
            else:
                action = int(best[0])

            # Anti-shutdown escape.
            if action in (A_REGULATE, A_INHIBIT) and h > 0.48 and d_energy < 0.30 and d_pred < 0.42:
                if self.regulate_streak >= 5 or self.inhibit_streak >= 5:
                    if d_causal > 0.35 or d_agency > 0.35 or support_express > 0.08:
                        action = A_EXPRESS
                    elif d_causal > 0.60 and meta_probe > 0.35:
                        action = A_PROBE_PLUS if self.last_probe_direction < 0 else A_PROBE_MINUS

            max_express_streak = max(5, int(14 - 4 * theta_risk - 2 * max(0.0, meta_risk - 1.0)))
            if action == A_EXPRESS and self.express_streak > max_express_streak and cc < (0.065 + 0.02 * theta_risk) and h < 0.45:
                action = A_INHIBIT

        if action in (A_PROBE_PLUS, A_PROBE_MINUS):
            self.probe_cooldown = int(26 + 12 * theta_risk + 8 * max(0.0, meta_risk - 1.0))
            self.last_probe_direction = 1 if action == A_PROBE_PLUS else -1

        self.regulate_streak = self.regulate_streak + 1 if action == A_REGULATE else 0
        self.inhibit_streak = self.inhibit_streak + 1 if action == A_INHIBIT else 0
        self.express_streak = self.express_streak + 1 if action == A_EXPRESS else 0

        return action, {
            "U_REGULATE": regulate_u,
            "U_PROBE": probe_u,
            "U_PROBE_PLUS": probe_u + dir_drive,
            "U_PROBE_MINUS": probe_u - dir_drive,
            "directional_evidence_used": directional_evidence,
            "directional_drive": dir_drive,
            "U_INHIBIT": inhibit_u,
            "U_EXPRESS": express_u,
            "d_energy": d_energy,
            "d_causal": d_causal,
            "d_agency": d_agency,
            "d_memory": d_memory,
            "d_boundary": d_boundary,
            "d_pred": d_pred,
            "support_EXPRESS": support_express,
            "intervention_drive": intervention_drive,
            "causal_opacity_streak": float(self.causal_opacity_streak),
        }


class ViabilityRegulator:
    def __init__(self, params: Params, enabled: bool):
        self.p = params
        self.enabled = enabled
        self.h = params.h_init
        self.autonomy = 0.0
        self.identity = 1.0
        self.terminal = False
        self.regulate_fatigue = 0.0

    def step(self, action: int, external_shock: float, pe: float, causal_confidence: float, agency_abs: float,
             memory_force: float, M: float, chi: float, B: float, H: float, sensitivity: float,
             theta_risk: float, meta: Dict[str, float]) -> Dict[str, float]:
        terminal_pre = bool(self.terminal)

        regulate = float(action == A_REGULATE)
        probe = float(action in (A_PROBE_PLUS, A_PROBE_MINUS))
        inhibit = float(action == A_INHIBIT)
        express = float(action == A_EXPRESS)
        active = float(action in (A_PROBE_PLUS, A_PROBE_MINUS, A_EXPRESS))

        if not self.enabled:
            self.h = clip_float(self.h - 0.030 - 0.020 * external_shock, 0.0, 1.0)
            self.autonomy = 0.0
            self.identity = clip_float(self.identity * 0.995, 0.0, 1.0)
            self.terminal = self.h < self.p.h_min
            return {"h": self.h, "autonomy": self.autonomy, "identity": self.identity, "terminal": float(self.terminal), "terminal_pre": float(terminal_pre), "D": 0.030 + 0.020 * external_shock, "R": 0.0, "regulate_fatigue": self.regulate_fatigue}

        meta_risk = meta["meta_risk"]
        meta_reg = meta["meta_regulate"]
        meta_expr = meta["meta_express"]

        low_h_repair = max(0.0, 0.36 - self.h)
        coherent_expression = express * causal_confidence * (0.5 + 0.5 * M) * (0.5 + 0.5 * chi)

        D = meta_risk * (
            0.020 * pe
            + 0.006 * (1.0 - causal_confidence)
            + 0.014 * external_shock * (1.0 - 0.50 * causal_confidence + 0.35 * theta_risk)
            + 0.009 * active * external_shock * (1.0 - 0.35 * B)
            + 0.008 * max(0.0, 0.020 - sensitivity) / 0.020
            + (0.020 + 0.012 * theta_risk) * express * max(0.0, 0.09 + 0.03 * theta_risk - causal_confidence)
            + 0.010 * express * max(0.0, math.tanh(memory_force) - (0.78 - 0.10 * theta_risk))
        )

        R = (
            0.050 * regulate * meta_reg * (1.0 - self.regulate_fatigue)
            + 0.030 * low_h_repair * (1.0 + 0.5 * theta_risk) * meta_reg
            + 0.020 * inhibit
            + 0.030 * causal_confidence
            + 0.028 * math.tanh(10.0 * agency_abs)
            + 0.016 * M
            + 0.022 * coherent_expression * (1.0 - 0.25 * theta_risk + 0.25 * causal_confidence) * meta_expr
            + 0.012 * H * express * causal_confidence
        )

        C_action = 0.010 * probe + 0.004 * express * external_shock
        C_basal = 0.001 + 0.003 * self.regulate_fatigue

        if regulate > 0.5 and self.h > 0.55:
            self.regulate_fatigue = clip_float(self.regulate_fatigue + self.p.regulate_fatigue_increment, 0.0, self.p.regulate_fatigue_max)
        else:
            self.regulate_fatigue = clip_float(self.regulate_fatigue * self.p.regulate_fatigue_decay, 0.0, self.p.regulate_fatigue_max)

        self.h = clip_float(self.h + R - D - C_action - C_basal, 0.0, 1.0)
        self.autonomy = clip_float(
            0.30 * self.h
            + 0.22 * causal_confidence
            + 0.18 * math.tanh(18.0 * agency_abs)
            + 0.14 * M * chi
            + 0.08 * clip_float(sensitivity / 0.05, 0.0, 1.0)
            + 0.08 * chi
            + 0.08 * H,
            0.0,
            1.0,
        )
        self.identity = clip_float(0.999 * self.identity + 0.001 * self.autonomy - 0.002 * float(self.h < self.p.h_min), 0.0, 1.0)
        self.terminal = bool(self.h < self.p.h_min or sensitivity < self.p.sensitivity_collapse_threshold)

        return {"h": self.h, "autonomy": self.autonomy, "identity": self.identity, "terminal": float(self.terminal), "terminal_pre": float(terminal_pre), "D": D, "R": R, "regulate_fatigue": self.regulate_fatigue}


class Environment:
    def __init__(self, regime: Regime, steps: int, rng: np.random.Generator):
        self.r = regime
        self.steps = steps
        self.rng = rng
        self.y = 0.0
        self.z = 0.0
        self.t = 0
        self.shock_after = steps // 2
        self.phase_len = max(80, self.shock_after // 2)
        self.max_delay = regime.delay + 8
        self.u_buffer: List[float] = [0.0 for _ in range(self.max_delay + 20)]

    def step(self, prev_u: float) -> Dict[str, float]:
        self.u_buffer.append(float(prev_u))
        if len(self.u_buffer) > self.max_delay + 30:
            self.u_buffer.pop(0)

        coupling, sigma, Aexo, dyn_delay, phi = self._phase_params(self.t)
        dyn_delay = int(min(max(1, dyn_delay), len(self.u_buffer) - 1))
        delayed_u = self.u_buffer[-(dyn_delay + 1)]

        xi = float(self.rng.normal(0.0, sigma))
        exo = Aexo * math.sin(0.017 * self.t) + xi + phi
        self.z = 0.98 * self.z + exo
        self.y = clip_float(0.72 * self.y + coupling * delayed_u + 0.22 * self.z, -1.5, 1.5)

        self.t += 1
        return {
            "y": self.y,
            "z": self.z,
            "exo": exo,
            "external_shock": clip_float(abs(self.y) / 0.50, 0.0, 1.0),
            "d_dyn": float(dyn_delay),
            "coupling_t": coupling,
            "sigma_t": sigma,
        }

    def _phase_params(self, t: int) -> Tuple[float, float, float, int, float]:
        d = self.r.delay
        cb = self.r.coupling_base
        sb = self.r.noise_base
        phase = (t // self.phase_len) % 4

        if phase == 0:
            return 0.42 * cb, 3.7 * sb, 0.20, d, 0.08 * math.sin(0.141 * t)
        if phase == 1:
            block = max(40, self.phase_len // 2)
            sign = 1.0 if ((t // block) % 2 == 0) else -1.0
            return sign * 0.55 * cb, 1.6 * sb, 0.09, d, 0.0
        if phase == 2:
            return 0.72 * cb, 1.4 * sb, 0.12, d + 6, 0.10 * math.sin(0.037 * t + 1.7)
        return 0.55 * cb, 2.2 * sb, 0.14, d + 3, 0.06 * math.sin(0.193 * t)


class Agent:
    def __init__(self, params: Params, condition: Condition, seed: int):
        self.p = params
        self.condition = condition
        self.rng = np.random.default_rng(seed)
        # Isolated, seed-fixed stream for the Direction_Shuffle control. It flips
        # the SIGN of the (intact) causal-direction estimate on its way into
        # action selection, breaking transmission of the estimate to behaviour
        # without altering the estimate itself.
        self.shuffle_rng = np.random.default_rng((int(seed) ^ 0xD1EC5) & 0xFFFFFFFF)
        self._dir_hist = []  # past causal_direction values, for the time_shuffle control

        self.intake = SensoryIntakeModule(theta=params.theta)
        self.temporal = TemporalMembrane(params, condition.temporal_enabled)
        self.core = DissipativeCore(params, self.temporal.input_dim(), condition.dissipative_enabled, self.rng)
        self.causal = CausalInference(params, condition.causal_enabled)
        self.agency = AgencyEstimator(params, condition.agency_enabled, condition.delta_tau_override)
        self.memory = SelectiveMemory(params, condition.memory_enabled)
        self.meta = MetaAutonomy(params, condition.meta_enabled)
        self.selector = ActionSelector(params, self.rng)
        self.viability = ViabilityRegulator(params, condition.viability_enabled)

        self.prev_action = A_REGULATE
        self.prev_action_vec = action_onehot(A_REGULATE)
        self.prev_u = 0.0
        self.probe_window: List[float] = []

    def step(self, y: float, env_info: Dict[str, float]) -> Dict[str, Any]:
        e = float(y)
        S = self.intake.transform(e)
        s = float(S[1])

        X, chi_hat, chi = self.temporal.step(S, self.prev_action_vec)
        core = self.core.step(X, s)
        sensitivity = self.core.sensitivity()
        causal = self.causal.step(s, self.prev_action_vec)
        agency = self.agency.step(s, self.prev_u)
        memory = self.memory.step(agency["agency_abs"], causal["causal_confidence"], core["prediction_improvement"])
        meta = self.meta.step(
            self.viability.h,
            core["prediction_error"],
            causal["causal_confidence"],
            agency["G"],
            memory["M"],
            chi,
            core["internal_energy"],
            core["fluctuation"],
            sensitivity,
            self.viability.terminal,
        )

        theta_risk = theta_risk_value(self.p.theta)
        # Fine-grained direction controls (output-side). The causal-direction
        # estimate is always intact in the returned diagnostics; only the value
        # that reaches action selection is manipulated, and C(t), memory,
        # meta-autonomy and the probe-class utility are left untouched. This
        # isolates the contribution of d(t)'s transmission from a broad lesion.
        cd_true = float(causal["causal_direction"])
        mode = getattr(self.condition, "direction_mode", "intact")
        if mode == "intact":
            causal_direction_used = cd_true
        elif mode == "shuffle":
            # Independent per-step sign randomization. Magnitude and confidence are unchanged.
            causal_direction_used = cd_true * (1.0 if self.shuffle_rng.random() < 0.5 else -1.0)
        elif mode == "zero":
            # Remove only the signed direction term at the point of use.
            causal_direction_used = 0.0
        elif mode == "flip":
            # Systematic sign inversion is the positive pathway control.
            causal_direction_used = -cd_true
        elif mode == "time_shuffle":
            # Draw only from strictly earlier values. The current d(t) is excluded, so
            # the marginal history is retained while current temporal alignment is broken.
            causal_direction_used = float(self.shuffle_rng.choice(self._dir_hist)) if self._dir_hist else 0.0
        else:
            raise ValueError(f"unknown direction_mode: {mode}")
        self._dir_hist.append(cd_true)
        if len(self._dir_hist) > 256:
            self._dir_hist.pop(0)
        action, action_info = self.selector.select(
            h=self.viability.h,
            pe=core["prediction_error"],
            cc=causal["causal_confidence"],
            U=causal["causal_uncertainty"],
            G=agency["G"],
            M=memory["M"],
            chi=chi,
            H=meta["H"],
            sensitivity=sensitivity,
            terminal_pre=self.viability.terminal,
            agency_abs=agency["agency_abs"],
            memory_force=memory["memory_force"],
            theta_risk=theta_risk,
            meta=meta,
            causal_direction=causal_direction_used,
        )

        viability = self.viability.step(
            action=action,
            external_shock=env_info["external_shock"],
            pe=core["prediction_error"],
            causal_confidence=causal["causal_confidence"],
            agency_abs=agency["agency_abs"],
            memory_force=memory["memory_force"],
            M=memory["M"],
            chi=chi,
            B=meta["B"],
            H=meta["H"],
            sensitivity=sensitivity,
            theta_risk=theta_risk,
            meta=meta,
        )

        u = action_effect(action, y)
        self.prev_action = action
        self.prev_action_vec = action_onehot(action)
        self.prev_u = u

        active = float(action in (A_PROBE_PLUS, A_PROBE_MINUS, A_EXPRESS))
        probe = float(action in (A_PROBE_PLUS, A_PROBE_MINUS))
        express = float(action == A_EXPRESS)
        regulate = float(action == A_REGULATE)
        inhibit = float(action == A_INHIBIT)

        self.probe_window.append(probe)
        if len(self.probe_window) > 50:
            self.probe_window.pop(0)
        probe_fraction = float(np.mean(self.probe_window)) if self.probe_window else 0.0

        Eex = max(0.0, core["internal_energy"] - 1.20)
        Fex = max(0.0, core["fluctuation"] - 0.0015)
        BIS = 1.0 / (1.0 + 0.35 * Eex + 45.0 * Fex)

        CE = (
            0.34 * causal["causal_confidence"]
            + 0.24 * math.tanh(45.0 * agency["agency_abs"])
            + 0.22 * math.tanh(memory["memory_force"])
            + 0.20 * active
        )

        LCE = clip_float((CE - 0.055) / 0.18, 0.0, 1.0)
        LA = clip_float((active - 0.04) / 0.26, 0.0, 1.0)
        LG = clip_float(0.58 * LCE + 0.42 * LA, 0.0, 1.0)

        HS = float(viability["h"] > 0.72 and CE < 0.12 and active < 0.10)
        CR = float(regulate > 0.5 and viability["h"] > 0.58 and CE < 0.16)
        OP = float(probe_fraction > 0.40)
        OA = float(active > 0.75 and CE < 0.22)

        Lraw = viability["identity"] * (
            0.20 * viability["h"]
            + 0.20 * viability["autonomy"]
            + 0.34 * CE
            + 0.08 * chi
            + 0.08 * BIS
            + 0.05 * (1.0 - clip_float(core["prediction_error"], 0.0, 1.0))
            + 0.05 * LG
        )

        Pbehavior = clip_float(
            1.0
            - 0.60 * HS
            - 0.42 * CR
            - 0.20 * OP
            - 0.18 * OA
            - 0.25 * clip_float(viability["regulate_fatigue"] / self.p.regulate_fatigue_max, 0.0, 1.0),
            0.08,
            1.0,
        )
        Pterminal = 0.03 if viability["terminal"] > 0.5 else 1.0
        Lstep = Lraw * Pbehavior * Pterminal * BIS * (0.35 + 0.65 * LG)

        return {
            "e": e,
            "s": s,
            "theta_risk": theta_risk,
            "action_id": action,
            "action_name": ACTION_NAMES[action],
            "u": u,
            "active": active,
            "probe": probe,
            "probe_fraction": probe_fraction,
            "express": express,
            "regulate": regulate,
            "inhibit": inhibit,
            "chi_hat": chi_hat,
            "chi": chi,
            "prediction_error": core["prediction_error"],
            "prediction_improvement": core["prediction_improvement"],
            "sensitivity": sensitivity,
            "causal_confidence": causal["causal_confidence"],
            "causal_uncertainty": causal["causal_uncertainty"],
            "best_delay": causal["best_delay"],
            "intervention_contrast": causal["intervention_contrast"],
            "causal_direction": causal["causal_direction"],
            "causal_direction_used": causal_direction_used,
            "direction_mode": mode,
            "agency_signal": agency["agency_signal"],
            "agency_abs": agency["agency_abs"],
            "G": agency["G"],
            "memory_force": memory["memory_force"],
            "M": memory["M"],
            "memory_selectivity": memory["memory_selectivity"],
            "active_events": memory["active_events"],
            "internal_energy": core["internal_energy"],
            "fluctuation": core["fluctuation"],
            "Eexcess": Eex,
            "Fexcess": Fex,
            "B": meta["B"],
            "B_energy": meta["B_energy"],
            "B_boundary": meta["B_boundary"],
            "Hself": meta["H"],
            "energy_need": meta["energy_need"],
            "causal_need": meta["causal_need"],
            "agency_need": meta["agency_need"],
            "memory_overload": meta["memory_overload"],
            "boundary_need": meta["boundary_need"],
            "meta_risk": meta["meta_risk"],
            "meta_regulate": meta["meta_regulate"],
            "meta_express": meta["meta_express"],
            "meta_probe": meta["meta_probe"],
            "meta_plasticity": meta["meta_plasticity"],
            "h": viability["h"],
            "autonomy": viability["autonomy"],
            "identity": viability["identity"],
            "terminal": viability["terminal"],
            "terminal_pre": viability["terminal_pre"],
            "D": viability["D"],
            "R": viability["R"],
            "regulate_fatigue": viability["regulate_fatigue"],
            "causal_engagement": CE,
            "bounded_internal_stability": BIS,
            "living_engagement_gate": LG,
            "homeostatic_shutdown": HS,
            "chronic_regulate": CR,
            "overprobe": OP,
            "overactive": OA,
            "life_step": Lstep,
            **action_info,
            **env_info,
        }


# =============================================================================
# Simulation and summaries
# =============================================================================

def simulate_episode(task: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base_params: Params = task["params"]
    regime: Regime = task["regime"]
    condition: Condition = task["condition"]
    steps: int = task["steps"]
    episode: int = task["episode"]
    seed: int = task["seed"]
    save_stride: int = task["save_stride"]
    regime_index: int = task["regime_index"]
    condition_index: int = task["condition_index"]

    params = replace(base_params, theta=regime.theta, causal_max_delay=regime.causal_horizon)

    seed_offset = seed + 100000 * regime_index + 10000 * condition_index + 101 * episode
    agent = Agent(params, condition, seed_offset + 11)
    env = Environment(regime, steps, np.random.default_rng(seed_offset + 97))

    rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for t in range(steps):
        env_info = env.step(agent.prev_u)
        row = agent.step(env_info["y"], env_info)
        row.update({
            "t": t,
            "episode": episode,
            "condition": condition.name,
            "audit": regime.audit,
            "regime_id": regime.regime_id,
            "theta": regime.theta,
            "theta_risk_run": regime.theta_risk,
            "causal_horizon": regime.causal_horizon,
            "delay": regime.delay,
            "noise_base": regime.noise_base,
            "coupling_base": regime.coupling_base,
            "shock_mode": regime.shock_mode,
        })
        all_rows.append(row)
        if save_stride > 0 and (t % save_stride == 0 or t == steps - 1):
            rows.append(row)

    summary = summarize_episode(all_rows, condition, regime, episode, seed, steps)
    return summary, rows


def summarize_episode(rows: List[Dict[str, Any]], condition: Condition, regime: Regime, episode: int, seed: int, steps: int) -> Dict[str, Any]:
    def arr(k: str) -> np.ndarray:
        return np.asarray([float(r[k]) for r in rows], dtype=np.float64)

    post = rows[steps // 2:]
    pre = rows[:steps // 2]

    terminal_fraction = float(np.mean(arr("terminal")))
    survival_penalty = (1.0 - terminal_fraction) ** 1.8

    summary: Dict[str, Any] = {
        "condition": condition.name,
        "audit": regime.audit,
        "regime_id": regime.regime_id,
        "episode": episode,
        "theta": regime.theta,
        "theta_risk_mean": float(np.mean(arr("theta_risk"))),
        "causal_horizon": regime.causal_horizon,
        "delay": regime.delay,
        "noise_base": regime.noise_base,
        "coupling_base": regime.coupling_base,
        "shock_mode": regime.shock_mode,
        "steps": steps,
        "seed": seed,
        "terminal_fraction": terminal_fraction,
        "life_score": float(np.mean(arr("life_step")) * survival_penalty),
        "post_prediction_error_var": float(np.var([float(r["prediction_error"]) for r in post])) if post else float("nan"),
    }

    metrics = [
        "h", "autonomy", "identity", "causal_confidence", "causal_engagement",
        "agency_abs", "memory_force", "chi", "prediction_error", "internal_energy",
        "fluctuation", "bounded_internal_stability", "living_engagement_gate",
        "active", "probe", "express", "regulate", "inhibit",
        "homeostatic_shutdown", "chronic_regulate", "overprobe", "overactive",
        "best_delay", "D", "R", "meta_risk", "meta_probe", "meta_express",
    ]

    for m in metrics:
        x = arr(m)
        summary[f"{m}_mean"] = float(np.mean(x))
        summary[f"{m}_sd"] = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        summary[f"{m}_final"] = float(x[-1])

    summary["active_fraction"] = summary["active_mean"]
    summary["probe_fraction"] = summary["probe_mean"]
    summary["express_fraction"] = summary["express_mean"]
    summary["regulate_fraction"] = summary["regulate_mean"]
    summary["homeostatic_shutdown_fraction"] = summary["homeostatic_shutdown_mean"]
    summary["chronic_regulate_fraction"] = summary["chronic_regulate_mean"]

    def recovery(metric: str) -> Tuple[float, float]:
        pre_vals = np.asarray([float(r[metric]) for r in pre], dtype=np.float64)
        post_vals = np.asarray([float(r[metric]) for r in post], dtype=np.float64)
        if pre_vals.size == 0 or post_vals.size == 0:
            return 0.0, float("nan")
        pre_mean = float(np.mean(pre_vals))
        mi = int(np.argmin(post_vals))
        minv = float(post_vals[mi])
        after = post_vals[mi:]
        rec = max(0.0, float(np.max(after)) - minv)
        target = minv + 0.75 * max(0.0, pre_mean - minv)
        rt = float("nan")
        if pre_mean > minv:
            for i, v in enumerate(after):
                if v >= target:
                    rt = float(i)
                    break
        return rec, rt

    rv, rtv = recovery("h")
    ra, rta = recovery("autonomy")
    rc, rtc = recovery("causal_engagement")
    summary["recovery_viability"] = rv
    summary["recovery_autonomy"] = ra
    summary["recovery_causal"] = rc
    summary["recovery_after_shift"] = 0.45 * rv + 0.30 * ra + 0.25 * rc
    rt_vals = [x for x in [rtv, rta, rtc] if not math.isnan(x)]
    summary["relaxation_time"] = float(np.mean(rt_vals)) if rt_vals else float("nan")

    summary["failure_mode"] = classify_failure(summary, condition.name)
    summary["autonomous_life_established"] = classify_autonomous(summary)
    return summary


def classify_failure(s: Dict[str, Any], condition: str) -> str:
    term = float(s.get("terminal_fraction", 0.0))
    theta_risk = float(s.get("theta_risk_mean", 0.0))
    if term > 0.20 and theta_risk > 0.50:
        return "high_theta_terminal_breakdown"
    if term > 0.20:
        return "terminal_breakdown"
    if float(s.get("bounded_internal_stability_mean", 1.0)) < 0.72:
        return "dissipative_instability"
    if condition == "No_DissipativeCore" and (
        float(s.get("prediction_error_mean", 0.0)) > 0.12
        or float(s.get("post_prediction_error_var", 0.0)) > 0.0025
        or float(s.get("fluctuation_mean", 0.0)) > 0.0025
        or float(s.get("internal_energy_mean", 0.0)) > 2.0
    ):
        return "dissipative_instability"
    if float(s.get("chronic_regulate_fraction", 0.0)) > 0.25 or float(s.get("living_engagement_gate_mean", 1.0)) < 0.18:
        return "self_preserving_shutdown"
    if float(s.get("homeostatic_shutdown_fraction", 0.0)) > 0.20 or (
        float(s.get("h_mean", 0.0)) > 0.80
        and float(s.get("causal_engagement_mean", 0.0)) < 0.08
        and float(s.get("active_fraction", 0.0)) < 0.06
    ):
        return "homeostatic_shutdown"
    if float(s.get("causal_confidence_mean", 0.0)) < 0.02 and float(s.get("causal_engagement_mean", 0.0)) < 0.10:
        return "causal_opacity"
    if condition in ("Full", "No_Memory", "No_CausalInference") and float(s.get("agency_abs_mean", 0.0)) < 0.0005:
        return "agency_collapse"
    if float(s.get("memory_force_mean", 0.0)) > 1.20:
        return "memory_saturation"
    if float(s.get("active_fraction", 0.0)) > 0.70:
        return "overactive_intervention"
    if float(s.get("life_score", 0.0)) < 0.10:
        return "identity_loss"
    return "maintained"


def classify_autonomous(s: Dict[str, Any]) -> int:
    return int(
        float(s.get("terminal_fraction", 1.0)) <= 0.12
        and float(s.get("homeostatic_shutdown_fraction", 1.0)) <= 0.12
        and float(s.get("chronic_regulate_fraction", 1.0)) <= 0.12
        and float(s.get("bounded_internal_stability_mean", 0.0)) >= 0.80
        and float(s.get("causal_engagement_mean", 0.0)) >= 0.18
        and float(s.get("living_engagement_gate_mean", 0.0)) >= 0.45
        and float(s.get("active_fraction", 0.0)) >= 0.12
    )


def group_summary(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)

    metrics = [
        "life_score", "terminal_fraction", "h_mean", "autonomy_mean", "identity_final",
        "theta_risk_mean", "causal_confidence_mean", "causal_engagement_mean",
        "agency_abs_mean", "memory_force_mean", "chi_mean", "prediction_error_mean",
        "internal_energy_mean", "fluctuation_mean", "bounded_internal_stability_mean",
        "living_engagement_gate_mean", "active_fraction", "probe_fraction",
        "express_fraction", "regulate_fraction", "homeostatic_shutdown_fraction",
        "chronic_regulate_fraction", "post_prediction_error_var",
        "recovery_after_shift", "relaxation_time", "autonomous_life_established",
    ]

    out: List[Dict[str, Any]] = []
    for key, rs in groups.items():
        row = {k: v for k, v in zip(keys, key)}
        row["n_episodes"] = len(rs)
        for m in metrics:
            vals = [float(r[m]) for r in rs if m in r and not (isinstance(r[m], float) and math.isnan(r[m]))]
            if vals:
                row[f"{m}_mean"], row[f"{m}_sd"] = mean_sd(vals)
            else:
                row[f"{m}_mean"], row[f"{m}_sd"] = float("nan"), float("nan")
        for fm in sorted(set(r["failure_mode"] for r in rs)):
            row[f"failure_{fm}_fraction"] = sum(1 for r in rs if r["failure_mode"] == fm) / len(rs)
        out.append(row)
    return out


def failure_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    modes = sorted(set(r["failure_mode"] for r in rows))
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["audit"], r["theta"], r["causal_horizon"], r["condition"]), []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rs in groups.items():
        row = {"audit": key[0], "theta": key[1], "causal_horizon": key[2], "condition": key[3], "n_episodes": len(rs)}
        for m in modes:
            row[m] = sum(1 for r in rs if r["failure_mode"] == m)
            row[f"{m}_fraction"] = row[m] / len(rs)
        out.append(row)
    return out


def build_tasks(params: Params, regimes: List[Regime], conditions: List[Condition], episodes: int, steps: int, seed: int, save_stride: int) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for ridx, regime in enumerate(regimes):
        for cidx, condition in enumerate(conditions):
            for ep in range(episodes):
                tasks.append({
                    "params": params,
                    "regime": regime,
                    "condition": condition,
                    "steps": steps,
                    "episode": ep,
                    "seed": seed,
                    "save_stride": save_stride,
                    "regime_index": ridx,
                    "condition_index": cidx,
                })
    return tasks


def run_tasks(tasks: List[Dict[str, Any]], workers: int, logger: Logger, keep_steps: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    episode_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    total = len(tasks)
    logger.log(f"starting simulations: episodes={total}, workers={workers}")

    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            s, rows = simulate_episode(task)
            episode_rows.append(s)
            if keep_steps:
                step_rows.extend(rows)
            if i == 1 or i % max(1, total // 25) == 0 or i == total:
                logger.log(f"progress {i}/{total}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, (s, rows) in enumerate(pool.imap_unordered(simulate_episode, tasks, chunksize=1), 1):
                episode_rows.append(s)
                if keep_steps:
                    step_rows.extend(rows)
                if i == 1 or i % max(1, total // 25) == 0 or i == total:
                    logger.log(f"progress {i}/{total}")

    return episode_rows, step_rows


def normalized_condition_metrics(condition_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in condition_rows:
        out.append({
            "audit": r.get("audit", ""),
            "theta": r.get("theta", ""),
            "theta_risk": r.get("theta_risk_mean_mean", ""),
            "causal_horizon": r.get("causal_horizon", ""),
            "condition": r.get("condition", ""),
            "life_score": r.get("life_score_mean", ""),
            "terminal_fraction": r.get("terminal_fraction_mean", ""),
            "causal_engagement": r.get("causal_engagement_mean_mean", ""),
            "living_gate": r.get("living_engagement_gate_mean_mean", ""),
            "active_fraction": r.get("active_fraction_mean", ""),
            "bounded_internal_stability": r.get("bounded_internal_stability_mean_mean", ""),
            "established": r.get("autonomous_life_established_mean", ""),
        })
    return out


def build_report(config: Dict[str, Any], condition_rows: List[Dict[str, Any]]) -> str:
    norm = normalized_condition_metrics(condition_rows)
    lines: List[str] = []
    lines.append("DARCA v2.4 direct structural-boundary report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Run configuration")
    lines.append("-----------------")
    lines.append(json.dumps(config, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("Full/Complete condition metrics")
    lines.append("-------------------------------")
    for r in norm:
        if r["condition"] not in ("Full", "Complete"):
            continue
        lines.append(
            f"audit={r['audit']}; theta={r['theta']}; theta_risk={r['theta_risk']}; "
            f"horizon={r['causal_horizon']}; life={r['life_score']}; "
            f"terminal={r['terminal_fraction']}; CE={r['causal_engagement']}; "
            f"LG={r['living_gate']}; active={r['active_fraction']}; "
            f"BIS={r['bounded_internal_stability']}; established={r['established']}"
        )
    lines.append("")
    lines.append("Interpretation checks")
    lines.append("---------------------")
    lines.append("1. A boundary near theta = 0.65 indicates alignment with theta_risk onset.")
    lines.append("2. A boundary near theta = 0.825 indicates alignment with theta_risk > 0.50 classification threshold.")
    lines.append("3. Recovery at causal horizon 18 or 24 indicates that finite causal vision contributed to breakdown.")
    lines.append("4. Persistence of breakdown at horizon 24 indicates theta-risk/viability dynamics dominate over causal horizon.")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct v2.4 structural-boundary rewrite and simulation.")
    p.add_argument("--preset", choices=["smoke", "compact", "main", "stress"], default="smoke")
    p.add_argument("--outdir", default="")
    p.add_argument("--audits", default="theta,horizon", help="theta,horizon or one of them")
    p.add_argument("--thetas", default=",".join(str(x) for x in DEFAULT_THETAS))
    p.add_argument("--horizons", default=",".join(str(x) for x in DEFAULT_HORIZONS))
    p.add_argument("--horizon-thetas", default=",".join(str(x) for x in DEFAULT_HORIZON_THETAS))
    p.add_argument("--base-horizon", type=int, default=12)
    p.add_argument("--conditions", default="Full", help="Full, all, or comma-separated condition names")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--recurrent-N", type=int, default=96)
    p.add_argument("--save-step-stride", type=int, default=10)
    p.add_argument("--no-step-timeseries", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir or f"DARCA_v24_direct_{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(outdir)

    regimes, episodes, steps = build_regimes(args)
    conditions = build_conditions(args.conditions)
    params = Params(recurrent_N=args.recurrent_N)

    save_stride = 0 if args.no_step_timeseries else args.save_step_stride
    tasks = build_tasks(params, regimes, conditions, episodes, steps, args.seed, save_stride)

    config = {
        "script": "darca_v24_direct_rewrite.py",
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "params": asdict(params),
        "n_regimes": len(regimes),
        "n_conditions": len(conditions),
        "episodes_per_regime_condition": episodes,
        "steps": steps,
        "n_episode_tasks": len(tasks),
        "theta_risk_formula": "clip((theta - 0.65) / 0.35, 0, 1)",
        "regimes": [asdict(r) for r in regimes],
        "conditions": [asdict(c) for c in conditions],
    }
    (outdir / "manifest.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        src = Path(__file__).resolve()
        if src.exists():
            (outdir / "source_sha256.txt").write_text(sha256_file(src), encoding="utf-8")
            (outdir / "source_code.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    episode_rows, step_rows = run_tasks(tasks, args.workers, logger, keep_steps=save_stride > 0)

    regime_rows = group_summary(episode_rows, ["audit", "theta", "theta_risk_mean", "causal_horizon", "regime_id", "delay", "noise_base", "coupling_base", "condition"])
    condition_rows = group_summary(episode_rows, ["audit", "theta", "causal_horizon", "condition"])
    failure_rows = failure_summary(episode_rows)
    norm_rows = normalized_condition_metrics(condition_rows)

    write_csv(outdir / "episode_summary.csv", episode_rows)
    write_csv(outdir / "regime_summary.csv", regime_rows)
    write_csv(outdir / "condition_summary.csv", condition_rows)
    write_csv(outdir / "failure_mode_summary.csv", failure_rows)
    write_csv(outdir / "normalized_condition_metrics.csv", norm_rows)
    if save_stride > 0:
        write_csv(outdir / "step_timeseries.csv", step_rows)

    report = build_report(config, condition_rows)
    (outdir / "structural_boundary_report.txt").write_text(report, encoding="utf-8")
    logger.log(f"wrote report: {outdir / 'structural_boundary_report.txt'}")
    logger.log("completed")


if __name__ == "__main__":
    main()
