#!/usr/bin/env python3
# ============================================================================
# Aα-NSGA-II: COMPLETE PIPELINE (v8.8)
#   Adaptive α + Adaptive Mutation + Fair HV + IGD/GD + Pareto Plots
#   + F2 family + α-sensitivity proxy Σ̂
#   + External benchmarks: SPEA2, MOEA/D, NSGA-III, cMLSGA
#     * strict: fixed α = FAIR_ALPHA (HEADLINE)
#     * fair:   α-adaptation wrapper (ABLATION)
#
# Reproduces all results, tables, and figures reported in:
#   "Co-Evolving the Confidence Level in Fuzzy Multi-Objective Optimization:
#    A Computable Criterion and Its Empirical Test" (IEEE TFS, 2026).
#
# Usage:
#   python aa_nsga2_pipeline.py --serial                # full run, serial
#   python aa_nsga2_pipeline.py                         # full run, parallel
#   python aa_nsga2_pipeline.py --test --serial         # smoke test (~1 min)
#   python aa_nsga2_pipeline.py --pareto-only           # figures only
#   python aa_nsga2_pipeline.py --resume                # resume from checkpoint
# ============================================================================

import numpy as np
import pandas as pd
import time
import os
import sys
import json
import pickle
import gc
import multiprocessing as mp
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed,
    TimeoutError as FutTimeout,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Axes3D.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*identical low.*')

EPS = 1e-12

# ---------------------------------------------------------------------------
# Portable output root: writes under <repo>/results by default. Override by
# setting the environment variable AA_NSGA2_OUTPUT or editing this block.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.environ.get(
    'AA_NSGA2_OUTPUT',
    os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'results')),
)

FAIR_ALPHA = 0.75
SIGMA_HAT_ALPHAS = (0.50, 0.75, 1.00)
SIGMA_HAT_RUNS_PER_ALPHA = 5
SIGMA_HAT_REF_QUANTILE = 0.95

SPEA2_ARCHIVE_SIZE = None
SPEA2_KNN_K = 1
MOEAD_NEIGHBORHOOD_T = 20
MOEAD_DIVISIONS = {2: 12, 3: 6}
NSGA3_DIVISIONS = {2: 12, 3: 6}
CMLSGA_N_SUBPOPS = 8
CMLSGA_MIGRATION_FREQ = 10

EXTERNAL_COMPARISON_MODE = 'both'

REF_FRONT_MAX_INPUT = 600
REF_FRONT_MAX_OUTPUT = 100
IGD_PER_ALGO_CAP = 150
IGD_SKIP = {'FScal5000', 'FScal10000'}


def _detect_jupyter() -> bool:
    try:
        return ('ipykernel' in sys.modules) or ('google.colab' in sys.modules)
    except Exception:
        return False


IN_JUPYTER = _detect_jupyter()
IS_TTY = (not IN_JUPYTER) and sys.stdout.isatty()
FORCE_PARALLEL_FROM_JUPYTER = False


# ============================================================================
# PART 0: PROGRESS MONITOR
# ============================================================================
class ProgressMonitor:
    HEARTBEAT_LOG_SEC = 30
    HEARTBEAT_DRAW_SEC = 2
    JUPYTER_DRAW_SEC = 30

    def __init__(self, total_tasks, output_dir,
                 log_name='progress.log', json_name='progress.json',
                 in_jupyter=False, no_console=False):
        self.total = total_tasks
        self.completed = 0
        self.failed = 0
        self.started_at = time.time()
        self.last_completion_at = self.started_at
        self.current_in_flight = {}
        self.per_problem_times = {}
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, log_name)
        self.json_path = os.path.join(output_dir, json_name)
        self.in_jupyter = in_jupyter
        self.no_console = no_console
        self._last_draw = 0.0
        self._last_log_hb = 0.0
        self._last_state_completed = -1
        os.makedirs(output_dir, exist_ok=True)
        self._log("=" * 78)
        self._log(f"[{self._now()}] MONITOR STARTED — {total_tasks} tasks")
        self._log("=" * 78)
        self._dump_json()
        self._draw(force=True)

    def start_task(self, name):
        self.current_in_flight[name] = time.time()

    def task_done(self, name, success=True):
        t0 = self.current_in_flight.pop(name, time.time())
        elapsed = time.time() - t0
        self.per_problem_times[name] = elapsed
        self.completed += 1
        if not success:
            self.failed += 1
        self.last_completion_at = time.time()
        marker = "✓" if success else "✗"
        self._log(f"[{self._now()}] {marker} {name}  ({elapsed:.1f}s)  "
                  f"[{self.completed}/{self.total}]")
        self._draw(force=True)
        self._dump_json()

    def tick(self):
        now = time.time()
        if now - self._last_log_hb >= self.HEARTBEAT_LOG_SEC:
            self._last_log_hb = now
            since = now - self.last_completion_at
            self._log(f"[{self._now()}] ♥ heartbeat  "
                      f"[{self.completed}/{self.total}]  "
                      f"active={len(self.current_in_flight)}  "
                      f"last_done={int(since)}s ago")
        self._draw()
        self._dump_json()

    def finish(self):
        total = time.time() - self.started_at
        self._log("-" * 78)
        self._log(f"[{self._now()}] ALL DONE — {self.completed}/{self.total} "
                  f"({self.failed} failed) in {self._fmt_dur(total)}")
        self._log("-" * 78)
        self._dump_json(final=True)
        if IS_TTY:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def _draw(self, force=False):
        if self.no_console:
            return
        now = time.time()
        threshold = self.JUPYTER_DRAW_SEC if self.in_jupyter else self.HEARTBEAT_DRAW_SEC
        state_changed = (self.completed != self._last_state_completed)
        if not force and not state_changed and (now - self._last_draw) < threshold:
            return
        self._last_draw = now
        self._last_state_completed = self.completed
        pct = self.completed / max(self.total, 1)
        bar_w = 30
        filled = int(round(bar_w * pct))
        bar = '█' * filled + '░' * (bar_w - filled)
        elapsed = time.time() - self.started_at
        eta_str = "~?"
        if self.completed > 0:
            rate = self.completed / max(elapsed, 1e-6)
            remaining = (self.total - self.completed) / max(rate, 1e-9)
            eta_str = self._fmt_dur(remaining)
        line = (f"  [{bar}] {self.completed:>3}/{self.total} "
                f"({pct*100:5.1f}%)  ETA {eta_str:>7}  "
                f"active={len(self.current_in_flight)}")
        if IS_TTY and not self.in_jupyter:
            sys.stdout.write('\r' + line + ' ' * 12)
            sys.stdout.flush()
        else:
            print(line, flush=True)

    def _dump_json(self, final=False):
        elapsed = time.time() - self.started_at
        data = {
            'total': self.total, 'completed': self.completed,
            'failed': self.failed,
            'percent': round(100.0 * self.completed / max(self.total, 1), 2),
            'elapsed_human': self._fmt_dur(elapsed),
            'active_count': len(self.current_in_flight),
            'in_flight': list(self.current_in_flight.keys()),
            'per_problem_seconds': {k: round(v, 2)
                                    for k, v in self.per_problem_times.items()},
            'final': final, 'updated_at': self._now(),
        }
        try:
            tmp = self.json_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.json_path)
        except Exception:
            pass

    def _log(self, msg):
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except Exception:
            pass

    @staticmethod
    def _now():
        return datetime.now().strftime('%H:%M:%S')

    @staticmethod
    def _fmt_dur(seconds):
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        m, s = divmod(seconds, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"


# ============================================================================
# PART 1: FUZZY MATHEMATICS
# ============================================================================

def _sort3(a, b, c):
    s = sorted([a, b, c])
    return float(s[0]), float(s[1]), float(s[2])


def _sort4(a, b, c, d):
    s = sorted([a, b, c, d])
    return float(s[0]), float(s[1]), float(s[2]), float(s[3])


class TriangularFuzzyNumber:
    __slots__ = ('aL', 'aM', 'aU')

    def __init__(self, aL, aM, aU):
        aL, aM, aU = _sort3(aL, aM, aU)
        self.aL, self.aM, self.aU = aL, aM, aU

    def __repr__(self):
        return f"TFN({self.aL:.4f}, {self.aM:.4f}, {self.aU:.4f})"

    def expected_value(self):
        return (self.aL + 4 * self.aM + self.aU) / 6.0


class TrapezoidalFuzzyNumber:
    __slots__ = ('a1', 'a2', 'a3', 'a4')

    def __init__(self, a1, a2, a3, a4):
        a1, a2, a3, a4 = _sort4(a1, a2, a3, a4)
        self.a1, self.a2, self.a3, self.a4 = a1, a2, a3, a4

    def __repr__(self):
        return f"TrFN({self.a1:.4f}, {self.a2:.4f}, {self.a3:.4f}, {self.a4:.4f})"

    def expected_value(self):
        return (self.a1 + 2 * self.a2 + 2 * self.a3 + self.a4) / 6.0

    def to_triangular(self):
        return TriangularFuzzyNumber(self.a1, 0.5 * (self.a2 + self.a3), self.a4)


def credibility_leq_triangular(A, B):
    denom = (A.aU - A.aL) + (B.aU - B.aL)
    if denom <= EPS:
        return 1.0 if A.aU <= B.aL else 0.0
    return float(np.clip((B.aU - A.aL) / denom, 0.0, 1.0))


def crisp_to_trapezoidal(x, delta=0.30, eta=1.0):
    spread = abs(x) * delta * eta if x != 0 else delta * eta
    return TrapezoidalFuzzyNumber(
        x - 1.0 * spread, x - 0.5 * spread,
        x + 0.5 * spread, x + 1.5 * spread,
    )


# ============================================================================
# PART 2: FUZZY OPERATORS
# ============================================================================

def _tfn_dominates_arrays(A, B):
    AL, AM, AU = A[:, 0], A[:, 1], A[:, 2]
    BL, BM, BU = B[:, 0], B[:, 1], B[:, 2]
    denom = (AU - AL) + (BU - BL)
    num = BU - AL
    credA = np.where(denom > EPS,
                     np.clip(num / np.where(denom > EPS, denom, 1.0), 0.0, 1.0),
                     (AU <= BL).astype(float))
    strict = AL > BU
    return bool(np.all(credA >= 0.5)) and bool(np.any(strict))


def fast_nondominated_sort_fuzzy(F, chunk_size=256):
    N = F.shape[0]
    if N == 0:
        return []
    L = F[:, :, 0].astype(float)
    U = F[:, :, 2].astype(float)
    W = U - L
    D = np.zeros((N, N), dtype=bool)
    chunk = max(1, int(chunk_size))
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Lc = L[start:end]
        Uc = U[start:end]
        Wc = W[start:end]
        num = U[None, :, :] - Lc[:, None, :]
        den = Wc[:, None, :] + W[None, :, :]
        safe_den = np.where(den > EPS, den, 1.0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            cred = np.clip(num / safe_den, 0.0, 1.0)
        degenerate = den <= EPS
        if degenerate.any():
            deg_vals = (U[None, :, :] <= Lc[:, None, :]).astype(float)
            cred[degenerate] = deg_vals[degenerate]
            del deg_vals
        weak_all = np.all(cred >= 0.5, axis=2)
        strict_any = np.any(Lc[:, None, :] > U[None, :, :], axis=2)
        D[start:end] = weak_all & strict_any
        del num, den, safe_den, cred, degenerate, weak_all, strict_any
    np.fill_diagonal(D, False)
    n = (D & ~D.T).sum(axis=0).astype(int)
    S = [np.flatnonzero(D[p]) for p in range(N)]
    fronts = [list(np.flatnonzero(n == 0))]
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(int(q))
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def fuzzy_crowding_distance(F, front):
    n = len(front)
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist
    idx_front = np.asarray(front, dtype=int)
    E_all = (F[idx_front, :, 0] + 4 * F[idx_front, :, 1] +
             F[idx_front, :, 2]) / 6.0
    for m in range(F.shape[1]):
        E = E_all[:, m]
        order = np.argsort(E)
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        rng = E[order[-1]] - E[order[0]]
        if rng <= EPS:
            continue
        mid = order[1:-1]
        dist[mid] += (E[order[2:]] - E[order[:-2]]) / rng
    return dist


def das_dennis_points(M, H):
    from itertools import combinations_with_replacement
    points = []
    for c in combinations_with_replacement(range(H + 1), M - 1):
        w = [c[0]]
        for i in range(1, M - 1):
            w.append(c[i] - c[i - 1])
        w.append(H - c[-1])
        points.append([x / H for x in w])
    return np.array(points)


# ============================================================================
# PART 3: FUZZY BENCHMARKS
# ============================================================================
class FuzzyBenchmark:
    name = "base"
    n_var = 30
    n_obj = 2

    def evaluate_crisp(self, x):
        raise NotImplementedError

    def evaluate_fuzzy(self, x, delta=0.30):
        crisp = self.evaluate_crisp(x)
        fuzzy = np.empty((len(crisp), 4))
        for i, val in enumerate(crisp):
            trfn = crisp_to_trapezoidal(val, delta=delta)
            fuzzy[i] = (trfn.a1, trfn.a2, trfn.a3, trfn.a4)
        return fuzzy


class FZDT1(FuzzyBenchmark):
    name, n_var, n_obj = "FZDT1", 30, 2
    def evaluate_crisp(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        return np.array([f1, g * (1 - np.sqrt(f1 / g))])


class FZDT2(FuzzyBenchmark):
    name, n_var, n_obj = "FZDT2", 30, 2
    def evaluate_crisp(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        return np.array([f1, g * (1 - (f1 / g) ** 2)])


class FZDT3(FuzzyBenchmark):
    name, n_var, n_obj = "FZDT3", 30, 2
    def evaluate_crisp(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        return np.array([f1, g * (1 - np.sqrt(f1 / g) -
                                   (f1 / g) * np.sin(10 * np.pi * f1))])


class FZDT4(FuzzyBenchmark):
    name, n_var, n_obj = "FZDT4", 10, 2
    def evaluate_crisp(self, x):
        f1 = x[0]
        g = 1.0 + 10.0 * (len(x) - 1) + np.sum(
            x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]))
        return np.array([f1, g * (1 - np.sqrt(f1 / g))])


class FZDT6(FuzzyBenchmark):
    name, n_var, n_obj = "FZDT6", 10, 2
    def evaluate_crisp(self, x):
        f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0])) ** 6
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (len(x) - 1)) ** 0.25
        return np.array([f1, g * (1 - (f1 / g) ** 2)])


class FDTLZ1(FuzzyBenchmark):
    name, n_var, n_obj = "FDTLZ1", 7, 3
    def evaluate_crisp(self, x):
        k = self.n_var - self.n_obj + 1
        g = 100 * (k + np.sum((x[self.n_obj - 1:] - 0.5) ** 2
                              - np.cos(20 * np.pi * (x[self.n_obj - 1:] - 0.5))))
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            f[i] = 0.5 * (1 + g)
            for j in range(self.n_obj - i - 1):
                f[i] *= x[j]
            if i > 0:
                f[i] *= (1 - x[self.n_obj - i - 1])
        return f


class FDTLZ2(FuzzyBenchmark):
    name, n_var, n_obj = "FDTLZ2", 12, 3
    def evaluate_crisp(self, x):
        g = np.sum((x[self.n_obj - 1:] - 0.5) ** 2)
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            f[i] = (1 + g)
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(0.5 * np.pi * x[j])
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * x[self.n_obj - i - 1])
        return f


class FDTLZ3(FuzzyBenchmark):
    name, n_var, n_obj = "FDTLZ3", 12, 3
    def evaluate_crisp(self, x):
        k = self.n_var - self.n_obj + 1
        g = 100 * (k + np.sum((x[self.n_obj - 1:] - 0.5) ** 2
                              - np.cos(20 * np.pi * (x[self.n_obj - 1:] - 0.5))))
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            f[i] = (1 + g)
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(0.5 * np.pi * x[j])
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * x[self.n_obj - i - 1])
        return f


class FDTLZ4(FuzzyBenchmark):
    name, n_var, n_obj = "FDTLZ4", 12, 3
    def evaluate_crisp(self, x):
        g = np.sum((x[self.n_obj - 1:] - 0.5) ** 2)
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            f[i] = (1 + g)
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(0.5 * np.pi * (x[j] ** 100))
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * (x[self.n_obj - i - 1] ** 100))
        return f


class FDTLZ7(FuzzyBenchmark):
    name, n_var, n_obj = "FDTLZ7", 22, 3
    def evaluate_crisp(self, x):
        k = self.n_var - self.n_obj + 1
        g = 1 + 9.0 / k * np.sum(x[self.n_obj - 1:])
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = x[i]
        h = self.n_obj - np.sum(
            f[:self.n_obj - 1] / (1 + g) *
            (1 + np.sin(3 * np.pi * f[:self.n_obj - 1])))
        f[self.n_obj - 1] = (1 + g) * h
        return f


def _wfg_bias_poly(y, a):
    return y ** a


def _wfg_bias_flat(y, A, B, C):
    return A + np.minimum(0, np.floor(y - B)) * A * (B - y) / B \
           - np.minimum(0, np.floor(C - y)) * (1 - A) * (C - y) / (1 - C)


def _wfg_bias_param(y, u, A, B, C):
    return y ** (B + (C - B) * (u ** (A - 2 * (1 - u) * (A - 2))))


def _wfg_shift_linear(y, A):
    return np.abs(y - A) / np.abs(np.floor(A - y) + A)


def _wfg_shift_deceptive(y, A, B, C):
    r = np.abs(y - A) / np.abs(np.floor(A - y) + A)
    base = np.clip(1 - B - y, 0.0, None)
    return r * (0.1 * (1 - y) * base ** C + 0.1 * y * (A - y))


def _wfg_shift_multi(y, A, B, C):
    r = np.abs(y - A) / np.abs(np.floor(A - y) + A)
    return (1 + np.cos((4 * A + 2) * np.pi * (0.5 - r))) / (2 * (1 + B))


def _wfg_reduce_sum(y, w):
    return np.sum(w * y) / np.sum(w)


def _wfg_reduce_nonsep(y, A):
    n = len(y)
    if A < 2:
        A = 2
    result = 0.0
    for j in range(n):
        for k in range(A - 1):
            result += np.abs(y[j] - y[(j + k + 1) % n])
    return (result + np.sum(y)) / (n * A)


def _shape_linear(x, m):
    return x / m


def _shape_convex(x, m):
    return (1 - np.cos(x * np.pi / 2)) / m


def _shape_concave(x, m):
    return np.sin(x * np.pi / 2) / m


def _shape_mixed(x, m):
    return (1 - x - np.cos(2 * np.pi * x + np.pi / 2) / (2 * np.pi)) / m


def _shape_disc(x, m):
    return (1 - x - np.cos(np.pi * x)) / m


class WFGBase(FuzzyBenchmark):
    def __init__(self, n_obj=3, n_var=12):
        self.n_obj = n_obj
        self.k = n_obj - 1
        self.l = n_var - self.k
        self.n_var = n_var

    def evaluate_crisp(self, z):
        z = np.asarray(z, dtype=float) * np.arange(1, self.n_var + 1) * 2.0 / self.n_var
        return self._shape(self._transform(z))


class FWFG1(WFGBase):
    name, n_var, n_obj = "FWFG1", 12, 3
    def _transform(self, z):
        n, k = len(z), self.k
        y = z.copy()
        y[:k] = _wfg_bias_poly(y[:k], 0.02)
        y[k:] = _wfg_bias_flat(y[k:], 0.8, 0.75, 0.85)
        y = _wfg_shift_linear(y, 0.35)
        w = np.arange(2, 2 * n + 1, 2)
        t = np.zeros(self.n_obj)
        k_seg = k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_sum(y[i * k_seg:(i + 1) * k_seg], w[:k_seg])
        t[-1] = _wfg_reduce_sum(y[k:], w[k:])
        return t

    def _shape(self, x):
        m = self.n_obj
        f = np.zeros(m)
        for i in range(m - 1):
            f[i] = _shape_convex(x[i], m) if i == 0 else _shape_mixed(x[i], m)
        f[-1] = _shape_mixed(x[-1], m)
        return f


class FWFG2(WFGBase):
    name, n_var, n_obj = "FWFG2", 12, 3
    def _transform(self, z):
        n, k = len(z), self.k
        y = _wfg_shift_linear(z, 0.35)
        t = np.zeros(self.n_obj)
        k_seg = k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_nonsep(y[i * k_seg:(i + 1) * k_seg], 2)
        t[-1] = _wfg_reduce_nonsep(y[k:], len(y[k:]))
        return t

    def _shape(self, x):
        m = self.n_obj
        f = np.zeros(m)
        for i in range(m - 1):
            f[i] = _shape_convex(x[i], m)
        f[-1] = _shape_disc(x[-1], m)
        return f


class FWFG3(WFGBase):
    name, n_var, n_obj = "FWFG3", 12, 3
    def _transform(self, z):
        y = _wfg_shift_linear(z, 0.35)
        t = np.zeros(self.n_obj)
        k_seg = self.k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_nonsep(y[i * k_seg:(i + 1) * k_seg], 2)
        t[-1] = _wfg_reduce_nonsep(y[self.k:], len(y[self.k:]))
        return t

    def _shape(self, x):
        return np.array([_shape_linear(x[i], self.n_obj)
                         for i in range(self.n_obj)])


class FWFG4(WFGBase):
    name, n_var, n_obj = "FWFG4", 12, 3
    def _transform(self, z):
        n, k = len(z), self.k
        y = _wfg_shift_multi(z, 30, 10, 0.35)
        w = np.ones(n)
        t = np.zeros(self.n_obj)
        k_seg = k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_sum(y[i * k_seg:(i + 1) * k_seg], w[:k_seg])
        t[-1] = _wfg_reduce_sum(y[k:], w[k:])
        return t

    def _shape(self, x):
        return np.array([_shape_concave(x[i], self.n_obj)
                         for i in range(self.n_obj)])


class FWFG5(WFGBase):
    name, n_var, n_obj = "FWFG5", 12, 3
    def _transform(self, z):
        n, k = len(z), self.k
        y = _wfg_shift_deceptive(z, 0.35, 0.001, 0.05)
        w = np.ones(n)
        t = np.zeros(self.n_obj)
        k_seg = k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_sum(y[i * k_seg:(i + 1) * k_seg], w[:k_seg])
        t[-1] = _wfg_reduce_sum(y[k:], w[k:])
        return t

    def _shape(self, x):
        return np.array([_shape_concave(x[i], self.n_obj)
                         for i in range(self.n_obj)])


class FWFG6(WFGBase):
    name, n_var, n_obj = "FWFG6", 12, 3
    def _transform(self, z):
        y = _wfg_shift_linear(z, 0.35)
        t = np.zeros(self.n_obj)
        k_seg = self.k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_nonsep(y[i * k_seg:(i + 1) * k_seg], k_seg)
        t[-1] = _wfg_reduce_nonsep(y[self.k:], len(y) - self.k)
        return t

    def _shape(self, x):
        return np.array([_shape_concave(x[i], self.n_obj)
                         for i in range(self.n_obj)])


class FWFG7(WFGBase):
    name, n_var, n_obj = "FWFG7", 12, 3
    def _transform(self, z):
        n, k = len(z), self.k
        w = np.ones(n)
        y = z.copy()
        for i in range(k):
            u = _wfg_reduce_sum(z[k:], w[k:])
            y[i] = _wfg_bias_param(z[i], u, 0.98 / 49.98, 0.02, 50)
        y = _wfg_shift_linear(y, 0.35)
        t = np.zeros(self.n_obj)
        k_seg = k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_sum(y[i * k_seg:(i + 1) * k_seg], w[:k_seg])
        t[-1] = _wfg_reduce_sum(y[k:], w[k:])
        return t

    def _shape(self, x):
        return np.array([_shape_concave(x[i], self.n_obj)
                         for i in range(self.n_obj)])


class FWFG8(WFGBase):
    name, n_var, n_obj = "FWFG8", 12, 3
    def _transform(self, z):
        n, k = len(z), self.k
        w = np.ones(n)
        y = z.copy()
        for i in range(k, n):
            u = _wfg_reduce_sum(z[:k], w[:k])
            y[i] = _wfg_bias_param(z[i], u, 0.98 / 49.98, 0.02, 50)
        y = _wfg_shift_linear(y, 0.35)
        t = np.zeros(self.n_obj)
        k_seg = k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_nonsep(y[i * k_seg:(i + 1) * k_seg], k_seg)
        t[-1] = _wfg_reduce_nonsep(y[k:], n - k)
        return t

    def _shape(self, x):
        return np.array([_shape_concave(x[i], self.n_obj)
                         for i in range(self.n_obj)])


class FWFG9(WFGBase):
    name, n_var, n_obj = "FWFG9", 12, 3
    def _transform(self, z):
        n, k = len(z), self.k
        w = np.ones(n)
        y = z.copy()
        for i in range(n - 1):
            u = _wfg_reduce_sum(z[i + 1:], w[i + 1:])
            y[i] = _wfg_bias_param(z[i], u, 0.98 / 49.98, 0.02, 50)
        y = _wfg_shift_deceptive(y, 0.35, 0.001, 0.05)
        t = np.zeros(self.n_obj)
        k_seg = k // (self.n_obj - 1)
        for i in range(self.n_obj - 1):
            t[i] = _wfg_reduce_nonsep(y[i * k_seg:(i + 1) * k_seg], k_seg)
        t[-1] = _wfg_reduce_nonsep(y[k:], n - k)
        return t

    def _shape(self, x):
        return np.array([_shape_concave(x[i], self.n_obj)
                         for i in range(self.n_obj)])


class FUF1(FuzzyBenchmark):
    name, n_var, n_obj = "FUF1", 30, 2
    def evaluate_crisp(self, x):
        J1 = np.arange(2, len(x), 2)
        J2 = np.arange(1, len(x), 2)
        f1 = x[0] + 2.0 / len(J1) * np.sum(
            (x[J1] - np.sin(6 * np.pi * x[0] + (J1 + 1) * np.pi / len(x))) ** 2)
        f2 = 1 - np.sqrt(x[0]) + 2.0 / len(J2) * np.sum(
            (x[J2] - np.sin(6 * np.pi * x[0] + (J2 + 1) * np.pi / len(x))) ** 2)
        return np.array([f1, f2])


class FUF2(FuzzyBenchmark):
    name, n_var, n_obj = "FUF2", 30, 2
    def evaluate_crisp(self, x):
        J1 = np.arange(2, len(x), 2)
        J2 = np.arange(1, len(x), 2)
        f1 = x[0] + 2.0 / len(J1) * np.sum(np.abs(x[J1] - 0.5 * (x[0] + 1)) ** 2)
        f2 = 1 - np.sqrt(x[0]) + 2.0 / len(J2) * np.sum(
            np.abs(x[J2] - 0.5 * (1 - np.sqrt(x[0]))) ** 2)
        return np.array([f1, f2])


class FUF3(FuzzyBenchmark):
    name, n_var, n_obj = "FUF3", 30, 2
    def evaluate_crisp(self, x):
        J1 = np.arange(2, len(x), 2)
        J2 = np.arange(1, len(x), 2)
        Y1 = x[J1] - x[0] ** (0.5 * (1 + 3 * (J1 - 2) / (len(x) - 2)))
        Y2 = x[J2] - x[0] ** (0.5 * (1 + 3 * (J2 - 2) / (len(x) - 2)))
        return np.array([x[0] + 2.0 / len(J1) * np.sum(Y1 ** 2),
                         1 - np.sqrt(x[0]) + 2.0 / len(J2) * np.sum(Y2 ** 2)])


class FUF4(FuzzyBenchmark):
    name, n_var, n_obj = "FUF4", 30, 2
    def evaluate_crisp(self, x):
        J1 = np.arange(2, len(x), 2)
        J2 = np.arange(1, len(x), 2)
        Y1 = x[J1] - np.sin(6 * np.pi * x[0] + (J1 + 1) * np.pi / len(x))
        Y2 = x[J2] - np.cos(6 * np.pi * x[0] + (J2 + 1) * np.pi / len(x))
        return np.array([x[0] + 2.0 / len(J1) * np.sum(Y1 ** 2),
                         1 - x[0] ** 2 + 2.0 / len(J2) * np.sum(Y2 ** 2)])


class FUF5(FuzzyBenchmark):
    name, n_var, n_obj = "FUF5", 30, 2
    def evaluate_crisp(self, x):
        J1 = np.arange(2, len(x), 2)
        J2 = np.arange(1, len(x), 2)
        Y1 = x[J1] - np.sin(6 * np.pi * x[0] + (J1 + 1) * np.pi / len(x))
        Y2 = x[J2] - np.cos(6 * np.pi * x[0] + (J2 + 1) * np.pi / len(x))
        return np.array([x[0] + (2.0 / len(J1)) * np.sum(np.abs(Y1)),
                         1 - x[0] + (2.0 / len(J2)) * np.sum(np.abs(Y2))])


class FUF6(FuzzyBenchmark):
    name, n_var, n_obj = "FUF6", 30, 2
    def evaluate_crisp(self, x):
        J1 = np.arange(2, len(x), 2)
        J2 = np.arange(1, len(x), 2)
        Y1 = x[J1] - np.sin(6 * np.pi * x[0] + (J1 + 1) * np.pi / len(x))
        Y2 = x[J2] - np.sin(6 * np.pi * x[0] + (J2 + 1) * np.pi / len(x))
        return np.array([x[0] + (2.0 / len(J1)) * np.sum(Y1 ** 2),
                         1 - x[0] + (2.0 / len(J2)) * np.sum(Y2 ** 2)])


class FUF7(FuzzyBenchmark):
    name, n_var, n_obj = "FUF7", 30, 2
    def evaluate_crisp(self, x):
        J1 = np.arange(2, len(x), 2)
        J2 = np.arange(1, len(x), 2)
        y1 = x[0] ** 0.2
        y2 = 1 - x[0] ** 0.2
        Y1 = x[J1] - np.sin(6 * np.pi * y1 + (J1 + 1) * np.pi / len(x))
        Y2 = x[J2] - np.sin(6 * np.pi * y2 + (J2 + 1) * np.pi / len(x))
        return np.array([y1 + (2.0 / len(J1)) * np.sum(Y1 ** 2),
                         y2 + (2.0 / len(J2)) * np.sum(Y2 ** 2)])


class FUF8(FuzzyBenchmark):
    name, n_var, n_obj = "FUF8", 30, 3
    def evaluate_crisp(self, x):
        J1 = np.arange(3, len(x), 3)
        J2 = np.arange(1, len(x), 3)
        J3 = np.arange(2, len(x), 3)
        Y1 = x[J1] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J1 + 1) * np.pi / len(x))
        Y2 = x[J2] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J2 + 1) * np.pi / len(x))
        Y3 = x[J3] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J3 + 1) * np.pi / len(x))
        return np.array([
            np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) +
            (2.0 / len(J1)) * np.sum(Y1 ** 2),
            np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) +
            (2.0 / len(J2)) * np.sum(Y2 ** 2),
            np.sin(0.5 * np.pi * x[0]) +
            (2.0 / len(J3)) * np.sum(Y3 ** 2)])


class FUF9(FuzzyBenchmark):
    name, n_var, n_obj = "FUF9", 30, 3
    def evaluate_crisp(self, x):
        J1 = np.arange(3, len(x), 3)
        J2 = np.arange(1, len(x), 3)
        J3 = np.arange(2, len(x), 3)
        Y1 = x[J1] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J1 + 1) * np.pi / len(x))
        Y2 = x[J2] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J2 + 1) * np.pi / len(x))
        Y3 = x[J3] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J3 + 1) * np.pi / len(x))
        return np.array([
            0.5 * (np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + 0.5) +
            (2.0 / len(J1)) * np.sum(Y1 ** 2),
            0.5 * (np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + 0.5) +
            (2.0 / len(J2)) * np.sum(Y2 ** 2),
            0.5 * (np.sin(0.5 * np.pi * x[0]) + 0.5) +
            (2.0 / len(J3)) * np.sum(Y3 ** 2)])


class FUF10(FuzzyBenchmark):
    name, n_var, n_obj = "FUF10", 30, 3
    def evaluate_crisp(self, x):
        J1 = np.arange(3, len(x), 3)
        J2 = np.arange(1, len(x), 3)
        J3 = np.arange(2, len(x), 3)
        Y1 = x[J1] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J1 + 1) * np.pi / len(x))
        Y2 = x[J2] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J2 + 1) * np.pi / len(x))
        Y3 = x[J3] - 2 * x[1] * np.sin(2 * np.pi * x[0] + (J3 + 1) * np.pi / len(x))
        return np.array([
            np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) +
            (2.0 / len(J1)) * np.sum(4 * Y1 ** 2 - np.cos(8 * np.pi * Y1) + 1),
            np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) +
            (2.0 / len(J2)) * np.sum(4 * Y2 ** 2 - np.cos(8 * np.pi * Y2) + 1),
            np.sin(0.5 * np.pi * x[0]) +
            (2.0 / len(J3)) * np.sum(4 * Y3 ** 2 - np.cos(8 * np.pi * Y3) + 1)])


class FUF11(FuzzyBenchmark):
    name, n_var, n_obj = "FUF11", 30, 2
    def evaluate_crisp(self, x):
        g = np.sum((x[self.n_obj - 1:] - 0.5) ** 2)
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            f[i] = (1 + g)
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(0.5 * np.pi * x[j])
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * x[self.n_obj - i - 1])
        return f


class FUF12(FuzzyBenchmark):
    name, n_var, n_obj = "FUF12", 30, 2
    def evaluate_crisp(self, x):
        g = np.sum((x[self.n_obj - 1:] - 0.5) ** 2)
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            f[i] = (1 + g)
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(0.5 * np.pi * x[j] ** 4)
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * x[self.n_obj - i - 1] ** 4)
        return f


def _maf_dtlz_base(x, n_obj):
    m = n_obj
    g = np.sum((x[m - 1:] - 0.5) ** 2)
    f = np.zeros(m)
    for i in range(m):
        f[i] = (1 + g)
        for j in range(m - i - 1):
            f[i] *= np.cos(0.5 * np.pi * x[j])
        if i > 0:
            f[i] *= np.sin(0.5 * np.pi * x[m - i - 1])
    return f


class FMaF1(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF1", 12, 3
    def evaluate_crisp(self, x):
        m = self.n_obj
        g = np.sum((x[m - 1:] - 0.5) ** 2)
        f = np.zeros(m)
        for i in range(m):
            f[i] = (1 + g)
            for j in range(m - i - 1):
                f[i] *= x[j]
            if i > 0:
                f[i] *= (1 - x[m - i - 1])
        return 1 - f


class FMaF2(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF2", 12, 3
    def evaluate_crisp(self, x):
        return _maf_dtlz_base(x, self.n_obj)


class FMaF3(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF3", 12, 3
    def evaluate_crisp(self, x):
        return _maf_dtlz_base(x, self.n_obj) ** 4


class FMaF4(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF4", 12, 3
    def evaluate_crisp(self, x):
        return 1 - _maf_dtlz_base(x, self.n_obj)


class FMaF5(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF5", 12, 3
    def evaluate_crisp(self, x):
        return _maf_dtlz_base(x, self.n_obj) ** 2


class FMaF6(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF6", 12, 3
    def evaluate_crisp(self, x):
        m = self.n_obj
        g = np.sum((x[m - 1:] - 0.5) ** 2)
        f = np.zeros(m)
        for i in range(m - 1):
            f[i] = x[i]
        f[m - 1] = (1 + g) * (m - np.sum(
            f[:m - 1] / (1 + g) * (1 + np.sin(3 * np.pi * f[:m - 1]))))
        return f


class FMaF7(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF7", 22, 3
    def evaluate_crisp(self, x):
        n, m = len(x), self.n_obj
        g = 1 + 9.0 / (n - m + 1) * np.sum(x[m - 1:])
        f = np.zeros(m)
        for i in range(m - 1):
            f[i] = x[i]
        f[m - 1] = (1 + g) * (m - np.sum(
            f[:m - 1] / (1 + g) * (1 + np.sin(3 * np.pi * f[:m - 1]))))
        return f


class FMaF8(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF8", 2, 3
    def evaluate_crisp(self, x):
        x1, x2 = x[0], x[1]
        return np.array([(x1 - 2) ** 2 + (x2 - 1) ** 2,
                         x1 ** 2 + (x2 + 1) ** 2,
                         (x1 - 1) ** 2 + x2 ** 2])


class FMaF9(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF9", 12, 3
    def evaluate_crisp(self, x):
        return _maf_dtlz_base(x, self.n_obj) ** 4


class FMaF10(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF10", 12, 3
    def evaluate_crisp(self, x):
        return FWFG1().evaluate_crisp(x)


class FMaF11(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF11", 12, 3
    def evaluate_crisp(self, x):
        return FWFG2().evaluate_crisp(x)


class FMaF12(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF12", 12, 3
    def evaluate_crisp(self, x):
        return FWFG3().evaluate_crisp(x)


class FMaF13(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF13", 5, 3
    def evaluate_crisp(self, x):
        return _maf_dtlz_base(x, self.n_obj) ** 2


class FMaF14(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF14", 12, 3
    def evaluate_crisp(self, x):
        return FDTLZ1().evaluate_crisp(x)


class FMaF15(FuzzyBenchmark):
    name, n_var, n_obj = "FMaF15", 12, 3
    def evaluate_crisp(self, x):
        return 1 - FDTLZ1().evaluate_crisp(x)


class FScal(FuzzyBenchmark):
    name = "FScal"
    def __init__(self, n_var=500, n_obj=3):
        self.n_var = n_var
        self.n_obj = n_obj
    def evaluate_crisp(self, x):
        g = np.sum((x[self.n_obj - 1:] - 0.5) ** 2)
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            f[i] = (1 + g)
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(0.5 * np.pi * x[j])
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * x[self.n_obj - i - 1])
        return f


class FScal5000(FuzzyBenchmark):
    name = "FScal5000"
    def __init__(self):
        self.n_var = 5000
        self.n_obj = 3
    def evaluate_crisp(self, x):
        return FScal(5000, 3).evaluate_crisp(x)


class FScal10000(FuzzyBenchmark):
    name = "FScal10000"
    def __init__(self):
        self.n_var = 10000
        self.n_obj = 3
    def evaluate_crisp(self, x):
        return FScal(10000, 3).evaluate_crisp(x)


class F2Base(FuzzyBenchmark):
    def __init__(self, base_cls=None, _base=None):
        if base_cls is None and _base is None:
            raise TypeError("F2Base requires either base_cls or _base")
        self.base = _base if _base is not None else base_cls()
        if getattr(type(self), "name", "base") == "base":
            self.name = "F2" + self.base.name
        self.n_var = self.base.n_var
        self.n_obj = self.base.n_obj

    def evaluate_crisp(self, x):
        return self.base.evaluate_crisp(x)

    def evaluate_fuzzy(self, x, delta=0.30):
        sigma = float(np.clip(x[0], 0.0, 1.0))
        eta = 0.15 + 0.30 * sigma
        crisp = self.evaluate_crisp(x)
        fuzzy = np.empty((len(crisp), 4))
        for i, val in enumerate(crisp):
            trfn = crisp_to_trapezoidal(val, delta=delta, eta=eta)
            fuzzy[i] = (trfn.a1, trfn.a2, trfn.a3, trfn.a4)
        return fuzzy


def _make_f2_subclass(base_cls):
    cls_name = f"F2{base_cls.__name__}"

    def _init(self, _b=base_cls, _name=cls_name):
        F2Base.__init__(self, _base=_b())
        self.name = _name

    return type(
        cls_name,
        (F2Base,),
        {
            "__init__": _init,
            "name": cls_name,
            "n_var": base_cls.n_var,
            "n_obj": base_cls.n_obj,
        },
    )


_F2_BASE_CLASSES = [FZDT1, FZDT2, FZDT3, FZDT4, FZDT6,
                    FDTLZ2, FDTLZ4, FDTLZ7,
                    FUF1, FUF2, FUF3, FUF4,
                    FMaF1, FMaF2, FMaF3]

F2_PROBLEMS = [_make_f2_subclass(cls) for cls in _F2_BASE_CLASSES]
F2_NAMES = [f"F2{cls.__name__}" for cls in _F2_BASE_CLASSES]


# ============================================================================
# PART 4: PROBLEM REGISTRY
# ============================================================================

ALL_PROBLEMS = [
    FZDT1, FZDT2, FZDT3, FZDT4, FZDT6,
    FDTLZ1, FDTLZ2, FDTLZ3, FDTLZ4, FDTLZ7,
    FWFG1, FWFG2, FWFG3, FWFG4, FWFG5, FWFG6, FWFG7, FWFG8, FWFG9,
    FUF1, FUF2, FUF3, FUF4, FUF5, FUF6, FUF7, FUF8, FUF9, FUF10, FUF11, FUF12,
    FMaF1, FMaF2, FMaF3, FMaF4, FMaF5, FMaF6, FMaF7, FMaF8, FMaF9,
    FMaF10, FMaF11, FMaF12, FMaF13, FMaF14, FMaF15,
    FScal, FScal5000, FScal10000,
] + F2_PROBLEMS

FAMILIES = {
    'ZDT': ['FZDT1', 'FZDT2', 'FZDT3', 'FZDT4', 'FZDT6'],
    'DTLZ': ['FDTLZ1', 'FDTLZ2', 'FDTLZ3', 'FDTLZ4', 'FDTLZ7'],
    'WFG': [f'FWFG{i}' for i in range(1, 10)],
    'CEC09': [f'FUF{i}' for i in range(1, 13)],
    'MaF': [f'FMaF{i}' for i in range(1, 16)],
    'Scalability': ['FScal', 'FScal5000', 'FScal10000'],
    'F2': F2_NAMES,
}

REGISTRY = {cls.name: cls for cls in ALL_PROBLEMS}

_missing_f2 = [n for n in F2_NAMES if n not in REGISTRY]
if _missing_f2:
    raise RuntimeError(
        f"F2 family not registered correctly. Missing from REGISTRY: "
        f"{_missing_f2}."
    )
del _missing_f2


PARETO_DEFAULT_PROBLEMS = [
    'FZDT1', 'FZDT2', 'FZDT3', 'FZDT4', 'FZDT6',
    'FDTLZ1', 'FDTLZ2', 'FDTLZ3', 'FDTLZ4', 'FDTLZ7',
    'FWFG1', 'FWFG2', 'FWFG3', 'FWFG4', 'FWFG5', 'FWFG6', 'FWFG7',
    'FWFG8', 'FWFG9',
    'FUF1', 'FUF2', 'FUF3', 'FUF4', 'FUF5', 'FUF6', 'FUF7', 'FUF8',
    'FUF9', 'FUF10', 'FUF11', 'FUF12',
    'FMaF1', 'FMaF2', 'FMaF3', 'FMaF4', 'FMaF5', 'FMaF6', 'FMaF7',
    'FMaF8', 'FMaF9', 'FMaF10', 'FMaF11', 'FMaF12', 'FMaF13', 'FMaF14',
    'FMaF15',
    'FScal', 'FScal5000', 'FScal10000',
] + F2_NAMES


def get_extended_benchmark(name):
    if name in REGISTRY:
        return REGISTRY[name]()
    if name.startswith("FScal"):
        return FScal(n_var=int(name.replace("FScal", "")))
    raise ValueError(f"Unknown benchmark: {name}")


def get_all_problems():
    problems = []
    for probs in FAMILIES.values():
        problems.extend(probs)
    return problems


# ============================================================================
# PART 5: INDIVIDUAL
# ============================================================================

@dataclass
class Individual:
    x: np.ndarray
    alpha: float
    fuzzy_obj: np.ndarray = field(default=None)
    rank: int = 0
    crowding: float = 0.0
    _weighted: np.ndarray = field(default=None, repr=False)

    def evaluate(self, benchmark, delta=0.30):
        fuzzy = benchmark.evaluate_fuzzy(self.x, delta=delta)
        M = fuzzy.shape[0]
        self.fuzzy_obj = np.empty((M, 3))
        for i in range(M):
            a1, a2, a3, a4 = fuzzy[i]
            aL = min(a1, a2, a3, a4)
            aU = max(a1, a2, a3, a4)
            aM = 0.5 * (a2 + a3)
            if aM < aL:
                aM = aL
            if aM > aU:
                aM = aU
            self.fuzzy_obj[i] = (aL, aM, aU)
        self._weighted = None

    def alpha_weighted_obj(self):
        if self._weighted is None:
            L = self.fuzzy_obj[:, 0]
            M = self.fuzzy_obj[:, 1]
            U = self.fuzzy_obj[:, 2]
            self._weighted = self.alpha * M + (1 - self.alpha) * (L + U) / 2.0
        return self._weighted


# ============================================================================
# PART 5A: Aα-NSGA-II
# ============================================================================

class AAlphaNSGA2:
    def __init__(self, benchmark, pop_size=60, n_gen=300,
                 p_crossover=0.9, eta_crossover=20.0,
                 p_mutation=None, eta_mutation=20.0,
                 p_local_search=0.3, alpha_bounds=(0.5, 1.0),
                 alpha_diversity_freq=10, delta=0.30, seed=42,
                 track_history=False):
        self.benchmark = benchmark
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_crossover = p_crossover
        self.eta_crossover = eta_crossover
        self.p_mutation = p_mutation if p_mutation is not None else 1.0 / benchmark.n_var
        self.eta_mutation = eta_mutation
        self.p_local_search = p_local_search
        self.alpha_L, self.alpha_U = alpha_bounds
        self.alpha_diversity_freq = alpha_diversity_freq
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.xl = np.zeros(benchmark.n_var)
        self.xu = np.ones(benchmark.n_var)
        self.population = []

    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            x = self.rng.uniform(self.xl, self.xu)
            alpha = self.rng.uniform(self.alpha_L, self.alpha_U)
            ind = Individual(x=x, alpha=alpha)
            ind.evaluate(self.benchmark, self.delta)
            self.population.append(ind)

    def tournament_selection(self, k=2):
        cand = self.rng.choice(self.population, size=k, replace=False)
        return min(cand, key=lambda ind: (ind.rank, -ind.crowding))

    def sbx_crossover(self, p1, p2):
        if self.rng.random() > self.p_crossover:
            return p1.x.copy(), p2.x.copy()
        u = self.rng.random(self.benchmark.n_var)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1.0 / (self.eta_crossover + 1)),
                        (1.0 / (2 * (1 - u + EPS))) ** (1.0 / (self.eta_crossover + 1)))
        c1 = 0.5 * ((1 + beta) * p1.x + (1 - beta) * p2.x)
        c2 = 0.5 * ((1 - beta) * p1.x + (1 + beta) * p2.x)
        return np.clip(c1, self.xl, self.xu), np.clip(c2, self.xl, self.xu)

    def blx_alpha_crossover(self, a1, a2):
        gamma = self.rng.uniform(-0.5, 1.5)
        return float(np.clip(a1 + gamma * (a2 - a1), self.alpha_L, self.alpha_U))

    def polynomial_mutation(self, x, p_mutation=None):
        if p_mutation is None:
            p_mutation = self.p_mutation
        y = x.copy()
        mask = self.rng.random(y.shape[0]) < p_mutation
        if not mask.any():
            return y
        u = self.rng.random(int(mask.sum()))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (self.eta_mutation + 1)) - 1.0,
                         1.0 - (2 * (1 - u + EPS)) ** (1.0 / (self.eta_mutation + 1)))
        y[mask] += delta * (self.xu[mask] - self.xl[mask])
        return np.clip(y, self.xl, self.xu)

    def gaussian_alpha_mutation(self, alpha, gen):
        sigma = 0.05 * (1 - gen / self.n_gen)
        return float(np.clip(alpha + self.rng.normal(0, sigma),
                             self.alpha_L, self.alpha_U))

    def _fuzzy_dominates(self, ind1, ind2):
        return _tfn_dominates_arrays(ind1.fuzzy_obj, ind2.fuzzy_obj)

    def local_search(self, ind, gen):
        if self.rng.random() > self.p_local_search:
            return ind
        if self.rng.random() < 0.2:
            x_new = np.clip(ind.x + self.rng.normal(0, 0.05, size=len(ind.x)),
                            self.xl, self.xu)
            cand = Individual(x=x_new, alpha=ind.alpha)
            cand.evaluate(self.benchmark, self.delta)
            if self._fuzzy_dominates(cand, ind):
                ind = cand
        if self.rng.random() < 0.2:
            for da in (-0.05, -0.02, 0.02, 0.05):
                a_new = float(np.clip(ind.alpha + da, self.alpha_L, self.alpha_U))
                cand = Individual(x=ind.x.copy(), alpha=a_new)
                cand.evaluate(self.benchmark, self.delta)
                if self._fuzzy_dominates(cand, ind):
                    ind = cand
                    break
        if self.rng.random() < 0.2:
            x_new = ind.x.copy()
            j = int(self.rng.integers(len(x_new)))
            x_new[j] = self.rng.uniform(self.xl[j], self.xu[j])
            cand = Individual(x=x_new, alpha=ind.alpha)
            cand.evaluate(self.benchmark, self.delta)
            if self._fuzzy_dominates(cand, ind):
                ind = cand
        return ind

    def alpha_diversity_maintenance(self):
        alphas = np.array([ind.alpha for ind in self.population])
        bins = np.linspace(self.alpha_L, self.alpha_U, 6)
        counts, _ = np.histogram(alphas, bins=bins)
        for i in range(len(counts)):
            if counts[i] > 0.5 * self.pop_size:
                mask = (alphas >= bins[i]) & (alphas < bins[i + 1])
                idx = np.where(mask)[0]
                n_reset = len(idx) // 2
                for j in idx[:n_reset]:
                    self.population[j].alpha = self.rng.uniform(self.alpha_L, self.alpha_U)
                    self.population[j]._weighted = None
                    self.population[j].evaluate(self.benchmark, self.delta)

    def assign_rank_and_crowding(self, population):
        F = np.array([ind.fuzzy_obj for ind in population])
        fronts = fast_nondominated_sort_fuzzy(F)
        for r, front in enumerate(fronts):
            for idx in front:
                population[idx].rank = r
            dist = fuzzy_crowding_distance(F, front)
            for k, idx in enumerate(front):
                population[idx].crowding = dist[k]

    def run(self):
        self.initialize()
        self.assign_rank_and_crowding(self.population)
        for gen in range(1, self.n_gen + 1):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1_x, c2_x = self.sbx_crossover(p1, p2)
                c1_a = self.blx_alpha_crossover(p1.alpha, p2.alpha)
                c2_a = self.blx_alpha_crossover(p1.alpha, p2.alpha)
                c1_x = self.polynomial_mutation(c1_x)
                c2_x = self.polynomial_mutation(c2_x)
                c1_a = self.gaussian_alpha_mutation(c1_a, gen)
                c2_a = self.gaussian_alpha_mutation(c2_a, gen)
                c1 = Individual(x=c1_x, alpha=c1_a)
                c2 = Individual(x=c2_x, alpha=c2_a)
                c1.evaluate(self.benchmark, self.delta)
                c2.evaluate(self.benchmark, self.delta)
                c1 = self.local_search(c1, gen)
                c2 = self.local_search(c2, gen)
                offspring.extend([c1, c2])
            offspring = offspring[:self.pop_size]
            combined = self.population + offspring
            self.assign_rank_and_crowding(combined)
            combined.sort(key=lambda ind: (ind.rank, -ind.crowding))
            self.population = combined[:self.pop_size]
            if gen % self.alpha_diversity_freq == 0:
                self.alpha_diversity_maintenance()
        return self.population

    def get_pareto_front(self):
        return [ind for ind in self.population if ind.rank == 0]

    def get_alphas(self):
        return np.array([ind.alpha for ind in self.population])


class AAlphaNSGA2AdaptiveMutation(AAlphaNSGA2):
    def initialize(self):
        super().initialize()
        self.mutation_rate = 0.05

    def run(self):
        self.initialize()
        self.assign_rank_and_crowding(self.population)
        for gen in range(1, self.n_gen + 1):
            offspring = []
            gen_success, gen_total = 0, 0
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1_x, c2_x = self.sbx_crossover(p1, p2)
                c1_a = self.blx_alpha_crossover(p1.alpha, p2.alpha)
                c2_a = self.blx_alpha_crossover(p1.alpha, p2.alpha)
                c1_x = self.polynomial_mutation(c1_x, p_mutation=self.mutation_rate)
                c2_x = self.polynomial_mutation(c2_x, p_mutation=self.mutation_rate)
                c1_a = self.gaussian_alpha_mutation(c1_a, gen)
                c2_a = self.gaussian_alpha_mutation(c2_a, gen)
                c1 = Individual(x=c1_x, alpha=c1_a)
                c2 = Individual(x=c2_x, alpha=c2_a)
                c1.evaluate(self.benchmark, self.delta)
                c2.evaluate(self.benchmark, self.delta)
                if c1.rank <= p1.rank and c1.rank <= p2.rank:
                    gen_success += 1
                if c2.rank <= p1.rank and c2.rank <= p2.rank:
                    gen_success += 1
                gen_total += 2
                c1 = self.local_search(c1, gen)
                c2 = self.local_search(c2, gen)
                offspring.extend([c1, c2])
            offspring = offspring[:self.pop_size]
            combined = self.population + offspring
            self.assign_rank_and_crowding(combined)
            combined.sort(key=lambda ind: (ind.rank, -ind.crowding))
            self.population = combined[:self.pop_size]
            if gen % self.alpha_diversity_freq == 0:
                self.alpha_diversity_maintenance()
            if gen_total > 0:
                sr = gen_success / gen_total
                self.mutation_rate *= 1.1 if sr > 0.2 else 0.9
                self.mutation_rate = float(np.clip(self.mutation_rate, 0.001, 0.5))
        return self.population


# ============================================================================
# PART 5B: BASELINES
# ============================================================================

class BaselineNSGA2:
    def __init__(self, benchmark, pop_size=60, n_gen=300,
                 p_crossover=0.9, eta_crossover=20.0,
                 p_mutation=None, eta_mutation=20.0,
                 alpha=0.75, delta=0.30, seed=42, track_history=False):
        self.benchmark = benchmark
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_crossover = p_crossover
        self.eta_crossover = eta_crossover
        self.p_mutation = p_mutation or 1.0 / benchmark.n_var
        self.eta_mutation = eta_mutation
        self.alpha = alpha
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.xl = np.zeros(benchmark.n_var)
        self.xu = np.ones(benchmark.n_var)
        self.population = []

    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            x = self.rng.uniform(self.xl, self.xu)
            ind = Individual(x=x, alpha=self.alpha)
            ind.evaluate(self.benchmark, self.delta)
            self.population.append(ind)

    def tournament_selection(self, k=2):
        cand = self.rng.choice(self.population, size=k, replace=False)
        return min(cand, key=lambda ind: (ind.rank, -ind.crowding))

    def sbx_crossover(self, p1, p2):
        if self.rng.random() > self.p_crossover:
            return p1.x.copy(), p2.x.copy()
        u = self.rng.random(self.benchmark.n_var)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1.0 / (self.eta_crossover + 1)),
                        (1.0 / (2 * (1 - u + EPS))) ** (1.0 / (self.eta_crossover + 1)))
        c1 = 0.5 * ((1 + beta) * p1.x + (1 - beta) * p2.x)
        c2 = 0.5 * ((1 - beta) * p1.x + (1 + beta) * p2.x)
        return np.clip(c1, self.xl, self.xu), np.clip(c2, self.xl, self.xu)

    def polynomial_mutation(self, x):
        y = x.copy()
        mask = self.rng.random(y.shape[0]) < self.p_mutation
        if not mask.any():
            return y
        u = self.rng.random(int(mask.sum()))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (self.eta_mutation + 1)) - 1.0,
                         1.0 - (2 * (1 - u + EPS)) ** (1.0 / (self.eta_mutation + 1)))
        y[mask] += delta * (self.xu[mask] - self.xl[mask])
        return np.clip(y, self.xl, self.xu)

    def assign_rank_and_crowding(self, population):
        F = np.array([ind.fuzzy_obj for ind in population])
        fronts = fast_nondominated_sort_fuzzy(F)
        for r, front in enumerate(fronts):
            for idx in front:
                population[idx].rank = r
            dist = fuzzy_crowding_distance(F, front)
            for k, idx in enumerate(front):
                population[idx].crowding = dist[k]

    def run(self):
        self.initialize()
        self.assign_rank_and_crowding(self.population)
        for gen in range(self.n_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1_x, c2_x = self.sbx_crossover(p1, p2)
                c1_x = self.polynomial_mutation(c1_x)
                c2_x = self.polynomial_mutation(c2_x)
                c1 = Individual(x=c1_x, alpha=self.alpha)
                c2 = Individual(x=c2_x, alpha=self.alpha)
                c1.evaluate(self.benchmark, self.delta)
                c2.evaluate(self.benchmark, self.delta)
                offspring.extend([c1, c2])
            offspring = offspring[:self.pop_size]
            combined = self.population + offspring
            self.assign_rank_and_crowding(combined)
            combined.sort(key=lambda ind: (ind.rank, -ind.crowding))
            self.population = combined[:self.pop_size]
        return self.population

    def get_pareto_front(self):
        return [ind for ind in self.population if ind.rank == 0]

    def get_alphas(self):
        return np.array([ind.alpha for ind in self.population])


class BaselineNSGA2HighMutation(BaselineNSGA2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_mutation = 0.05


class BaselineSelfAdaptiveMutation(BaselineNSGA2):
    def initialize(self):
        super().initialize()
        self.mutation_rate = 0.05

    def _adaptive_mutation(self, x):
        y = x.copy()
        mask = self.rng.random(y.shape[0]) < self.mutation_rate
        if not mask.any():
            return y
        u = self.rng.random(int(mask.sum()))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (self.eta_mutation + 1)) - 1.0,
                         1.0 - (2 * (1 - u + EPS)) ** (1.0 / (self.eta_mutation + 1)))
        y[mask] += delta * (self.xu[mask] - self.xl[mask])
        return np.clip(y, self.xl, self.xu)

    def run(self):
        self.initialize()
        self.assign_rank_and_crowding(self.population)
        for gen in range(self.n_gen):
            offspring = []
            gen_success, gen_total = 0, 0
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1_x, c2_x = self.sbx_crossover(p1, p2)
                c1_x = self._adaptive_mutation(c1_x)
                c2_x = self._adaptive_mutation(c2_x)
                c1 = Individual(x=c1_x, alpha=self.alpha)
                c2 = Individual(x=c2_x, alpha=self.alpha)
                c1.evaluate(self.benchmark, self.delta)
                c2.evaluate(self.benchmark, self.delta)
                if c1.rank <= p1.rank and c1.rank <= p2.rank:
                    gen_success += 1
                if c2.rank <= p1.rank and c2.rank <= p2.rank:
                    gen_success += 1
                gen_total += 2
                offspring.extend([c1, c2])
            offspring = offspring[:self.pop_size]
            combined = self.population + offspring
            self.assign_rank_and_crowding(combined)
            combined.sort(key=lambda ind: (ind.rank, -ind.crowding))
            self.population = combined[:self.pop_size]
            if gen_total > 0:
                sr = gen_success / gen_total
                self.mutation_rate *= 1.1 if sr > 0.2 else 0.9
                self.mutation_rate = float(np.clip(self.mutation_rate, 0.001, 0.5))
        return self.population


# ============================================================================
# PART 5C: EXTERNAL ALGORITHMS
# ============================================================================

class SPEA2:
    def __init__(self, benchmark, pop_size=60, n_gen=300,
                 p_crossover=0.9, eta_crossover=20.0,
                 p_mutation=None, eta_mutation=20.0,
                 alpha=FAIR_ALPHA, delta=0.30, seed=42,
                 archive_size=None, k_nn=SPEA2_KNN_K):
        self.benchmark = benchmark
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_crossover = p_crossover
        self.eta_crossover = eta_crossover
        self.p_mutation = p_mutation or 1.0 / benchmark.n_var
        self.eta_mutation = eta_mutation
        self.alpha = alpha
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.xl = np.zeros(benchmark.n_var)
        self.xu = np.ones(benchmark.n_var)
        self.archive_size = archive_size or pop_size
        self.k_nn = k_nn
        self.population = []
        self.archive = []

    def _make(self, x):
        ind = Individual(x=x, alpha=self.alpha)
        ind.evaluate(self.benchmark, self.delta)
        return ind

    def initialize(self):
        self.population = [
            self._make(self.rng.uniform(self.xl, self.xu))
            for _ in range(self.pop_size)
        ]
        self.archive = []

    def _spea2_dominance_matrix(self, F_all):
        L = F_all[:, :, 0]
        U = F_all[:, :, 2]
        W = U - L
        num = U[None, :, :] - L[:, None, :]
        den = W[:, None, :] + W[None, :, :]
        safe_den = np.where(den > EPS, den, 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            cred = np.clip(num / safe_den, 0.0, 1.0)
        cred = np.where(den <= EPS,
                        (U[:, None, :] <= L[None, :, :]).astype(float),
                        cred)
        weak_all = np.all(cred >= 0.5, axis=2)
        strict_any = np.any(L[:, None, :] > U[None, :, :], axis=2)
        D = weak_all & strict_any
        np.fill_diagonal(D, False)
        return D

    def _spea2_fitness(self, combined):
        n = len(combined)
        F_all = np.array([ind.fuzzy_obj for ind in combined])
        D = self._spea2_dominance_matrix(F_all)
        strength = D.sum(axis=1)
        raw = (D.T @ strength).astype(int)
        W = np.array([ind.alpha_weighted_obj() for ind in combined])
        dist = np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        k = min(self.k_nn, n - 1)
        sorted_d = np.sort(dist, axis=1)
        density = 1.0 / (sorted_d[:, k - 1] + 2.0) if k > 0 else np.zeros(n)
        fitness = raw + density
        return fitness

    def _environmental_selection(self, combined):
        fitness = self._spea2_fitness(combined)
        order = np.argsort(fitness)
        return [combined[i] for i in order[:self.archive_size]]

    def tournament(self, pool, k=2):
        cand = self.rng.choice(pool, size=min(k, len(pool)), replace=False)
        return min(cand, key=lambda ind: ind.rank)

    def sbx(self, p1, p2):
        if self.rng.random() > self.p_crossover:
            return p1.x.copy(), p2.x.copy()
        u = self.rng.random(self.benchmark.n_var)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1.0 / (self.eta_crossover + 1)),
                        (1.0 / (2 * (1 - u + EPS))) ** (1.0 / (self.eta_crossover + 1)))
        c1 = 0.5 * ((1 + beta) * p1.x + (1 - beta) * p2.x)
        c2 = 0.5 * ((1 - beta) * p1.x + (1 + beta) * p2.x)
        return np.clip(c1, self.xl, self.xu), np.clip(c2, self.xl, self.xu)

    def poly_mut(self, x, p_mutation=None):
        if p_mutation is None:
            p_mutation = self.p_mutation
        y = x.copy()
        mask = self.rng.random(y.shape[0]) < p_mutation
        if not mask.any():
            return y
        u = self.rng.random(int(mask.sum()))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (self.eta_mutation + 1)) - 1.0,
                         1.0 - (2 * (1 - u + EPS)) ** (1.0 / (self.eta_mutation + 1)))
        y[mask] += delta * (self.xu[mask] - self.xl[mask])
        return np.clip(y, self.xl, self.xu)

    def run(self):
        self.initialize()
        self.archive = self._environmental_selection(self.population)
        F_arch = np.array([ind.fuzzy_obj for ind in self.archive])
        fronts = fast_nondominated_sort_fuzzy(F_arch)
        for r, front in enumerate(fronts):
            for idx in front:
                self.archive[idx].rank = r

        for gen in range(self.n_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament(self.archive)
                p2 = self.tournament(self.archive)
                c1x, c2x = self.sbx(p1, p2)
                c1x = self.poly_mut(c1x)
                c2x = self.poly_mut(c2x)
                offspring.append(self._make(c1x))
                offspring.append(self._make(c2x))
            offspring = offspring[:self.pop_size]
            combined = self.archive + offspring
            self.archive = self._environmental_selection(combined)
            F_arch = np.array([ind.fuzzy_obj for ind in self.archive])
            fronts = fast_nondominated_sort_fuzzy(F_arch)
            for r, front in enumerate(fronts):
                for idx in front:
                    self.archive[idx].rank = r

        self.population = self.archive
        return self.population

    def get_pareto_front(self):
        return [ind for ind in self.population if ind.rank == 0]

    def get_alphas(self):
        return np.array([ind.alpha for ind in self.population])


class MOEAD:
    def __init__(self, benchmark, pop_size=60, n_gen=300,
                 p_crossover=0.9, eta_crossover=20.0,
                 p_mutation=None, eta_mutation=20.0,
                 alpha=FAIR_ALPHA, delta=0.30, seed=42,
                 neighbourhood_T=MOEAD_NEIGHBORHOOD_T,
                 n_neighbors=None, divisions=None):
        self.benchmark = benchmark
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_crossover = p_crossover
        self.eta_crossover = eta_crossover
        self.p_mutation = p_mutation or 1.0 / benchmark.n_var
        self.eta_mutation = eta_mutation
        self.alpha = alpha
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.xl = np.zeros(benchmark.n_var)
        self.xu = np.ones(benchmark.n_var)
        self.T = neighbourhood_T

        M = benchmark.n_obj
        H = (divisions or MOEAD_DIVISIONS).get(M, 6)
        weights = das_dennis_points(M, H)
        if len(weights) >= pop_size:
            self.weights = weights[:pop_size]
        else:
            reps = int(np.ceil(pop_size / len(weights)))
            self.weights = np.tile(weights, (reps, 1))[:pop_size]
        self.n_neighbors = n_neighbors or min(self.T, pop_size - 1)

        D = np.sqrt(((self.weights[:, None, :] - self.weights[None, :, :]) ** 2).sum(axis=2))
        self.neighbors = np.argsort(D, axis=1)[:, :self.n_neighbors]

        self.population = []
        self.ideal = np.full(M, np.inf)

    def _make(self, x):
        ind = Individual(x=x, alpha=self.alpha)
        ind.evaluate(self.benchmark, self.delta)
        return ind

    def _scalarized(self, ind, weight):
        F = ind.alpha_weighted_obj()
        return float(np.max(weight * np.abs(F - self.ideal)))

    def _update_ideal(self, ind):
        F = ind.alpha_weighted_obj()
        self.ideal = np.minimum(self.ideal, F)

    def initialize(self):
        self.population = [
            self._make(self.rng.uniform(self.xl, self.xu))
            for _ in range(self.pop_size)
        ]
        for ind in self.population:
            self._update_ideal(ind)

    def sbx(self, p1, p2):
        if self.rng.random() > self.p_crossover:
            return p1.x.copy(), p2.x.copy()
        u = self.rng.random(self.benchmark.n_var)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1.0 / (self.eta_crossover + 1)),
                        (1.0 / (2 * (1 - u + EPS))) ** (1.0 / (self.eta_crossover + 1)))
        c1 = 0.5 * ((1 + beta) * p1.x + (1 - beta) * p2.x)
        c2 = 0.5 * ((1 - beta) * p1.x + (1 + beta) * p2.x)
        return np.clip(c1, self.xl, self.xu), np.clip(c2, self.xl, self.xu)

    def poly_mut(self, x, p_mutation=None):
        if p_mutation is None:
            p_mutation = self.p_mutation
        y = x.copy()
        mask = self.rng.random(y.shape[0]) < p_mutation
        if not mask.any():
            return y
        u = self.rng.random(int(mask.sum()))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (self.eta_mutation + 1)) - 1.0,
                         1.0 - (2 * (1 - u + EPS)) ** (1.0 / (self.eta_mutation + 1)))
        y[mask] += delta * (self.xu[mask] - self.xl[mask])
        return np.clip(y, self.xl, self.xu)

    def run(self):
        self.initialize()
        for gen in range(self.n_gen):
            for i in range(self.pop_size):
                nbrs = self.neighbors[i]
                k1, k2 = self.rng.choice(nbrs, size=2, replace=False)
                p1, p2 = self.population[k1], self.population[k2]
                c1x, c2x = self.sbx(p1, p2)
                c1x = self.poly_mut(c1x)
                c2x = self.poly_mut(c2x)
                c1 = self._make(c1x)
                c2 = self._make(c2x)
                for child in (c1, c2):
                    self._update_ideal(child)
                    for j in nbrs:
                        w = self.weights[j]
                        if (self._scalarized(child, w) <
                                self._scalarized(self.population[j], w)):
                            self.population[j] = child
        F_all = np.array([ind.fuzzy_obj for ind in self.population])
        fronts = fast_nondominated_sort_fuzzy(F_all)
        for r, front in enumerate(fronts):
            for idx in front:
                self.population[idx].rank = r
        return self.population

    def get_pareto_front(self):
        return [ind for ind in self.population if ind.rank == 0]

    def get_alphas(self):
        return np.array([ind.alpha for ind in self.population])


class NSGA3:
    def __init__(self, benchmark, pop_size=60, n_gen=300,
                 p_crossover=0.9, eta_crossover=20.0,
                 p_mutation=None, eta_mutation=20.0,
                 alpha=FAIR_ALPHA, delta=0.30, seed=42,
                 divisions=None):
        self.benchmark = benchmark
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_crossover = p_crossover
        self.eta_crossover = eta_crossover
        self.p_mutation = p_mutation or 1.0 / benchmark.n_var
        self.eta_mutation = eta_mutation
        self.alpha = alpha
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.xl = np.zeros(benchmark.n_var)
        self.xu = np.ones(benchmark.n_var)

        M = benchmark.n_obj
        H = (divisions or NSGA3_DIVISIONS).get(M, 6)
        ref = das_dennis_points(M, H)
        if len(ref) >= pop_size:
            self.ref_points = ref[:pop_size]
        else:
            reps = int(np.ceil(pop_size / len(ref)))
            self.ref_points = np.tile(ref, (reps, 1))[:pop_size]
        self.population = []

    def _make(self, x):
        ind = Individual(x=x, alpha=self.alpha)
        ind.evaluate(self.benchmark, self.delta)
        return ind

    def initialize(self):
        self.population = [
            self._make(self.rng.uniform(self.xl, self.xu))
            for _ in range(self.pop_size)
        ]

    def tournament(self, k=2):
        cand = self.rng.choice(self.population, size=k, replace=False)
        return min(cand, key=lambda ind: (ind.rank, -ind.crowding))

    def sbx(self, p1, p2):
        if self.rng.random() > self.p_crossover:
            return p1.x.copy(), p2.x.copy()
        u = self.rng.random(self.benchmark.n_var)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1.0 / (self.eta_crossover + 1)),
                        (1.0 / (2 * (1 - u + EPS))) ** (1.0 / (self.eta_crossover + 1)))
        c1 = 0.5 * ((1 + beta) * p1.x + (1 - beta) * p2.x)
        c2 = 0.5 * ((1 - beta) * p1.x + (1 + beta) * p2.x)
        return np.clip(c1, self.xl, self.xu), np.clip(c2, self.xl, self.xu)

    def poly_mut(self, x, p_mutation=None):
        if p_mutation is None:
            p_mutation = self.p_mutation
        y = x.copy()
        mask = self.rng.random(y.shape[0]) < p_mutation
        if not mask.any():
            return y
        u = self.rng.random(int(mask.sum()))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (self.eta_mutation + 1)) - 1.0,
                         1.0 - (2 * (1 - u + EPS)) ** (1.0 / (self.eta_mutation + 1)))
        y[mask] += delta * (self.xu[mask] - self.xl[mask])
        return np.clip(y, self.xl, self.xu)

    def _niche_selection(self, combined, n_select):
        if n_select <= 0:
            return []
        F_all = np.array([ind.fuzzy_obj for ind in combined])
        fronts = fast_nondominated_sort_fuzzy(F_all)
        for r, front in enumerate(fronts):
            for idx in front:
                combined[idx].rank = r
        norms = np.linalg.norm(self.ref_points, axis=1)
        norms = np.where(norms < EPS, 1.0, norms)
        ref_unit = self.ref_points / norms[:, None]
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= n_select:
                selected.extend(combined[i] for i in front)
            else:
                remaining = n_select - len(selected)
                W = np.array([combined[i].alpha_weighted_obj() for i in front])
                proj = W @ ref_unit.T
                dist_perp = np.sqrt(np.clip(
                    (W ** 2).sum(axis=1)[:, None] - proj ** 2, 0.0, None))
                nearest = np.argmin(dist_perp, axis=1)
                niche_count = np.zeros(len(self.ref_points), dtype=int)
                if len(selected) > 0:
                    W_sel = np.array([ind.alpha_weighted_obj() for ind in selected])
                    proj_sel = W_sel @ ref_unit.T
                    dist_sel = np.sqrt(np.clip(
                        (W_sel ** 2).sum(axis=1)[:, None] - proj_sel ** 2,
                        0.0, None))
                    nearest_sel = np.argmin(dist_sel, axis=1)
                    for k in nearest_sel:
                        niche_count[k] += 1
                order = np.argsort([niche_count[nearest[k]]
                                    for k in range(len(front))])
                chosen = [combined[front[k]] for k in order[:remaining]]
                selected.extend(chosen)
                break
        return selected

    def run(self):
        self.initialize()
        for gen in range(self.n_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament()
                p2 = self.tournament()
                c1x, c2x = self.sbx(p1, p2)
                c1x = self.poly_mut(c1x)
                c2x = self.poly_mut(c2x)
                offspring.append(self._make(c1x))
                offspring.append(self._make(c2x))
            offspring = offspring[:self.pop_size]
            combined = self.population + offspring
            self.population = self._niche_selection(combined, self.pop_size)
        return self.population

    def get_pareto_front(self):
        return [ind for ind in self.population if ind.rank == 0]

    def get_alphas(self):
        return np.array([ind.alpha for ind in self.population])


class cMLSGA:
    def __init__(self, benchmark, pop_size=60, n_gen=300,
                 p_crossover=0.9, eta_crossover=20.0,
                 p_mutation=None, eta_mutation=20.0,
                 alpha=FAIR_ALPHA, delta=0.30, seed=42,
                 n_subpops=CMLSGA_N_SUBPOPS,
                 migration_freq=CMLSGA_MIGRATION_FREQ):
        self.benchmark = benchmark
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_crossover = p_crossover
        self.eta_crossover = eta_crossover
        self.p_mutation = p_mutation or 1.0 / benchmark.n_var
        self.eta_mutation = eta_mutation
        self.alpha = alpha
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.xl = np.zeros(benchmark.n_var)
        self.xu = np.ones(benchmark.n_var)
        self.n_subpops = max(1, n_subpops)
        self.migration_freq = migration_freq
        self.deme_size = max(2, pop_size // self.n_subpops)
        self.population = []

    def _make(self, x):
        ind = Individual(x=x, alpha=self.alpha)
        ind.evaluate(self.benchmark, self.delta)
        return ind

    def _make_deme(self):
        return [self._make(self.rng.uniform(self.xl, self.xu))
                for _ in range(self.deme_size)]

    def initialize(self):
        self.demes = [self._make_deme() for _ in range(self.n_subpops)]

    def tournament(self, pool, k=2):
        cand = self.rng.choice(pool, size=min(k, len(pool)), replace=False)
        return min(cand, key=lambda ind: ind.rank)

    def sbx(self, p1, p2):
        if self.rng.random() > self.p_crossover:
            return p1.x.copy(), p2.x.copy()
        u = self.rng.random(self.benchmark.n_var)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1.0 / (self.eta_crossover + 1)),
                        (1.0 / (2 * (1 - u + EPS))) ** (1.0 / (self.eta_crossover + 1)))
        c1 = 0.5 * ((1 + beta) * p1.x + (1 - beta) * p2.x)
        c2 = 0.5 * ((1 - beta) * p1.x + (1 + beta) * p2.x)
        return np.clip(c1, self.xl, self.xu), np.clip(c2, self.xl, self.xu)

    def poly_mut(self, x, p_mutation=None):
        if p_mutation is None:
            p_mutation = self.p_mutation
        y = x.copy()
        mask = self.rng.random(y.shape[0]) < p_mutation
        if not mask.any():
            return y
        u = self.rng.random(int(mask.sum()))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (self.eta_mutation + 1)) - 1.0,
                         1.0 - (2 * (1 - u + EPS)) ** (1.0 / (self.eta_mutation + 1)))
        y[mask] += delta * (self.xu[mask] - self.xl[mask])
        return np.clip(y, self.xl, self.xu)

    def _rank_deme(self, deme):
        F = np.array([ind.fuzzy_obj for ind in deme])
        fronts = fast_nondominated_sort_fuzzy(F)
        for r, front in enumerate(fronts):
            for idx in front:
                deme[idx].rank = r

    def _evolve_deme(self, deme, n_gen):
        for _ in range(n_gen):
            offspring = []
            while len(offspring) < self.deme_size:
                p1 = self.tournament(deme)
                p2 = self.tournament(deme)
                c1x, c2x = self.sbx(p1, p2)
                c1x = self.poly_mut(c1x)
                c2x = self.poly_mut(c2x)
                offspring.append(self._make(c1x))
                offspring.append(self._make(c2x))
            offspring = offspring[:self.deme_size]
            combined = deme + offspring
            self._rank_deme(combined)
            combined.sort(key=lambda ind: ind.rank)
            deme[:] = combined[:self.deme_size]
        return deme

    def _migrate(self):
        bests = [min(d, key=lambda ind: ind.rank) for d in self.demes]
        for i, deme in enumerate(self.demes):
            j = (i + 1) % self.n_subpops
            k = int(self.rng.integers(len(self.demes[j])))
            self.demes[j][k] = bests[i]

    def run(self):
        self.initialize()
        generations_per_migration = max(1, self.migration_freq)
        n_migrations = max(1, self.n_gen // generations_per_migration)
        gens_per_round = max(1, self.n_gen // n_migrations)
        for _ in range(n_migrations):
            for i in range(self.n_subpops):
                self.demes[i] = self._evolve_deme(
                    self.demes[i], gens_per_round)
            self._migrate()
        self.population = [ind for deme in self.demes for ind in deme]
        self._rank_deme(self.population)
        return self.population

    def get_pareto_front(self):
        return [ind for ind in self.population if ind.rank == 0]

    def get_alphas(self):
        return np.array([ind.alpha for ind in self.population])


class _AlphaMixin:
    def _init_alpha(self):
        self.alpha_L = 0.5
        self.alpha_U = 1.0
        self._alpha_rng = self.rng

    def _random_alpha(self):
        return float(self._alpha_rng.uniform(self.alpha_L, self.alpha_U))

    def _blx_alpha(self, a1, a2):
        gamma = self._alpha_rng.uniform(-0.5, 1.5)
        return float(np.clip(a1 + gamma * (a2 - a1),
                             self.alpha_L, self.alpha_U))

    def _gauss_alpha(self, alpha, gen):
        sigma = 0.05 * max(0.0, 1.0 - gen / max(1, self.n_gen))
        return float(np.clip(alpha + self._alpha_rng.normal(0, sigma),
                             self.alpha_L, self.alpha_U))


class AlphaWrappedSPEA2(_AlphaMixin, SPEA2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_alpha()
        self.alpha = None

    def _make(self, x, alpha=None):
        a = self._random_alpha() if alpha is None else alpha
        ind = Individual(x=x, alpha=a)
        ind.evaluate(self.benchmark, self.delta)
        return ind

    def initialize(self):
        self.population = [
            self._make(self.rng.uniform(self.xl, self.xu))
            for _ in range(self.pop_size)
        ]
        self.archive = []

    def run(self):
        self.initialize()
        self.archive = self._environmental_selection(self.population)
        F_arch = np.array([ind.fuzzy_obj for ind in self.archive])
        fronts = fast_nondominated_sort_fuzzy(F_arch)
        for r, front in enumerate(fronts):
            for idx in front:
                self.archive[idx].rank = r

        for gen in range(self.n_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament(self.archive)
                p2 = self.tournament(self.archive)
                c1x, c2x = self.sbx(p1, p2)
                c1x = self.poly_mut(c1x)
                c2x = self.poly_mut(c2x)
                c1a = self._blx_alpha(p1.alpha, p2.alpha)
                c2a = self._blx_alpha(p1.alpha, p2.alpha)
                c1a = self._gauss_alpha(c1a, gen)
                c2a = self._gauss_alpha(c2a, gen)
                offspring.append(self._make(c1x, c1a))
                offspring.append(self._make(c2x, c2a))
            offspring = offspring[:self.pop_size]
            combined = self.archive + offspring
            self.archive = self._environmental_selection(combined)
            F_arch = np.array([ind.fuzzy_obj for ind in self.archive])
            fronts = fast_nondominated_sort_fuzzy(F_arch)
            for r, front in enumerate(fronts):
                for idx in front:
                    self.archive[idx].rank = r
        self.population = self.archive
        return self.population


class AlphaWrappedMOEAD(_AlphaMixin, MOEAD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_alpha()
        self.alpha = None

    def _make(self, x, alpha=None):
        a = self._random_alpha() if alpha is None else alpha
        ind = Individual(x=x, alpha=a)
        ind.evaluate(self.benchmark, self.delta)
        return ind

    def initialize(self):
        self.population = [
            self._make(self.rng.uniform(self.xl, self.xu))
            for _ in range(self.pop_size)
        ]
        for ind in self.population:
            self._update_ideal(ind)

    def run(self):
        self.initialize()
        for gen in range(self.n_gen):
            for i in range(self.pop_size):
                nbrs = self.neighbors[i]
                k1, k2 = self.rng.choice(nbrs, size=2, replace=False)
                p1, p2 = self.population[k1], self.population[k2]
                c1x, c2x = self.sbx(p1, p2)
                c1x = self.poly_mut(c1x)
                c2x = self.poly_mut(c2x)
                c1a = self._gauss_alpha(self._blx_alpha(p1.alpha, p2.alpha), gen)
                c2a = self._gauss_alpha(self._blx_alpha(p1.alpha, p2.alpha), gen)
                c1 = self._make(c1x, c1a)
                c2 = self._make(c2x, c2a)
                for child in (c1, c2):
                    self._update_ideal(child)
                    for j in nbrs:
                        w = self.weights[j]
                        if (self._scalarized(child, w) <
                                self._scalarized(self.population[j], w)):
                            self.population[j] = child
        F_all = np.array([ind.fuzzy_obj for ind in self.population])
        fronts = fast_nondominated_sort_fuzzy(F_all)
        for r, front in enumerate(fronts):
            for idx in front:
                self.population[idx].rank = r
        return self.population


class AlphaWrappedNSGA3(_AlphaMixin, NSGA3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_alpha()
        self.alpha = None

    def _make(self, x, alpha=None):
        a = self._random_alpha() if alpha is None else alpha
        ind = Individual(x=x, alpha=a)
        ind.evaluate(self.benchmark, self.delta)
        return ind

    def initialize(self):
        self.population = [
            self._make(self.rng.uniform(self.xl, self.xu))
            for _ in range(self.pop_size)
        ]

    def run(self):
        self.initialize()
        for gen in range(self.n_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament()
                p2 = self.tournament()
                c1x, c2x = self.sbx(p1, p2)
                c1x = self.poly_mut(c1x)
                c2x = self.poly_mut(c2x)
                c1a = self._gauss_alpha(self._blx_alpha(p1.alpha, p2.alpha), gen)
                c2a = self._gauss_alpha(self._blx_alpha(p1.alpha, p2.alpha), gen)
                offspring.append(self._make(c1x, c1a))
                offspring.append(self._make(c2x, c2a))
            offspring = offspring[:self.pop_size]
            combined = self.population + offspring
            self.population = self._niche_selection(combined, self.pop_size)
        return self.population


class AlphaWrappedcMLSGA(_AlphaMixin, cMLSGA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_alpha()
        self.alpha = None

    def _make(self, x, alpha=None):
        a = self._random_alpha() if alpha is None else alpha
        ind = Individual(x=x, alpha=a)
        ind.evaluate(self.benchmark, self.delta)
        return ind

    def _make_deme(self):
        return [self._make(self.rng.uniform(self.xl, self.xu))
                for _ in range(self.deme_size)]

    def _evolve_deme(self, deme, n_gen):
        for gen in range(n_gen):
            offspring = []
            while len(offspring) < self.deme_size:
                p1 = self.tournament(deme)
                p2 = self.tournament(deme)
                c1x, c2x = self.sbx(p1, p2)
                c1x = self.poly_mut(c1x)
                c2x = self.poly_mut(c2x)
                c1a = self._gauss_alpha(self._blx_alpha(p1.alpha, p2.alpha), gen)
                c2a = self._gauss_alpha(self._blx_alpha(p1.alpha, p2.alpha), gen)
                offspring.append(self._make(c1x, c1a))
                offspring.append(self._make(c2x, c2a))
            offspring = offspring[:self.deme_size]
            combined = deme + offspring
            self._rank_deme(combined)
            combined.sort(key=lambda ind: ind.rank)
            deme[:] = combined[:self.deme_size]
        return deme

    def run(self):
        self.initialize()
        generations_per_migration = max(1, self.migration_freq)
        n_migrations = max(1, self.n_gen // generations_per_migration)
        gens_per_round = max(1, self.n_gen // n_migrations)
        for _ in range(n_migrations):
            for i in range(self.n_subpops):
                self.demes[i] = self._evolve_deme(self.demes[i], gens_per_round)
            self._migrate()
        self.population = [ind for deme in self.demes for ind in deme]
        self._rank_deme(self.population)
        return self.population


# ============================================================================
# PART 6: METRICS
# ============================================================================

def hypervolume(F, ref):
    if len(F) == 0:
        return 0.0
    mask = np.all(F <= ref, axis=1)
    F = F[mask]
    if len(F) == 0:
        return 0.0
    if F.shape[1] == 2:
        return _hv_2d(F, ref)
    return _hv_mc(F, ref, n_samples=3000)


def _hv_2d(F, ref):
    idx = np.argsort(F[:, 0])
    F = F[idx]
    hv = 0.0
    prev_y = ref[1]
    for x, y in F:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


_MC_RNG = np.random.default_rng(0)


def _hv_mc(F, ref, n_samples=3000):
    lows = np.min(F, axis=0)
    samples = _MC_RNG.uniform(lows, ref, size=(n_samples, F.shape[1]))
    dominated = np.zeros(n_samples, dtype=bool)
    for f in F:
        dominated |= np.all(samples >= f, axis=1)
    return float(np.mean(dominated) * np.prod(ref - lows))


def _fixed_alpha_objectives(population, alpha=FAIR_ALPHA, force=False):
    pf = [ind for ind in population if ind.rank == 0]
    if not pf:
        return np.empty((0, 0))
    out = np.empty((len(pf), pf[0].fuzzy_obj.shape[0]))
    for i, ind in enumerate(pf):
        if force:
            a = float(alpha)
        elif ind.alpha is None:
            a = float(alpha)
        else:
            a = float(ind.alpha)
        out[i] = a * ind.fuzzy_obj[:, 1] + (1 - a) * (ind.fuzzy_obj[:, 0] + ind.fuzzy_obj[:, 2]) / 2.0
    return out


def compute_pareto_hv(population, fixed_alpha=FAIR_ALPHA,
                      alpha_mode='per_individual'):
    force = (alpha_mode == 'fixed')
    F = _fixed_alpha_objectives(population, alpha=fixed_alpha, force=force)
    if F.size == 0:
        return 0.0, np.array([])
    ref = np.max(F, axis=0) * 1.1 + 1.0
    return hypervolume(F, ref), F


def _igd_gd_against_reference(F, ref_front):
    F = np.asarray(F, dtype=float)
    ref_front = np.asarray(ref_front, dtype=float)
    if F.size == 0 or ref_front.size == 0:
        return float('nan'), float('nan')
    D = np.sqrt(((ref_front[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))
    igd = float(np.mean(np.min(D, axis=1)))
    gd = float(np.mean(np.min(D, axis=0)))
    return igd, gd


def _ref_front_diagonal(ref_front):
    ref_front = np.asarray(ref_front, dtype=float)
    if ref_front.size == 0:
        return 1.0
    span = ref_front.max(axis=0) - ref_front.min(axis=0)
    diag = float(np.sqrt((span ** 2).sum()))
    return diag if diag > EPS else 1.0


def build_empirical_reference_front(all_fronts,
                                    k=REF_FRONT_MAX_OUTPUT,
                                    seed=0,
                                    max_sort_size=REF_FRONT_MAX_INPUT):
    if not all_fronts:
        return np.empty((0, 0))
    stacked = np.vstack([F for F in all_fronts if F.size > 0])
    if stacked.size == 0:
        return np.empty((0, 0))
    if len(stacked) > max_sort_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(stacked), size=max_sort_size, replace=False)
        stacked = stacked[idx]
    uniq = np.unique(np.round(stacked, decimals=10), axis=0)
    M = uniq.shape[1]
    F_tfn = np.stack([uniq, uniq, uniq], axis=2)
    fronts = fast_nondominated_sort_fuzzy(F_tfn, chunk_size=256)
    front0 = fronts[0] if fronts else []
    nd = uniq[front0] if len(front0) > 0 else uniq
    if len(nd) > k:
        order = np.argsort(nd[:, 0])
        idx = np.linspace(0, len(nd) - 1, k).astype(int)
        nd = nd[order][idx]
    return nd


# ============================================================================
# PART 6B: Σ̂ PROXY
# ============================================================================

def compute_sigma_hat(prob_name, alphas=SIGMA_HAT_ALPHAS,
                      seed=2026, pop_size=60, n_gen=300,
                      n_runs_per_alpha=SIGMA_HAT_RUNS_PER_ALPHA,
                      ref_quantile=SIGMA_HAT_REF_QUANTILE):
    """Evaluate Σ̂ using the SAME population under each α value."""
    benchmark = get_extended_benchmark(prob_name)
    algo = AAlphaNSGA2AdaptiveMutation

    per_alpha_fronts = {a: [] for a in alphas}
    rng = np.random.default_rng(seed)

    for r in range(n_runs_per_alpha):
        s = int(rng.integers(0, 2**31 - 1))
        algo_obj = algo(benchmark, pop_size=pop_size, n_gen=n_gen, seed=s)
        pop = algo_obj.run()
        for a in alphas:
            F = _fixed_alpha_objectives(pop, alpha=a, force=True)
            if F.size > 0:
                per_alpha_fronts[a].append(F)

    all_F = [F for a in alphas for F in per_alpha_fronts[a]]
    if not all_F:
        return float('nan'), [float('nan')] * len(alphas)
    stacked = np.vstack(all_F)
    ref = np.quantile(stacked, ref_quantile, axis=0) * 1.05 + 1e-6

    mean_hvs = []
    for a in alphas:
        hvs_a = []
        for F in per_alpha_fronts[a]:
            mask = np.all(F <= ref, axis=1)
            F_clipped = F[mask]
            hvs_a.append(hypervolume(F_clipped, ref)
                         if F_clipped.size > 0 else 0.0)
        mean_hvs.append(float(np.mean(hvs_a)) if hvs_a else 0.0)

    mean_hv = float(np.mean(mean_hvs))
    sigma = (max(mean_hvs) - min(mean_hvs)) / mean_hv if mean_hv > EPS else 0.0
    return float(sigma), mean_hvs


def _probe_f2_override():
    print("\n  [probe] F2 modulation check:")
    x_lo = np.array([0.0] + [0.5] * 29)
    x_hi = np.array([1.0] + [0.5] * 29)
    for name, xs in (('FZDT1', x_lo), ('F2FZDT1 @x0=0', x_lo),
                     ('F2FZDT1 @x0=1', x_hi)):
        b = get_extended_benchmark(name if name == 'FZDT1' else 'F2FZDT1')
        fz = b.evaluate_fuzzy(xs, delta=0.30)
        widths = fz[:, 3] - fz[:, 0]
        print(f"    {name:<18} f1 TFN=[{fz[0,0]:+.3f},{fz[0,1]:+.3f},"
              f"{fz[0,2]:+.3f},{fz[0,3]:+.3f}]  width={widths[0]:.3f}")


def verify_sigma_hat_families(sigma_dir):
    csv_path = os.path.join(sigma_dir, 'sigma_hat.csv')
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
        f1 = df[df['family'] == 'F1']['sigma_hat'].values
        f2 = df[df['family'] == 'F2']['sigma_hat'].values
        if len(f1) and len(f2):
            m1, m2 = float(np.mean(f1)), float(np.mean(f2))
            ratio = m2 / m1 if m1 > EPS else float('inf')
            print(f"\n  Σ̂ sanity: mean F1 = {m1:.4f}  mean F2 = {m2:.4f}  "
                  f"ratio = {ratio:.2f}")
            if m1 > 0.015:
                print(f"  ⚠ NOTE: F1 mean Σ̂ = {m1:.4f} is not ≈ 0. "
                      f"This is expected because F1's trapezoidal geometry "
                      f"has centroid ≠ midpoint, so α-weighting still "
                      f"changes HV even for a fixed population.")
            if ratio < 1.5:
                print(f"  ⚠ WARNING: F2/F1 Σ̂ ratio = {ratio:.2f} is "
                      f"too small. F2 modulation may not be active. "
                      f"Check the [probe] output at startup.")
            else:
                print(f"  ✓ F2/F1 Σ̂ ratio = {ratio:.2f} — F2 is "
                      f"correctly more α-sensitive than F1.")
    except Exception as e:
        print(f"  Σ̂ sanity check failed: {e}")


# ============================================================================
# PART 7: CONFIG
# ============================================================================

CONFIG = {
    'problems': get_all_problems(),
    'families': FAMILIES,
    'n_runs': 10,
    'pop_size': 60,
    'n_gen': 300,
    'seed_base': 2026,
    'n_workers': None,
    'use_parallel': True,
    'quiet': False,
    'results_dir':       os.path.join(OUTPUT_ROOT, 'results'),
    'figures_dir':       os.path.join(OUTPUT_ROOT, 'figures'),
    'latex_dir':         os.path.join(OUTPUT_ROOT, 'latex'),
    'supp_latex_dir':    os.path.join(OUTPUT_ROOT, 'supplementary_latex'),
    'sigma_hat_dir':     os.path.join(OUTPUT_ROOT, 'sigma_hat'),
    'external_dir':      os.path.join(OUTPUT_ROOT, 'external'),
    'reference_algo':    'Aα-NSGA-II (adaptive mutation)',
    'bootstrap_samples': 5000,
    'sigma_hat_seed':    2026,
    'sigma_hat_problems': F2_NAMES
                          + ['FZDT1', 'FZDT2', 'FDTLZ2',
                             'FUF1', 'FUF2', 'FMaF1', 'FMaF2'],
}

INTERNAL_ALGO_NAMES = [
    'Aα-NSGA-II (adaptive mutation)',
    'Aα-NSGA-II',
    'NSGA-II',
    'NSGA-II (high mutation)',
    'Self-Adaptive Mutation',
]

EXTERNAL_ALGO_NAMES = [
    'SPEA2+α',
    'MOEA/D+α',
    'NSGA-III+α',
    'cMLSGA+α',
]

EXTERNAL_ALGO_NAMES_STRICT = [
    'SPEA2',
    'MOEA/D',
    'NSGA-III',
    'cMLSGA',
]

ALGO_NAMES = INTERNAL_ALGO_NAMES + EXTERNAL_ALGO_NAMES + EXTERNAL_ALGO_NAMES_STRICT

ALGO_COLORS = {
    'Aα-NSGA-II (adaptive mutation)': '#1f77b4',
    'SPEA2': '#d62728',
    'MOEA/D': '#2ca02c',
    'NSGA-III': '#ff7f0e',
    'cMLSGA': '#9467bd',
    'Aα-NSGA-II': '#8c564b',
    'NSGA-II': '#e377c2',
    'NSGA-II (high mutation)': '#7f7f7f',
    'Self-Adaptive Mutation': '#bcbd22',
    'SPEA2+α': '#17becf',
    'MOEA/D+α': '#aec7e8',
    'NSGA-III+α': '#ffbb78',
    'cMLSGA+α': '#98df8a',
}

ALGO_MARKERS = {
    'Aα-NSGA-II (adaptive mutation)': 'o',
    'SPEA2': 's',
    'MOEA/D': '^',
    'NSGA-III': 'D',
    'cMLSGA': 'v',
    'Aα-NSGA-II': 'p',
    'NSGA-II': 'X',
    'NSGA-II (high mutation)': '*',
    'Self-Adaptive Mutation': 'h',
    'SPEA2+α': 'o',
    'MOEA/D+α': 's',
    'NSGA-III+α': '^',
    'cMLSGA+α': 'D',
}


def get_algo_config(pop_size, n_gen, external_mode=None):
    if external_mode is None:
        external_mode = EXTERNAL_COMPARISON_MODE

    common = {'pop_size': pop_size, 'n_gen': n_gen}
    algos = [
        ('Aα-NSGA-II (adaptive mutation)',
         AAlphaNSGA2AdaptiveMutation, {**common}),
        ('Aα-NSGA-II',
         AAlphaNSGA2, {**common}),
        ('NSGA-II',
         BaselineNSGA2, {**common, 'alpha': FAIR_ALPHA}),
        ('NSGA-II (high mutation)',
         BaselineNSGA2HighMutation, {**common, 'alpha': FAIR_ALPHA}),
        ('Self-Adaptive Mutation',
         BaselineSelfAdaptiveMutation, {**common, 'alpha': FAIR_ALPHA}),
    ]

    if external_mode in ('fair', 'both'):
        algos.extend([
            ('SPEA2+α',
             AlphaWrappedSPEA2, {**common,
                                 'archive_size': SPEA2_ARCHIVE_SIZE,
                                 'k_nn': SPEA2_KNN_K}),
            ('MOEA/D+α',
             AlphaWrappedMOEAD, {**common,
                                 'neighbourhood_T': MOEAD_NEIGHBORHOOD_T,
                                 'divisions': MOEAD_DIVISIONS}),
            ('NSGA-III+α',
             AlphaWrappedNSGA3, {**common,
                                 'divisions': NSGA3_DIVISIONS}),
            ('cMLSGA+α',
             AlphaWrappedcMLSGA, {**common,
                                  'n_subpops': CMLSGA_N_SUBPOPS,
                                  'migration_freq': CMLSGA_MIGRATION_FREQ}),
        ])

    if external_mode in ('strict', 'both'):
        algos.extend([
            ('SPEA2',
             SPEA2, {**common, 'alpha': FAIR_ALPHA,
                     'archive_size': SPEA2_ARCHIVE_SIZE,
                     'k_nn': SPEA2_KNN_K}),
            ('MOEA/D',
             MOEAD, {**common, 'alpha': FAIR_ALPHA,
                     'neighbourhood_T': MOEAD_NEIGHBORHOOD_T,
                     'divisions': MOEAD_DIVISIONS}),
            ('NSGA-III',
             NSGA3, {**common, 'alpha': FAIR_ALPHA,
                     'divisions': NSGA3_DIVISIONS}),
            ('cMLSGA',
             cMLSGA, {**common, 'alpha': FAIR_ALPHA,
                      'n_subpops': CMLSGA_N_SUBPOPS,
                      'migration_freq': CMLSGA_MIGRATION_FREQ}),
        ])

    return algos


# ============================================================================
# PART 8: WORKER + IGD/GD PASS
# ============================================================================

def _worker_init():
    try:
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    except Exception:
        pass


def _save_checkpoint(results, path):
    if path is None:
        return
    try:
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(results, f)
        os.replace(tmp, path)
    except Exception:
        pass


def run_single_problem(args):
    prob_name, n_runs, pop_size, n_gen, seed_base = args
    try:
        benchmark = get_extended_benchmark(prob_name)
        algos = get_algo_config(pop_size, n_gen)
        results = {}
        for algo_name, algo_cls, kwargs in algos:
            hvs_raw, alphas = [], []
            errors = 0
            for run_idx in range(n_runs):
                seed = seed_base + run_idx
                try:
                    algo = algo_cls(benchmark, seed=seed, **kwargs)
                    pop = algo.run()
                except Exception:
                    errors += 1
                    continue
                hv_raw, F = compute_pareto_hv(pop)
                if len(F) == 0:
                    continue
                hvs_raw.append(hv_raw)
                if hasattr(algo, 'get_alphas'):
                    try:
                        alphas.append(float(np.mean(algo.get_alphas())))
                    except Exception:
                        pass
            results[algo_name] = {
                'hv_raw_mean': float(np.mean(hvs_raw)) if hvs_raw else 0.0,
                'hv_raw_std': float(np.std(hvs_raw)) if hvs_raw else 0.0,
                'hv_raw_all': hvs_raw,
                'alpha_mean': float(np.mean(alphas)) if alphas else np.nan,
                'alpha_mean_all': alphas,
                'n_errors': errors,
                'igd_mean': np.nan, 'igd_std': np.nan, 'igd_all': [],
                'gd_mean': np.nan, 'gd_std': np.nan, 'gd_all': [],
                'igd_norm_mean': np.nan, 'gd_norm_mean': np.nan,
            }
        return (prob_name, results)
    except Exception as e:
        return (prob_name, {
            '_error': f"{type(e).__name__}: {e}",
            '_traceback': traceback.format_exc(),
        })


def _igd_gd_checkpoint_path():
    return os.path.join(CONFIG['results_dir'], 'igd_gd_checkpoint.pkl')


def _load_igd_gd_checkpoint():
    p = _igd_gd_checkpoint_path()
    if os.path.exists(p):
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _save_igd_gd_checkpoint(state):
    p = _igd_gd_checkpoint_path()
    try:
        tmp = p + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(state, f)
        os.replace(tmp, p)
    except Exception:
        pass


def compute_igd_gd_pass(all_results, problems, pop_size, n_gen,
                        seed_base, n_runs, verbose=True):
    if verbose:
        print("\n" + "=" * 78)
        print("IGD/GD POST-PROCESSING PASS (v8.8)")
        print("=" * 78)

    state = _load_igd_gd_checkpoint()
    if verbose and state:
        print(f"  Resuming: {len(state)} problems already processed")

    algos = get_algo_config(pop_size, n_gen)
    n_cached, n_computed, n_skipped = 0, 0, 0

    for prob_idx, prob_name in enumerate(problems, 1):
        if prob_name in state:
            cached = state[prob_name]
            if isinstance(cached, dict) and cached.get('__skipped__'):
                if verbose:
                    print(f"  [{prob_idx}/{len(problems)}] {prob_name:<14} "
                          f"(skipped, cached)")
                n_skipped += 1
                continue
            for algo_name, vals in cached.items():
                if algo_name in all_results.get(prob_name, {}):
                    all_results[prob_name][algo_name].update(vals)
            if verbose:
                print(f"  [{prob_idx}/{len(problems)}] {prob_name:<14} "
                      f"(cached)")
            n_cached += 1
            continue

        if prob_name in IGD_SKIP:
            if verbose:
                print(f"  [{prob_idx}/{len(problems)}] {prob_name:<14} "
                      f"IGD/GD skipped (reference front unstable)")
            state[prob_name] = {'__skipped__': True}
            _save_igd_gd_checkpoint(state)
            n_skipped += 1
            continue

        if prob_name not in all_results:
            continue
        res = all_results[prob_name]
        if '_error' in res:
            continue

        try:
            benchmark = get_extended_benchmark(prob_name)
        except Exception:
            continue

        per_algo_fronts = {a: [] for a, _, _ in algos}
        for algo_name, algo_cls, kwargs in algos:
            if algo_name not in res:
                continue
            for run_idx in range(n_runs):
                seed = seed_base + run_idx
                try:
                    algo = algo_cls(benchmark, seed=seed, **kwargs)
                    pop = algo.run()
                except Exception:
                    continue
                F = _fixed_alpha_objectives(pop, alpha=FAIR_ALPHA,
                                             force=False)
                if F.size > 0:
                    per_algo_fronts[algo_name].append(F)
                del algo, pop

        balanced_fronts = []
        for algo_name, fronts in per_algo_fronts.items():
            if not fronts:
                continue
            algo_points = np.vstack(fronts)
            if len(algo_points) > IGD_PER_ALGO_CAP:
                rng_local = np.random.default_rng(
                    seed_base + (abs(hash(algo_name)) % (2**31)))
                idx = rng_local.choice(len(algo_points),
                                       size=IGD_PER_ALGO_CAP,
                                       replace=False)
                algo_points = algo_points[idx]
            balanced_fronts.append(algo_points)

        ref_front = build_empirical_reference_front(
            balanced_fronts, k=REF_FRONT_MAX_OUTPUT)
        if ref_front.size == 0:
            del per_algo_fronts, balanced_fronts
            gc.collect()
            continue
        ref_diag = _ref_front_diagonal(ref_front)

        cached_for_problem = {}
        for algo_name, fronts in per_algo_fronts.items():
            if algo_name not in res:
                continue
            igds, gds, igds_n, gds_n = [], [], [], []
            for F in fronts:
                igd, gd = _igd_gd_against_reference(F, ref_front)
                if not np.isnan(igd):
                    igds.append(igd)
                    igds_n.append(igd / ref_diag)
                if not np.isnan(gd):
                    gds.append(gd)
                    gds_n.append(gd / ref_diag)
            vals = {
                'igd_mean': float(np.mean(igds)) if igds else np.nan,
                'igd_std':  float(np.std(igds))  if igds else np.nan,
                'igd_all':  igds,
                'gd_mean':  float(np.mean(gds))  if gds else np.nan,
                'gd_std':   float(np.std(gds))   if gds else np.nan,
                'gd_all':   gds,
                'igd_norm_mean': float(np.mean(igds_n)) if igds_n else np.nan,
                'gd_norm_mean':  float(np.mean(gds_n))  if gds_n else np.nan,
            }
            res[algo_name].update(vals)
            cached_for_problem[algo_name] = vals

        del per_algo_fronts, balanced_fronts, ref_front, benchmark
        gc.collect()

        state[prob_name] = cached_for_problem
        _save_igd_gd_checkpoint(state)
        n_computed += 1

        if verbose:
            ref_name = CONFIG['reference_algo']
            igd_dbg = cached_for_problem.get(ref_name, {}).get(
                'igd_mean', float('nan'))
            print(f"  [{prob_idx}/{len(problems)}] {prob_name:<14} "
                  f"ref=|{len(cached_for_problem)}|  "
                  f"IGD({ref_name})={igd_dbg:.4f}")

    if verbose:
        print(f"\n  IGD/GD summary: "
              f"{n_cached} cached, {n_computed} computed, "
              f"{n_skipped} skipped")

    if verbose:
        n_missing = 0
        for prob_name, res in all_results.items():
            if not isinstance(res, dict) or '_error' in res:
                continue
            for algo_name in res:
                if algo_name.startswith('_'):
                    continue
                igd = res[algo_name].get('igd_mean', np.nan)
                if np.isnan(igd):
                    n_missing += 1
        if n_missing > 0:
            print(f"  ⚠ {n_missing} (problem, algorithm) pairs still have "
                  f"NaN IGD. This is expected for FScal5000/FScal10000 "
                  f"and for any problem in IGD_SKIP.")

    return all_results


# ============================================================================
# PART 9: SERIAL / PARALLEL EXECUTION
# ============================================================================

def _wait_some(futures, timeout=300):
    from concurrent.futures import wait, FIRST_COMPLETED
    done, not_done = wait(futures, timeout=timeout,
                          return_when=FIRST_COMPLETED)
    return list(done), list(not_done)


def run_serial(problems, n_runs, pop_size, n_gen, seed_base,
               checkpoint_path=None, verbose=True, monitor=None):
    all_results = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                all_results = pickle.load(f)
            if verbose:
                print(f"  Loaded {len(all_results)} finished problems")
        except Exception:
            all_results = {}
    todo = [p for p in problems if p not in all_results]
    tasks = [(p, n_runs, pop_size, n_gen, seed_base + i * 10000)
             for i, p in enumerate(todo)]
    if verbose:
        print(f"\n[SERIAL] {len(todo)} problems to run "
              f"({len(all_results)} cached)")
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        prob_name = task[0]
        if monitor:
            monitor.start_task(prob_name)
        if verbose and not CONFIG['quiet']:
            print(f"  [{i}/{len(tasks)}] Running {prob_name}...", flush=True)
        _, results = run_single_problem(task)
        if '_error' in results:
            if verbose:
                print(f"    ✗ {results['_error']}")
            if monitor:
                monitor.task_done(prob_name, success=False)
            continue
        all_results[prob_name] = results
        _save_checkpoint(all_results, checkpoint_path)
        if monitor:
            monitor.task_done(prob_name, success=True)
    if verbose:
        print(f"\nAll done in {(time.time() - t0) / 60:.1f} min")
    return all_results


def run_parallel(problems, n_runs, pop_size, n_gen, seed_base,
                 n_workers=None, checkpoint_path=None, verbose=True,
                 monitor=None):
    if IN_JUPYTER and not FORCE_PARALLEL_FROM_JUPYTER:
        if verbose:
            print("\n[JUPYTER MODE] Using serial execution for stability.")
            print("    (Set FORCE_PARALLEL_FROM_JUPYTER = True to override.)")
        return run_serial(problems, n_runs, pop_size, n_gen, seed_base,
                          checkpoint_path, verbose, monitor)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)
    n_workers = max(1, min(n_workers, len(problems)))

    all_results = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                all_results = pickle.load(f)
            if verbose:
                print(f"  Loaded {len(all_results)} finished problems")
        except Exception:
            all_results = {}

    todo = [p for p in problems if p not in all_results]
    tasks = [(p, n_runs, pop_size, n_gen, seed_base + i * 10000)
             for i, p in enumerate(todo)]
    if not todo:
        if verbose:
            print("  All problems already cached.")
        return all_results

    if verbose:
        print(f"\n[PARALLEL] Workers: {n_workers} | "
              f"{len(todo)} problems to run ({len(all_results)} cached)\n")

    t0 = time.time()
    quiet = CONFIG.get('quiet', False)
    in_flight = {}

    def _track_start(name):
        in_flight[name] = time.time()
        if monitor:
            monitor.start_task(name)

    def _track_done(prob_name, results, completed):
        in_flight.pop(prob_name, None)
        if '_error' in results:
            if verbose:
                print(f"  ✗ {prob_name}: {results['_error']}", flush=True)
            if monitor:
                monitor.task_done(prob_name, success=False)
            return
        all_results[prob_name] = results
        _save_checkpoint(all_results, checkpoint_path)
        if monitor:
            monitor.task_done(prob_name, success=True)
        elif verbose and not quiet:
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1e-6)
            eta = (len(tasks) - completed) / max(rate, 1e-6)
            print(f"  [{completed}/{len(tasks)}] {prob_name} done "
                  f"(ETA: {eta / 60:.1f} min)", flush=True)

    try:
        pe_kwargs = dict(
            max_workers=n_workers,
            mp_context=mp.get_context('spawn'),
            initializer=_worker_init,
        )
        try:
            pe = ProcessPoolExecutor(max_tasks_per_child=1, **pe_kwargs)
        except TypeError:
            pe = ProcessPoolExecutor(**pe_kwargs)
        with pe as executor:
            futures = {}
            for t in tasks:
                _track_start(t[0])
                futures[executor.submit(run_single_problem, t)] = t
            completed = 0
            pending = list(futures.keys())
            while pending:
                done, pending = _wait_some(pending, timeout=300)
                for fut in done:
                    try:
                        prob_name, results = fut.result(timeout=60)
                    except Exception as e:
                        if verbose:
                            print(f"  ✗ Worker crashed: {e}", flush=True)
                        continue
                    completed += 1
                    _track_done(prob_name, results, completed)
    except Exception as e:
        if verbose:
            print(f"\n⚠ Parallel pool failed ({e}). Falling back to serial.")
        remaining = [p for p in problems if p not in all_results]
        serial_results = run_serial(
            remaining, n_runs, pop_size, n_gen, seed_base,
            checkpoint_path, verbose, monitor,
        )
        all_results.update(serial_results)

    if verbose:
        print(f"\nAll done in {(time.time() - t0) / 60:.1f} min")
    return all_results


# ============================================================================
# PART 10: STATISTICS
# ============================================================================

def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    count = 0
    for xi in x:
        count += np.sum(xi > y) - np.sum(xi < y)
    return count / (nx * ny)


def cohens_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_var = ((nx - 1) * np.var(x, ddof=1) +
                  (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    if pooled_var <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled_var))


def bootstrap_ci(data, n_samples=5000, confidence=0.95):
    data = np.asarray(data)
    if len(data) < 2:
        return (float(data[0]) if len(data) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(42)
    n = len(data)
    idx = rng.integers(0, n, size=(n_samples, n))
    means = data[idx].mean(axis=1)
    lo = np.percentile(means, (1 - confidence) / 2 * 100)
    hi = np.percentile(means, (1 + confidence) / 2 * 100)
    return float(np.mean(data)), float(lo), float(hi)


def run_statistical_tests(all_results, problems, algo_names,
                          reference='Aα-NSGA-II (adaptive mutation)',
                          internal_names=None, external_names=None,
                          label=''):
    print("\n" + "=" * 90)
    print(f"STATISTICAL ANALYSIS {label}")
    print("=" * 90)
    if internal_names is None:
        internal_names = INTERNAL_ALGO_NAMES
    if external_names is None:
        external_names = EXTERNAL_ALGO_NAMES

    agg_rank = {a: [] for a in algo_names}
    agg_hv = {a: [] for a in algo_names}
    agg_igd = {a: [] for a in algo_names}
    agg_gd = {a: [] for a in algo_names}
    agg_igd_n = {a: [] for a in algo_names}
    agg_gd_n = {a: [] for a in algo_names}
    for prob_name in problems:
        if prob_name not in all_results:
            continue
        res = all_results[prob_name]
        if '_error' in res:
            continue
        vals = [(a, res[a]['hv_raw_mean']) for a in algo_names if a in res]
        if not vals:
            continue
        vals.sort(key=lambda x: -x[1])
        for r, (a, _) in enumerate(vals, 1):
            agg_rank[a].append(r)
        best_hv = vals[0][1]
        if best_hv > 0:
            for a, hv in vals:
                agg_hv[a].append(hv / best_hv)
        for a in algo_names:
            if a not in res:
                continue
            igd = res[a].get('igd_mean', np.nan)
            gd = res[a].get('gd_mean', np.nan)
            igd_n = res[a].get('igd_norm_mean', np.nan)
            gd_n = res[a].get('gd_norm_mean', np.nan)
            if not np.isnan(igd):
                agg_igd[a].append(igd)
            if not np.isnan(gd):
                agg_gd[a].append(gd)
            if not np.isnan(igd_n):
                agg_igd_n[a].append(igd_n)
            if not np.isnan(gd_n):
                agg_gd_n[a].append(gd_n)

    print(f"\n--- Shapiro-Wilk ---")
    for algo in algo_names:
        if len(agg_rank[algo]) < 3:
            continue
        W, p = stats.shapiro(agg_rank[algo])
        print(f"{algo:<38} W={W:.4f}  p={p:.4e}")

    print(f"\n--- Friedman Test ---")
    try:
        sets = [agg_rank[a] for a in algo_names if len(agg_rank[a]) > 0]
        if len(sets) >= 3 and all(len(s) == len(sets[0]) for s in sets):
            stat, p = stats.friedmanchisquare(*sets)
            print(f"Chi-square = {stat:.4f}, p = {p:.6e}")
    except Exception as e:
        print(f"Friedman failed: {e}")

    print(f"\n--- Kruskal-Wallis ---")
    try:
        groups = [agg_hv[a] for a in algo_names if len(agg_hv[a]) > 0]
        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            print(f"H = {stat:.4f}, p = {p:.6e}")
    except Exception as e:
        print(f"Kruskal-Wallis failed: {e}")

    pairwise_results = {}
    for family_label, family_names in (
            ('INTERNAL ABLATION', internal_names),
            (f'EXTERNAL ({label})', external_names)):
        print(f"\n--- Pairwise Wilcoxon vs. reference ({family_label}) ---")
        n_comparisons = max(1, len(family_names) - 1)
        threshold = 0.05 / n_comparisons
        if reference not in agg_rank or len(agg_rank[reference]) < 2:
            continue
        ref_data = np.array(agg_rank[reference])
        for algo in family_names:
            if algo == reference or len(agg_rank[algo]) < 2:
                continue
            if len(agg_rank[algo]) != len(ref_data):
                continue
            try:
                _, p = stats.wilcoxon(ref_data, np.array(agg_rank[algo]))
            except Exception:
                p = np.nan
            d = cohens_d(ref_data, np.array(agg_rank[algo]))
            cl = cliffs_delta(ref_data, np.array(agg_rank[algo]))
            sig = "YES" if (not np.isnan(p) and p < threshold) else "no"
            pairwise_results[algo] = {
                'family': family_label,
                'p': p, 'cohens_d': d, 'cliffs_delta': cl,
                'significant': sig,
            }
            print(f"{algo:<38} p={p:.3e}  d={d:+.3f}  delta={cl:+.3f}  {sig}")
        print(f"Bonferroni threshold ({family_label}): {threshold:.6f}")

    print(f"\n--- Bootstrap 95% CIs (HV) ---")
    ci_results = {}
    for algo in algo_names:
        if len(agg_hv[algo]) < 2:
            continue
        mean, lo, hi = bootstrap_ci(agg_hv[algo], CONFIG['bootstrap_samples'])
        ci_results[algo] = (mean, lo, hi)
        print(f"{algo:<38} {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    print(f"\n--- Mean IGD / GD (raw, lower is better) ---")
    igd_summary, gd_summary = {}, {}
    igd_norm_summary, gd_norm_summary = {}, {}
    for algo in algo_names:
        if agg_igd[algo]:
            igd_summary[algo] = float(np.mean(agg_igd[algo]))
        if agg_gd[algo]:
            gd_summary[algo] = float(np.mean(agg_gd[algo]))
        if agg_igd_n[algo]:
            igd_norm_summary[algo] = float(np.mean(agg_igd_n[algo]))
        if agg_gd_n[algo]:
            gd_norm_summary[algo] = float(np.mean(agg_gd_n[algo]))
        if algo in igd_summary:
            print(f"{algo:<38} IGD={igd_summary[algo]:.4f}  "
                  f"GD={gd_summary.get(algo, float('nan')):.4f}  "
                  f"IGD_n={igd_norm_summary.get(algo, float('nan')):.4f}")
    if not igd_summary:
        print("  (no IGD/GD values available — did "
              "compute_igd_gd_pass run?)")

    print(f"\n--- Mean Ranks ---")
    for algo in algo_names:
        if agg_rank[algo]:
            print(f"{algo:<38} {np.mean(agg_rank[algo]):.2f}  "
                  f"std={np.std(agg_rank[algo]):.2f}")

    return {
        'agg_rank': agg_rank, 'agg_hv': agg_hv,
        'agg_igd': agg_igd, 'agg_gd': agg_gd,
        'agg_igd_norm': agg_igd_n, 'agg_gd_norm': agg_gd_n,
        'igd_summary': igd_summary, 'gd_summary': gd_summary,
        'igd_norm_summary': igd_norm_summary,
        'gd_norm_summary': gd_norm_summary,
        'pairwise': pairwise_results,
        'bootstrap_cis': ci_results,
    }


def report_framing(stats_results, algo_names):
    print("\n" + "=" * 90)
    print("FRAMING REPORT — What the data actually supports")
    print("=" * 90)

    ref = 'Aα-NSGA-II (adaptive mutation)'
    full_hv = np.mean(stats_results['agg_hv'].get(ref, [np.nan]))
    full_rank = np.mean(stats_results['agg_rank'].get(ref, [np.nan]))

    print(f"\nReference (full mechanism): {ref}")
    print(f"  Normalized HV = {full_hv:.4f}")
    print(f"  Mean rank      = {full_rank:.2f}")

    print("\nComponent ablations (all vs. reference):")
    for algo in algo_names:
        if algo == ref or not stats_results['agg_hv'].get(algo):
            continue
        hv = np.mean(stats_results['agg_hv'][algo])
        rank = np.mean(stats_results['agg_rank'][algo])
        pw = stats_results['pairwise'].get(algo, {})
        sig = pw.get('significant', 'n/a')
        p = pw.get('p', float('nan'))
        print(f"  {algo:<38} HV={hv:.4f}  rank={rank:5.2f}  "
              f"p={p:.2e}  sig={sig}")

    tied = []
    for algo in algo_names:
        if algo == ref:
            continue
        pw = stats_results['pairwise'].get(algo, {})
        if pw.get('significant') == 'no':
            tied.append(algo)
    print("\n--- Recommended claim language ---")
    if tied:
        print("The following algorithms are STATISTICALLY TIED with the")
        print("reference under Bonferroni-corrected Wilcoxon:")
        for a in tied:
            print(f"  - {a}")
        print("\nRecommendation: the contribution is the COMBINATION of")
        print("adaptive-α and adaptive-mutation, not adaptive-α alone.")
    else:
        print("The reference is statistically significantly better than all")
        print("ablations. The contribution can be attributed to the full")
        print("combined mechanism.")
    print("=" * 90)


# ============================================================================
# PART 11: MAIN VISUALIZATIONS
# ============================================================================

def _valid_results(all_results):
    return {k: v for k, v in all_results.items() if '_error' not in v}


def plot_per_family_heatmap(all_results, problems, algo_names, families, save_dir):
    print("  Generating per-family heatmap...")
    all_results = _valid_results(all_results)
    prob_order = []
    family_bounds = []
    for family, probs in families.items():
        probs_in = [p for p in probs if p in all_results]
        if not probs_in:
            continue
        family_bounds.append((family, len(prob_order),
                              len(prob_order) + len(probs_in)))
        prob_order.extend(probs_in)

    matrix = np.zeros((len(prob_order), len(algo_names)))
    for i, prob_name in enumerate(prob_order):
        res = all_results[prob_name]
        best_hv = max(res[a]['hv_raw_mean'] for a in algo_names
                      if a in res and res[a]['hv_raw_mean'] > 0)
        if best_hv <= 0:
            continue
        for j, algo in enumerate(algo_names):
            if algo in res:
                matrix[i, j] = res[algo]['hv_raw_mean'] / best_hv

    fig, ax = plt.subplots(figsize=(14, max(8, len(prob_order) * 0.25)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.05)
    ax.set_xticks(np.arange(len(algo_names)))
    ax.set_xticklabels(algo_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(prob_order)))
    ax.set_yticklabels(prob_order, fontsize=7)
    for family, start, end in family_bounds:
        ax.axhline(y=start - 0.5, color='black', linewidth=1.5)
        ax.axhline(y=end - 0.5, color='black', linewidth=1.5)
        mid = (start + end) / 2 - 0.5
        ax.text(-0.6, mid, family, fontsize=11, fontweight='bold',
                rotation=90, va='center', ha='center',
                transform=ax.get_yaxis_transform())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=6, color=color)
    ax.set_title(f'Normalized α-weighted HV (α = {FAIR_ALPHA}) per Problem',
                 fontsize=13, pad=15)
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    path = os.path.join(save_dir, 'heatmap_per_problem.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_per_family_bars(all_results, problems, algo_names, families, save_dir):
    print("  Generating per-family bar chart...")
    all_results = _valid_results(all_results)
    families_present = []
    for family, probs in families.items():
        probs_in = [p for p in probs if p in all_results]
        if probs_in:
            families_present.append((family, probs_in))
    if not families_present:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    n_fam = len(families_present)
    n_algos = len(algo_names)
    width = 0.8 / n_algos
    x = np.arange(n_fam)
    for j, algo in enumerate(algo_names):
        means = []
        for family, probs in families_present:
            vals = []
            for prob_name in probs:
                res = all_results[prob_name]
                best_hv = max(res[a]['hv_raw_mean'] for a in algo_names
                              if a in res and res[a]['hv_raw_mean'] > 0)
                if best_hv > 0 and algo in res:
                    vals.append(res[algo]['hv_raw_mean'] / best_hv)
            means.append(np.mean(vals) if vals else 0.0)
        ax.bar(x + j * width - (n_algos - 1) * width / 2,
               means, width, label=algo,
               color=ALGO_COLORS.get(algo, None),
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f[0] for f in families_present], fontsize=10)
    ax.set_ylabel('Mean Normalized α-weighted HV', fontsize=11)
    ax.set_title(f'Performance by Family (α = {FAIR_ALPHA})', fontsize=13)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    path = os.path.join(save_dir, 'family_bars.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_rank_distribution(all_results, problems, algo_names, save_dir):
    print("  Generating rank distribution...")
    all_results = _valid_results(all_results)
    ranks = {a: [] for a in algo_names}
    for prob_name in problems:
        if prob_name not in all_results:
            continue
        res = all_results[prob_name]
        vals = [(a, res[a]['hv_raw_mean']) for a in algo_names if a in res]
        vals.sort(key=lambda x: -x[1])
        for r, (a, _) in enumerate(vals, 1):
            ranks[a].append(r)
    if not any(ranks[a] for a in algo_names):
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(algo_names))
    bp = ax.boxplot([ranks[a] for a in algo_names],
                    positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(ALGO_COLORS.get(algo_names[i], '#cccccc'))
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(algo_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Rank (1 = best)', fontsize=11)
    ax.set_title('Algorithm Rank Distribution', fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'rank_distribution.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_alpha_analysis(all_results, problems, algo_names, save_dir):
    print("  Generating alpha analysis...")
    all_results = _valid_results(all_results)
    alpha_by_problem = []
    family_labels = []
    for family, probs in FAMILIES.items():
        for prob_name in probs:
            if (prob_name in all_results and
                    'Aα-NSGA-II (adaptive mutation)' in all_results[prob_name]):
                a = all_results[prob_name]['Aα-NSGA-II (adaptive mutation)']['alpha_mean']
                if not np.isnan(a):
                    alpha_by_problem.append(a)
                    family_labels.append(family)
    if not alpha_by_problem:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(alpha_by_problem, bins=15, color='steelblue',
            edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(alpha_by_problem), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {np.mean(alpha_by_problem):.3f}')
    ax.axvline(0.75, color='green', linestyle=':', linewidth=2,
               label='Fixed α = 0.75')
    ax.set_xlabel('Converged α', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Converged α of Aα-NSGA-II (adaptive mutation)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax = axes[1]
    family_alphas = {}
    for a, f in zip(alpha_by_problem, family_labels):
        family_alphas.setdefault(f, []).append(a)
    families_list = list(family_alphas.keys())
    positions = np.arange(len(families_list))
    bp = ax.boxplot([family_alphas[f] for f in families_list],
                    positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.axhline(0.75, color='green', linestyle=':', linewidth=2,
               label='Fixed α = 0.75')
    ax.set_xticks(positions)
    ax.set_xticklabels(families_list, fontsize=10)
    ax.set_ylabel('Converged α', fontsize=11)
    ax.set_title('Converged α by Family', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'alpha_analysis.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_scalability(all_results, problems, algo_names, save_dir):
    all_results = _valid_results(all_results)
    scal_probs = [p for p in problems if p.startswith('FScal')]
    if not scal_probs:
        return
    print("  Generating scalability plot...")
    sizes = []
    for p in scal_probs:
        try:
            s = int(p.replace('FScal', ''))
        except ValueError:
            s = 500
        sizes.append(s)
    order = np.argsort(sizes)
    sizes = [sizes[i] for i in order]
    scal_probs = [scal_probs[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False
    for algo in algo_names:
        hvs = []
        for prob_name in scal_probs:
            if prob_name not in all_results:
                hvs.append(np.nan)
                continue
            if algo in all_results[prob_name]:
                best = max(all_results[prob_name][a]['hv_raw_mean']
                           for a in algo_names
                           if a in all_results[prob_name]
                           and all_results[prob_name][a]['hv_raw_mean'] > 0)
                hvs.append(all_results[prob_name][algo]['hv_raw_mean'] / best
                           if best > 0 else 0)
            else:
                hvs.append(np.nan)
        if not np.all(np.isnan(hvs)):
            ax.plot(sizes, hvs, 'o-', label=algo,
                    color=ALGO_COLORS.get(algo, None), linewidth=1.5)
            any_plotted = True
    if not any_plotted:
        plt.close()
        return
    ax.set_xlabel('Number of Variables', fontsize=11)
    ax.set_ylabel('Normalized α-weighted HV', fontsize=11)
    ax.set_title('Scalability: Hypervolume', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    path = os.path.join(save_dir, 'scalability.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_bootstrap_cis(ci_results, algo_names, save_dir):
    if not ci_results:
        return
    print("  Generating bootstrap CI plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algo_names):
        if algo not in ci_results:
            continue
        mean, lo, hi = ci_results[algo]
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]],
                    fmt='o', color=ALGO_COLORS.get(algo, 'black'),
                    capsize=6, capthick=2, markersize=10)
    ax.set_yticks(np.arange(len(algo_names)))
    ax.set_yticklabels(algo_names, fontsize=9)
    ax.set_xlabel('Mean Normalized α-weighted HV (95% Bootstrap CI)',
                  fontsize=11)
    ax.set_title('Bootstrap 95% Confidence Intervals', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bootstrap_cis.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_igd_gd_bars(stats_results, algo_names, save_dir):
    igd = stats_results.get('igd_summary', {})
    gd = stats_results.get('gd_summary', {})
    igd_n = stats_results.get('igd_norm_summary', {})
    gd_n = stats_results.get('gd_norm_summary', {})
    if not igd and not gd:
        print("  (skipping IGD/GD bars — no values)")
        return
    print("  Generating IGD/GD bar chart...")
    algos = [a for a in algo_names if a in igd or a in gd]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, source, title in (
            (axes[0, 0], igd, 'IGD (raw)'),
            (axes[0, 1], gd, 'GD (raw)'),
            (axes[1, 0], igd_n, 'IGD (normalized)'),
            (axes[1, 1], gd_n, 'GD (normalized)')):
        vals = [source.get(a, np.nan) for a in algos]
        ypos = np.arange(len(algos))
        colors = [ALGO_COLORS.get(a, '#cccccc') for a in algos]
        ax.barh(ypos, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels(algos, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(save_dir, 'igd_gd.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_alpha_adaptation_effect(all_results, problems, save_dir):
    pairs = [('SPEA2', 'SPEA2+α'),
             ('MOEA/D', 'MOEA/D+α'),
             ('NSGA-III', 'NSGA-III+α'),
             ('cMLSGA', 'cMLSGA+α')]
    probs = [p for p in problems
             if p in all_results and '_error' not in all_results[p]]
    if not probs:
        return
    print("  Generating α-adaptation effect plot...")
    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(pairs)
    w = 0.8 / n
    x = np.arange(len(probs))
    for j, (strict, fair) in enumerate(pairs):
        vals = []
        for p in probs:
            r = all_results[p]
            if (strict in r and fair in r
                    and r[strict].get('hv_raw_mean', 0) > 0):
                vals.append(r[fair]['hv_raw_mean'] /
                            r[strict]['hv_raw_mean'])
            else:
                vals.append(np.nan)
        ax.bar(x + j * w - (n - 1) * w / 2, vals, w,
               label=f'{strict} → {fair}',
               color=ALGO_COLORS.get(fair, None),
               edgecolor='black', linewidth=0.4)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(probs, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('HV(fair) / HV(strict)   (>1 ⇒ α-adaptation helps)')
    ax.set_title('Effect of α-adaptation on each competitor algorithm',
                 fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'alpha_adaptation_effect.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def generate_all_visualizations(all_results, problems, algo_names, families,
                                ci_results, save_dir, stats_results=None):
    print("\n" + "=" * 78)
    print("GENERATING MAIN VISUALIZATIONS")
    print("=" * 78)
    os.makedirs(save_dir, exist_ok=True)
    plot_per_family_heatmap(all_results, problems, algo_names, families, save_dir)
    plot_per_family_bars(all_results, problems, algo_names, families, save_dir)
    plot_rank_distribution(all_results, problems, algo_names, save_dir)
    plot_alpha_analysis(all_results, problems, algo_names, save_dir)
    plot_scalability(all_results, problems, algo_names, save_dir)
    plot_bootstrap_cis(ci_results, algo_names, save_dir)
    plot_alpha_adaptation_effect(all_results, problems, save_dir)
    if stats_results is not None:
        plot_igd_gd_bars(stats_results, algo_names, save_dir)


# ============================================================================
# PART 11B: PARETO FRONT PLOTS
# ============================================================================

_CACHE_DIR_NAME = '_cache'


def _pareto_cache_path(save_dir, prob_name, tag, algo_name):
    cache = os.path.join(save_dir, _CACHE_DIR_NAME)
    safe = (algo_name.replace('α', 'alpha').replace('/', '_')
            .replace('\\', '_').replace(' ', '_'))
    return os.path.join(cache, f"{prob_name}__{tag}__{safe}.npy")


def _load_cached_front(save_dir, prob_name, tag, algo_name):
    p = _pareto_cache_path(save_dir, prob_name, tag, algo_name)
    if os.path.exists(p):
        try:
            return np.load(p, allow_pickle=False)
        except Exception:
            return None
    return None


def _save_cached_front(save_dir, prob_name, tag, algo_name, F):
    cache = os.path.join(save_dir, _CACHE_DIR_NAME)
    os.makedirs(cache, exist_ok=True)
    p = _pareto_cache_path(save_dir, prob_name, tag, algo_name)
    try:
        np.save(p, np.asarray(F, dtype=float))
    except Exception:
        pass


def _downsample_front(F, max_points=60):
    if len(F) <= max_points:
        return F
    order = np.argsort(F[:, 0])
    idx = np.linspace(0, len(F) - 1, max_points).astype(int)
    return F[order][idx]


def _plot_fronts_from_dict(prob_name, fronts, alpha, save_dir,
                          make_3d=True, make_2d_projections=True,
                          max_points=60, n_obj=None):
    if n_obj is None:
        if not fronts:
            return
        for F in fronts.values():
            if F.size:
                n_obj = F.shape[1]
                break
    if not n_obj or not fronts:
        return

    hero_name = 'Aα-NSGA-II (adaptive mutation)'

    def get_z_order(algo_name):
        if algo_name == hero_name: return 10
        if algo_name.startswith('SPEA2'): return 4
        if algo_name.startswith('MOEA/D'): return 3
        if algo_name.startswith('NSGA-III'): return 2
        if algo_name.startswith('cMLSGA'): return 1
        return 5

    sorted_algos = sorted(fronts.keys(), key=get_z_order)

    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        for algo_name in sorted_algos:
            F = fronts[algo_name]
            if len(F) == 0: continue
            Fd = _downsample_front(F, max_points)
            order = np.argsort(Fd[:, 0])
            color = ALGO_COLORS.get(algo_name, 'black')
            marker = ALGO_MARKERS.get(algo_name, 'o')
            z = get_z_order(algo_name)
            ax.plot(Fd[order, 0], Fd[order, 1], marker=marker, markersize=7,
                    label=f"{algo_name}", color=color, linewidth=1.5,
                    alpha=0.9, markeredgecolor='black', markeredgewidth=0.5,
                    zorder=z)
        ax.set_xlabel('\(f_1\) (α-weighted)', fontsize=12)
        ax.set_ylabel('\(f_2\) (α-weighted)', fontsize=12)
        ax.set_title(f'{prob_name} — Pareto Fronts (α={alpha})', fontsize=14)
        ax.legend(fontsize=10, framealpha=0.9, edgecolor='black')
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        path = os.path.join(save_dir, f'pareto_{prob_name}_2d.png')
        plt.savefig(path, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")
        return

    pairs = [(0, 1), (0, 2), (1, 2)]
    proj_titles = [('\(f_1\)', '\(f_2\)'), ('\(f_1\)', '\(f_3\)'), ('\(f_2\)', '\(f_3\)')]

    if make_2d_projections:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (i, j), (xl, yl) in zip(axes, pairs, proj_titles):
            for algo_name in sorted_algos:
                F = fronts[algo_name]
                if len(F) == 0: continue
                Fd = _downsample_front(F, max_points)
                color = ALGO_COLORS.get(algo_name, 'black')
                marker = ALGO_MARKERS.get(algo_name, 'o')
                z = get_z_order(algo_name)
                ax.scatter(Fd[:, i], Fd[:, j], s=45, color=color, marker=marker,
                           alpha=0.85, edgecolors='black', linewidths=0.6,
                           label=algo_name if i == 0 and j == 1 else None,
                           zorder=z)
            ax.set_xlabel(xl, fontsize=12)
            ax.set_ylabel(yl, fontsize=12)
            ax.grid(alpha=0.3, linestyle='--')
        axes[0].legend(fontsize=10, framealpha=0.9, edgecolor='black',
                       loc='best', markerscale=1.2)
        fig.suptitle(f'{prob_name} — 2D Projections (α={alpha})', fontsize=15, y=1.02)
        plt.tight_layout()
        path = os.path.join(save_dir, f'pareto_{prob_name}_2dproj.png')
        plt.savefig(path, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")

    if make_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for algo_name in sorted_algos:
            F = fronts[algo_name]
            if len(F) == 0: continue
            Fd = _downsample_front(F, max_points)
            color = ALGO_COLORS.get(algo_name, 'black')
            marker = ALGO_MARKERS.get(algo_name, 'o')
            z = get_z_order(algo_name)
            size = 45 if algo_name == hero_name else 35
            opac = 1.0 if algo_name == hero_name else 0.75
            ax.scatter(Fd[:, 0], Fd[:, 1], Fd[:, 2],
                       label=f"{algo_name} (|PF|={len(F)})",
                       s=size, color=color, marker=marker,
                       alpha=opac, edgecolors='black', linewidths=0.6,
                       zorder=z)
        ax.set_xlabel('$f_1$', fontsize=12, labelpad=10)
        ax.set_ylabel('$f_2$', fontsize=12, labelpad=10)
        ax.set_zlabel('$f_3$', fontsize=12, labelpad=10)
        ax.set_title(f'{prob_name} — Pareto Fronts (α={alpha})', fontsize=14)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9,
                  edgecolor='black', markerscale=1.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        path = os.path.join(save_dir, f'pareto_{prob_name}_3d.png')
        plt.savefig(path, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")


def _tag_algo_configs(ps, ng):
    all_cfg = get_algo_config(ps, ng, external_mode='both')
    internal_set = set(INTERNAL_ALGO_NAMES)
    fair_set = set(EXTERNAL_ALGO_NAMES)
    strict_set = set(EXTERNAL_ALGO_NAMES_STRICT)
    return {
        'ablation': [c for c in all_cfg if c[0] in internal_set],
        'fair':     [c for c in all_cfg
                     if c[0] in ({'Aα-NSGA-II (adaptive mutation)'}
                                 | fair_set)],
        'strict':   [c for c in all_cfg
                     if c[0] in ({'Aα-NSGA-II (adaptive mutation)'}
                                 | strict_set)],
    }


def generate_all_pareto_fronts(problems, save_dir, alpha=FAIR_ALPHA,
                                quick=False, algo_subset='all',
                                force_recompute=False):
    print("\n" + "=" * 78)
    print(f"GENERATING PARETO FRONT PLOTS (fixed α = {alpha})")
    print(f"  quick={quick}  algo_subset={algo_subset}  "
          f"force_recompute={force_recompute}")
    print("=" * 78)
    os.makedirs(save_dir, exist_ok=True)
    ps, ng = (40, 150) if quick else (60, 300)

    tags_cfg = _tag_algo_configs(ps, ng)
    if algo_subset == 'all':
        tags = ['ablation', 'fair', 'strict']
    elif algo_subset in tags_cfg:
        tags = [algo_subset]
    else:
        raise ValueError(
            f"algo_subset must be one of 'all', 'ablation', 'fair', "
            f"'strict' — got {algo_subset!r}"
        )

    for tag in tags:
        cfg = tags_cfg[tag]
        out_dir = os.path.join(save_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        for prob_idx, prob_name in enumerate(problems, 1):
            try:
                benchmark = get_extended_benchmark(prob_name)
            except Exception as e:
                print(f"  [{prob_idx}/{len(problems)}] {prob_name}: "
                      f"get_benchmark failed: {e}")
                continue

            n_obj = benchmark.n_obj
            fronts = {}
            print(f"  [{prob_idx}/{len(problems)}] {prob_name} "
                  f"({tag}, n_obj={n_obj}) ...", flush=True)

            for algo_name, algo_cls, kwargs in cfg:
                F = None
                if not force_recompute:
                    F = _load_cached_front(save_dir, prob_name, tag,
                                           algo_name)
                if F is None:
                    try:
                        algo = algo_cls(benchmark, seed=2026, **kwargs)
                        pop = algo.run()
                        F = _fixed_alpha_objectives(pop, alpha=alpha,
                                                    force=False)
                        _save_cached_front(save_dir, prob_name, tag,
                                           algo_name, F)
                        print(f"      {algo_name}: |PF|={len(F)}",
                              flush=True)
                        del algo, pop
                    except Exception as e:
                        print(f"      {algo_name}: FAILED "
                              f"({type(e).__name__}: {e})", flush=True)
                        traceback.print_exc()
                        F = None
                else:
                    print(f"      {algo_name}: cached |PF|={len(F)}",
                          flush=True)
                if F is not None and F.size > 0:
                    fronts[algo_name] = F

            if not fronts:
                print(f"      (no fronts; skipping plot)", flush=True)
                continue

            try:
                _plot_fronts_from_dict(
                    prob_name, fronts, alpha, out_dir,
                    make_3d=True,
                    make_2d_projections=True,
                    max_points=40,
                    n_obj=n_obj,
                )
            except Exception as e:
                print(f"      plot failed for {prob_name} ({tag}): "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            finally:
                del fronts
                gc.collect()


# ============================================================================
# PART 11C: Σ̂ TABLE
# ============================================================================

def generate_sigma_hat_table(all_results, problems, save_dir,
                             n_runs_per_alpha=SIGMA_HAT_RUNS_PER_ALPHA):
    print("\n" + "=" * 78)
    print("COMPUTING α-SENSITIVITY PROXY Σ̂  (same population per α)")
    print("=" * 78)
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for prob_name in problems:
        try:
            sigma, hvs = compute_sigma_hat(
                prob_name,
                alphas=SIGMA_HAT_ALPHAS,
                seed=CONFIG['sigma_hat_seed'],
                pop_size=CONFIG['pop_size'],
                n_gen=CONFIG['n_gen'],
                n_runs_per_alpha=n_runs_per_alpha,
            )
            rows.append({
                'problem': prob_name,
                'family': 'F2' if prob_name.startswith('F2') else 'F1',
                'sigma_hat': sigma,
                'hv_0.50': hvs[0],
                'hv_0.75': hvs[1],
                'hv_1.00': hvs[2],
            })
            print(f"  {prob_name:<14}  Σ̂ = {sigma:.4f}  "
                  f"HV(0.5)={hvs[0]:.3f}  HV(0.75)={hvs[1]:.3f}  "
                  f"HV(1.0)={hvs[2]:.3f}")
        except Exception as e:
            print(f"  ✗ {prob_name}: {type(e).__name__}: {e}")

    if not rows:
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, 'sigma_hat.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n  CSV saved to {csv_path}")

    df = df.sort_values(['family', 'problem']).reset_index(drop=True)

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Empirical α-sensitivity proxy $\widehat{\Sigma}$ '
        r'per problem. $\widehat{\Sigma}$ is the normalized spread of the '
        r'fixed-$\alpha$ hypervolume across $\alpha \in \{0.5, 0.75, 1.0\}$, '
        r'averaged over '
        f'{n_runs_per_alpha} seeds and computed against a '
        r'single reference point (union of all fronts). '
        r'Each population is evaluated under all three $\alpha$ values '
        r'to isolate sensitivity from cross-seed variability.}',
        r'\label{tab:sigma_hat}',
        r'\small',
        r'\begin{tabular}{@{}llccc@{}}', r'\toprule',
        r'Problem & Family & $\widehat{\Sigma}$ & '
        r'$\overline{\mathrm{HV}}(0.5)$ & '
        r'$\overline{\mathrm{HV}}(1.0)$ \\',
        r'\midrule',
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['problem']} & {row['family']} & "
            f"{row['sigma_hat']:.4f} & {row['hv_0.50']:.3f} & "
            f"{row['hv_1.00']:.3f} \\\\"
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    tex_path = os.path.join(save_dir, 'table_sigma_hat.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX table saved to {tex_path}")

    verify_sigma_hat_families(save_dir)


# ============================================================================
# PART 12: LATEX TABLES
# ============================================================================

def _build_external_table(all_results, problems, algo_names,
                          stats_results, save_dir, filename, caption, label):
    norm = {a: [] for a in algo_names}
    ranks = {a: [] for a in algo_names}
    igds = {a: [] for a in algo_names}
    gds = {a: [] for a in algo_names}
    igds_n = {a: [] for a in algo_names}
    gds_n = {a: [] for a in algo_names}
    for prob_name in problems:
        if prob_name not in all_results:
            continue
        res = all_results[prob_name]
        vals = [(a, res[a]['hv_raw_mean']) for a in algo_names if a in res]
        if not vals:
            continue
        vals.sort(key=lambda x: -x[1])
        best_hv = vals[0][1]
        for r, (a, hv) in enumerate(vals, 1):
            ranks[a].append(r)
            if best_hv > 0:
                norm[a].append(hv / best_hv)
        for a in algo_names:
            if a not in res:
                continue
            igd = res[a].get('igd_mean', np.nan)
            gd = res[a].get('gd_mean', np.nan)
            igd_n = res[a].get('igd_norm_mean', np.nan)
            gd_n = res[a].get('gd_norm_mean', np.nan)
            if not np.isnan(igd):
                igds[a].append(igd)
            if not np.isnan(gd):
                gds[a].append(gd)
            if not np.isnan(igd_n):
                igds_n[a].append(igd_n)
            if not np.isnan(gd_n):
                gds_n[a].append(gd_n)

    lines = [
        r'\begin{table*}[htbp]', r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{@{}lccccccc@{}}', r'\toprule',
        r'Algorithm & Norm.\ HV & Std. & Rank & Wins & IGD & GD & '
        r'IGD$_n$ \\',
        r'\midrule',
    ]
    for algo in algo_names:
        if not norm[algo]:
            continue
        mean_hv = np.mean(norm[algo])
        std_hv = np.std(norm[algo])
        mean_rank = np.mean(ranks[algo])
        wins = sum(1 for i in range(len(norm[algo])) if ranks[algo][i] == 1)
        igd_s = (f'{np.mean(igds[algo]):.4f}' if igds[algo] else '--')
        gd_s = (f'{np.mean(gds[algo]):.4f}' if gds[algo] else '--')
        igd_n_s = (f'{np.mean(igds_n[algo]):.4f}' if igds_n[algo] else '--')
        lines.append(f'{algo} & {mean_hv:.4f} & {std_hv:.4f} & '
                     f'{mean_rank:.2f} & {wins} & {igd_s} & {gd_s} & '
                     f'{igd_n_s} \\\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    with open(os.path.join(save_dir, filename), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_latex_tables(all_results, problems, algo_names, families,
                          stats_strict, stats_fair, save_dir):
    print("\n" + "=" * 78)
    print("GENERATING MAIN LATEX TABLES")
    print("=" * 78)
    os.makedirs(save_dir, exist_ok=True)
    all_results = _valid_results(all_results)

    strict_algos = ['Aα-NSGA-II (adaptive mutation)'] + EXTERNAL_ALGO_NAMES_STRICT
    _build_external_table(
        all_results, problems, strict_algos, stats_strict, save_dir,
        filename='table_external_strict.tex',
        caption=(
            r'External comparison against standard state-of-the-art '
            r'algorithms with fixed $\alpha = 0.75$. This is the primary '
            r'external comparison; it tests whether adaptive-$\alpha$ is '
            r'valuable. IGD/GD are computed against a memory-safe, '
            r'per-algorithm-balanced empirical reference front; lower is '
            r'better. IGD$_n$ is IGD divided by the reference front '
            r'bounding-box diagonal for cross-problem comparability.'
        ),
        label='tab:external_strict',
    )

    fair_algos = ['Aα-NSGA-II (adaptive mutation)'] + EXTERNAL_ALGO_NAMES
    _build_external_table(
        all_results, problems, fair_algos, stats_fair, save_dir,
        filename='table_external_fair.tex',
        caption=(
            r'Ablation: competitors are $\alpha$-wrapped so they receive the '
            r'same $\alpha$-adaptation capability as A$\alpha$-NSGA-II. This '
            r'isolates the contribution of the selection and mutation '
            r'mechanisms when the $\alpha$ advantage is neutralized.'
        ),
        label='tab:external_fair',
    )

    lines = [
        r'\begin{table*}[htbp]', r'\centering',
        r'\caption{Internal ablation: overall performance on all problems. '
        r'Normalized α-weighted HV (fixed $\alpha = 0.75$) and raw IGD/GD.}',
        r'\label{tab:overall_internal}',
        r'\begin{tabular}{@{}lccccccc@{}}', r'\toprule',
        r'Algorithm & Norm.\ HV & Std. & Rank & Wins & IGD & GD & '
        r'IGD$_n$ \\',
        r'\midrule',
    ]
    for algo in INTERNAL_ALGO_NAMES:
        hv_all = []
        ranks_all = []
        igd_all = []
        gd_all = []
        igd_n_all = []
        for prob_name in problems:
            if prob_name not in all_results:
                continue
            res = all_results[prob_name]
            vals = [(a, res[a]['hv_raw_mean']) for a in INTERNAL_ALGO_NAMES
                    if a in res]
            if not vals:
                continue
            vals.sort(key=lambda x: -x[1])
            best = vals[0][1]
            if best > 0 and algo in res:
                hv_all.append(res[algo]['hv_raw_mean'] / best)
            for r, (a, _) in enumerate(vals, 1):
                if a == algo:
                    ranks_all.append(r)
            if algo in res:
                if not np.isnan(res[algo].get('igd_mean', np.nan)):
                    igd_all.append(res[algo]['igd_mean'])
                if not np.isnan(res[algo].get('gd_mean', np.nan)):
                    gd_all.append(res[algo]['gd_mean'])
                if not np.isnan(res[algo].get('igd_norm_mean', np.nan)):
                    igd_n_all.append(res[algo]['igd_norm_mean'])
        if not hv_all:
            continue
        lines.append(
            f'{algo} & {np.mean(hv_all):.4f} & {np.std(hv_all):.4f} & '
            f'{np.mean(ranks_all):.2f} & '
            f'{sum(1 for r in ranks_all if r == 1)} & '
            f'{(np.mean(igd_all) if igd_all else float("nan")):.4f} & '
            f'{(np.mean(gd_all) if gd_all else float("nan")):.4f} & '
            f'{(np.mean(igd_n_all) if igd_n_all else float("nan")):.4f} \\\\'
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    with open(os.path.join(save_dir, 'table_overall.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))

    lines = [
        r'\begin{table*}[htbp]', r'\centering',
        r'\caption{Mean Normalized α-weighted HV by Family}',
        r'\label{tab:family}',
        r'\begin{tabular}{@{}l' + 'c' * len(algo_names) + r'@{}}',
        r'\toprule',
        r'Family & ' + ' & '.join(algo_names) + r' \\', r'\midrule',
    ]
    for family, probs in families.items():
        probs_in = [p for p in probs if p in all_results]
        if not probs_in:
            continue
        row = [family]
        for algo in algo_names:
            vals = []
            for prob_name in probs_in:
                res = all_results[prob_name]
                best_hv = max(res[a]['hv_raw_mean'] for a in algo_names
                              if a in res and res[a]['hv_raw_mean'] > 0)
                if best_hv > 0 and algo in res:
                    vals.append(res[algo]['hv_raw_mean'] / best_hv)
            row.append(f'{np.mean(vals):.3f}' if vals else '--')
        lines.append(' & '.join(row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    with open(os.path.join(save_dir, 'table_family.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))

    for tag, stats_res, ext_names in (
            ('strict', stats_strict, EXTERNAL_ALGO_NAMES_STRICT),
            ('fair', stats_fair, EXTERNAL_ALGO_NAMES)):
        lines = [
            r'\begin{table}[htbp]', r'\centering',
            r'\caption{Pairwise Wilcoxon (Bonferroni) on ranks — '
            + tag + r' external comparison.}',
            r'\label{tab:statistics_' + tag + '}',
            r'\begin{tabular}{@{}llcccc@{}}', r'\toprule',
            r'Family & Comparison & $p$ & Cohen $d$ & Cliff $\delta$ & Sig. \\',
            r'\midrule',
        ]
        ref = CONFIG['reference_algo']
        for algo, res in stats_res.get('pairwise', {}).items():
            if algo not in ext_names:
                continue
            lines.append(
                f"{res.get('family', '--')} & {ref} vs. {algo} & "
                f"{res['p']:.2e} & {res['cohens_d']:.3f} & "
                f"{res['cliffs_delta']:.3f} & {res['significant']} \\\\"
            )
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
        with open(os.path.join(save_dir,
                               f'table_statistics_{tag}.tex'), 'w',
                  encoding='utf-8') as f:
            f.write('\n'.join(lines))

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Bootstrap 95\% Confidence Intervals (Normalized HV)}',
        r'\label{tab:bootstrap}',
        r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
        r'Algorithm & Mean & CI Lower & CI Upper \\', r'\midrule',
    ]
    for algo in algo_names:
        if algo in stats_strict['bootstrap_cis']:
            mean, lo, hi = stats_strict['bootstrap_cis'][algo]
            lines.append(f'{algo} & {mean:.4f} & {lo:.4f} & {hi:.4f} \\\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    with open(os.path.join(save_dir, 'table_bootstrap.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Generated LaTeX tables in {save_dir}/")


def generate_per_problem_table(all_results, problems, algo_names, families,
                                save_dir):
    print("\n" + "=" * 78)
    print("GENERATING PER-PROBLEM TABLE")
    print("=" * 78)
    os.makedirs(save_dir, exist_ok=True)
    all_results = _valid_results(all_results)
    lines = [
        r'\begin{table*}[htbp]', r'\centering',
        r'\caption{Per-Problem Normalized α-weighted HV (best in bold)}',
        r'\label{tab:per_problem}',
        r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{@{}l' + 'c' * len(algo_names) + r'@{}}',
        r'\toprule',
        r'\textbf{Problem} & ' + ' & '.join([f'\\textbf{{{a}}}'
                                              for a in algo_names]) + r' \\',
        r'\midrule',
    ]
    family_rows = {}
    for family, probs in families.items():
        probs_in = [p for p in probs if p in all_results]
        if not probs_in:
            continue
        family_rows[family] = []
        for prob_name in probs_in:
            res = all_results[prob_name]
            best_hv = max(res[a]['hv_raw_mean'] for a in algo_names
                          if a in res and res[a]['hv_raw_mean'] > 0)
            if best_hv <= 0:
                continue
            row_vals = []
            for algo in algo_names:
                if algo in res:
                    row_vals.append((algo, res[algo]['hv_raw_mean'] / best_hv))
                else:
                    row_vals.append((algo, np.nan))
            valid = [(a, v) for a, v in row_vals if not np.isnan(v)]
            best_val = max(v for _, v in valid) if valid else 0
            family_rows[family].append((prob_name, row_vals, best_val))
    for family, rows in family_rows.items():
        lines.append(f'\\multicolumn{{{len(algo_names) + 1}}}{{l}}'
                     f'{{\\textit{{{family}}}}} \\\\')
        lines.append(r'\midrule')
        for prob_name, row_vals, best_val in rows:
            cells = []
            for algo, val in row_vals:
                if np.isnan(val):
                    cells.append('--')
                elif abs(val - best_val) < 1e-6:
                    cells.append(f'\\textbf{{{val:.3f}}}')
                else:
                    cells.append(f'{val:.3f}')
            lines.append(f'{prob_name} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    if lines[-1] == r'\midrule':
        lines[-1] = r'\bottomrule'
    lines.extend([r'\end{tabular}', r'\end{table*}'])
    with open(os.path.join(save_dir, 'table_per_problem.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"    Saved: {os.path.join(save_dir, 'table_per_problem.tex')}")

    csv_rows = []
    for family, rows in family_rows.items():
        for prob_name, row_vals, best_val in rows:
            row = {'family': family, 'problem': prob_name}
            for algo, val in row_vals:
                row[algo] = val
            csv_rows.append(row)
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        df.to_csv(os.path.join(save_dir, 'per_problem_results.csv'),
                  index=False, encoding='utf-8')


# ============================================================================
# PART 13: SETTINGS AND MAIN
# ============================================================================

def apply_settings(mode=None, use_parallel=None, n_workers=None,
                   runs=None, pop=None, gen=None, quiet=None,
                   external_mode=None):
    if mode == 'test':
        CONFIG['problems'] = ['FZDT1', 'FZDT2', 'FDTLZ2',
                              'F2FZDT1', 'F2FUF1']
        CONFIG['n_runs'] = 2
        CONFIG['pop_size'] = 30
        CONFIG['n_gen'] = 50
        CONFIG['results_dir']    = os.path.join(OUTPUT_ROOT, 'test_results')
        CONFIG['figures_dir']    = os.path.join(OUTPUT_ROOT, 'test_figures')
        CONFIG['latex_dir']      = os.path.join(OUTPUT_ROOT, 'test_latex')
        CONFIG['supp_latex_dir'] = os.path.join(OUTPUT_ROOT,
                                                'test_supplementary_latex')
        CONFIG['sigma_hat_dir']  = os.path.join(OUTPUT_ROOT, 'test_sigma_hat')
        CONFIG['external_dir']   = os.path.join(OUTPUT_ROOT, 'test_external')
    else:
        CONFIG['results_dir']    = os.path.join(OUTPUT_ROOT, 'results')
        CONFIG['figures_dir']    = os.path.join(OUTPUT_ROOT, 'figures')
        CONFIG['latex_dir']      = os.path.join(OUTPUT_ROOT, 'latex')
        CONFIG['supp_latex_dir'] = os.path.join(OUTPUT_ROOT,
                                                'supplementary_latex')
        CONFIG['sigma_hat_dir']  = os.path.join(OUTPUT_ROOT, 'sigma_hat')
        CONFIG['external_dir']   = os.path.join(OUTPUT_ROOT, 'external')
    if runs is not None:
        CONFIG['n_runs'] = runs
    if pop is not None:
        CONFIG['pop_size'] = pop
    if gen is not None:
        CONFIG['n_gen'] = gen
    if use_parallel is not None:
        CONFIG['use_parallel'] = use_parallel
    if n_workers is not None:
        CONFIG['n_workers'] = n_workers
    if quiet is not None:
        CONFIG['quiet'] = quiet
    if external_mode is not None:
        global EXTERNAL_COMPARISON_MODE
        EXTERNAL_COMPARISON_MODE = external_mode
    if CONFIG['n_workers'] is None:
        CONFIG['n_workers'] = max(1, mp.cpu_count() // 2)


def run_pareto_only(problems=None, quick=False,
                    pop=None, gen=None,
                    algo_subset='all',
                    force_recompute=False):
    apply_settings(mode='full', pop=pop, gen=gen)
    if problems is None:
        problems = PARETO_DEFAULT_PROBLEMS
    ps = CONFIG['pop_size']
    ng = CONFIG['n_gen']
    if quick:
        ps, ng = 40, 150

    out = os.path.join(CONFIG['figures_dir'], 'pareto_fronts')
    os.makedirs(out, exist_ok=True)
    for tag in ('ablation', 'fair', 'strict'):
        os.makedirs(os.path.join(out, tag), exist_ok=True)

    print("=" * 78)
    print(f"PARETO-ONLY MODE (v8.8) — {len(problems)} problems, "
          f"pop={ps}, gen={ng}, α={FAIR_ALPHA}")
    print(f"  algo_subset   = {algo_subset}")
    print(f"  quick         = {quick}")
    print(f"  recompute     = {force_recompute}")
    print(f"Output: {out}")
    print("=" * 78)

    generate_all_pareto_fronts(
        problems, out, alpha=FAIR_ALPHA, quick=quick,
        algo_subset=algo_subset,
        force_recompute=force_recompute,
    )

    print("\nPareto-only mode complete.")


def main(mode=None, use_parallel=None, n_workers=None,
         runs=None, pop=None, gen=None, quiet=None,
         pareto_quick=True, compute_sigma=True,
         include_external=True, external_mode=None,
         skip_main_runs=False, skip_igd_gd=False,
         pareto_only=False, pareto_problems=None,
         pareto_algos='all', pareto_force=False):
    apply_settings(mode=mode, use_parallel=use_parallel, n_workers=n_workers,
                   runs=runs, pop=pop, gen=gen, quiet=quiet,
                   external_mode=external_mode)

    if pareto_only:
        return run_pareto_only(
            problems=pareto_problems,
            quick=pareto_quick,
            pop=pop, gen=gen,
            algo_subset=pareto_algos,
            force_recompute=pareto_force,
        )

    print("=" * 90)
    print("Aα-NSGA-II: COMPLETE PIPELINE (v8.8)")
    print(f"Problems: {len(CONFIG['problems'])} | "
          f"Runs: {CONFIG['n_runs']} | "
          f"Pop: {CONFIG['pop_size']} | Gen: {CONFIG['n_gen']} | "
          f"Workers: {CONFIG['n_workers']} | "
          f"Mode: {'parallel' if CONFIG['use_parallel'] else 'serial'}")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"Environment: {'Jupyter/Colab' if IN_JUPYTER else 'Terminal'}")
    print(f"External comparison mode: {EXTERNAL_COMPARISON_MODE}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Fair HV metric uses α = {FAIR_ALPHA}")
    print("=" * 90)

    for d in [CONFIG['results_dir'], CONFIG['figures_dir'],
              CONFIG['latex_dir'], CONFIG['supp_latex_dir'],
              CONFIG['sigma_hat_dir'], CONFIG['external_dir']]:
        os.makedirs(d, exist_ok=True)
        print(f"  Ensured: {d}")

    if compute_sigma:
        _probe_f2_override()

    _cfg_strict = get_algo_config(CONFIG['pop_size'], CONFIG['n_gen'],
                                   external_mode='strict')
    for name, _, kwargs in _cfg_strict:
        if name in EXTERNAL_ALGO_NAMES_STRICT:
            assert kwargs.get('alpha', FAIR_ALPHA) == FAIR_ALPHA, (
                f"Strict-mode {name} uses alpha={kwargs.get('alpha')}, "
                f"expected FAIR_ALPHA={FAIR_ALPHA}"
            )

    checkpoint_path = os.path.join(CONFIG['results_dir'], 'checkpoint.pkl')

    if not skip_main_runs:
        if CONFIG['use_parallel']:
            all_results = run_parallel(
                CONFIG['problems'], CONFIG['n_runs'], CONFIG['pop_size'],
                CONFIG['n_gen'], CONFIG['seed_base'],
                n_workers=CONFIG['n_workers'],
                checkpoint_path=checkpoint_path,
                verbose=not CONFIG['quiet'],
                monitor=None,
            )
        else:
            all_results = run_serial(
                CONFIG['problems'], CONFIG['n_runs'], CONFIG['pop_size'],
                CONFIG['n_gen'], CONFIG['seed_base'],
                checkpoint_path=checkpoint_path,
                verbose=True,
                monitor=None,
            )
    else:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                all_results = pickle.load(f)
            print(f"\n[skip_main_runs] Loaded {len(all_results)} problems "
                  f"from checkpoint")
        else:
            all_results = {}
            print("\n[skip_main_runs] No checkpoint found — nothing to do.")

    if all_results and not skip_igd_gd:
        all_results = compute_igd_gd_pass(
            all_results, CONFIG['problems'],
            CONFIG['pop_size'], CONFIG['n_gen'],
            CONFIG['seed_base'], CONFIG['n_runs'],
            verbose=True,
        )
        _save_checkpoint(all_results, checkpoint_path)
    elif skip_igd_gd:
        print("\n[skip_igd_gd] IGD/GD pass skipped.")

    if not skip_igd_gd and all_results:
        has_igd = any(
            not np.isnan(res[a].get('igd_mean', np.nan))
            for res in all_results.values() if isinstance(res, dict) and '_error' not in res
            for a in res if not a.startswith('_')
        )
        if not has_igd:
            print("  ⚠ WARNING: No IGD/GD values found in results. LaTeX "
                  "tables will show '--' for IGD/GD columns. Re-run without "
                  "--skip-igd-gd to compute them.")

    stats_fair = run_statistical_tests(
        all_results, CONFIG['problems'], ALGO_NAMES,
        reference=CONFIG['reference_algo'],
        internal_names=INTERNAL_ALGO_NAMES,
        external_names=EXTERNAL_ALGO_NAMES if include_external else [],
        label='[FAIR = ABLATION]',
    )

    strict_algo_names = INTERNAL_ALGO_NAMES + EXTERNAL_ALGO_NAMES_STRICT
    stats_strict = run_statistical_tests(
        all_results, CONFIG['problems'], strict_algo_names,
        reference=CONFIG['reference_algo'],
        internal_names=INTERNAL_ALGO_NAMES,
        external_names=EXTERNAL_ALGO_NAMES_STRICT if include_external else [],
        label='[STRICT = PRIMARY]',
    )

    report_framing(stats_strict, strict_algo_names)

    generate_all_visualizations(
        all_results, CONFIG['problems'], ALGO_NAMES, CONFIG['families'],
        stats_strict['bootstrap_cis'], CONFIG['figures_dir'],
        stats_results=stats_strict,
    )

    generate_latex_tables(
        all_results, CONFIG['problems'], ALGO_NAMES, CONFIG['families'],
        stats_strict, stats_fair, CONFIG['latex_dir']
    )

    generate_per_problem_table(
        all_results, CONFIG['problems'], ALGO_NAMES, CONFIG['families'],
        CONFIG['supp_latex_dir']
    )

    if compute_sigma:
        try:
            generate_sigma_hat_table(
                all_results,
                CONFIG['sigma_hat_problems'],
                CONFIG['sigma_hat_dir'],
            )
        except Exception as e:
            print(f"  Σ̂ computation failed: {e}")
            traceback.print_exc()

    try:
        generate_all_pareto_fronts(
            CONFIG['problems'],
            os.path.join(CONFIG['figures_dir'], 'pareto_fronts'),
            alpha=FAIR_ALPHA,
            quick=pareto_quick,
        )
    except Exception as e:
        print(f"  Pareto front generation failed: {e}")
        traceback.print_exc()

    external_rows = []
    for prob_name, results in all_results.items():
        if '_error' in results:
            continue
        for algo in (EXTERNAL_ALGO_NAMES + EXTERNAL_ALGO_NAMES_STRICT):
            if algo in results:
                external_rows.append({
                    'problem': prob_name,
                    'algorithm': algo,
                    'hv_raw_mean': results[algo]['hv_raw_mean'],
                    'hv_raw_std': results[algo]['hv_raw_std'],
                    'igd_mean': results[algo].get('igd_mean', np.nan),
                    'gd_mean': results[algo].get('gd_mean', np.nan),
                    'igd_norm_mean': results[algo].get('igd_norm_mean', np.nan),
                    'gd_norm_mean': results[algo].get('gd_norm_mean', np.nan),
                })
    if external_rows:
        pd.DataFrame(external_rows).to_csv(
            os.path.join(CONFIG['external_dir'], 'external_results.csv'),
            index=False, encoding='utf-8')

    rows = []
    for prob_name, results in all_results.items():
        if '_error' in results:
            continue
        for algo, res in results.items():
            rows.append({
                'problem': prob_name, 'algorithm': algo,
                'hv_raw_mean': res['hv_raw_mean'],
                'hv_raw_std': res['hv_raw_std'],
                'alpha_mean': res.get('alpha_mean', np.nan),
                'igd_mean': res.get('igd_mean', np.nan),
                'gd_mean': res.get('gd_mean', np.nan),
                'igd_norm_mean': res.get('igd_norm_mean', np.nan),
                'gd_norm_mean': res.get('gd_norm_mean', np.nan),
            })
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(CONFIG['results_dir'], 'results.csv'),
            index=False, encoding='utf-8')

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETE")
    print("=" * 90)


def _cli_entry():
    try:
        mp.freeze_support()
    except Exception:
        pass
    import argparse
    parser = argparse.ArgumentParser(
        description='Aα-NSGA-II Pipeline v8.8')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no-pareto', action='store_true')
    parser.add_argument('--full-pareto', action='store_true')
    parser.add_argument('--no-sigma', action='store_true')
    parser.add_argument('--no-external', action='store_true')
    parser.add_argument('--no-igd-gd', action='store_true')
    parser.add_argument('--external-mode', choices=['fair', 'strict', 'both'],
                        default='both')
    parser.add_argument('--force-parallel', action='store_true')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--runs', type=int, default=None)
    parser.add_argument('--pop', type=int, default=None)
    parser.add_argument('--gen', type=int, default=None)
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--pareto-only', action='store_true',
                        help='Generate ONLY the Pareto front figures.')
    parser.add_argument('--pareto-problems', type=str, default=None,
                        help='Comma-separated problems for --pareto-only.')
    parser.add_argument('--pareto-algos',
                        choices=['all', 'ablation', 'fair', 'strict'],
                        default='all')
    parser.add_argument('--pareto-force', action='store_true')
    args, _ = parser.parse_known_args()

    if args.force_parallel:
        globals()['FORCE_PARALLEL_FROM_JUPYTER'] = True

    if args.pareto_only:
        probs = None
        if args.pareto_problems:
            if args.pareto_problems.strip().lower() == 'all':
                probs = PARETO_DEFAULT_PROBLEMS
            else:
                probs = [s.strip() for s in args.pareto_problems.split(',')
                         if s.strip()]
        run_pareto_only(
            problems=probs,
            quick=not args.full_pareto,
            pop=args.pop, gen=args.gen,
            algo_subset=args.pareto_algos,
            force_recompute=args.pareto_force,
        )
        return

    mode = 'test' if args.test else 'full'
    use_parallel = not args.serial

    if args.no_pareto:
        globals()['generate_all_pareto_fronts'] = lambda *a, **kw: None

    main(mode=mode, use_parallel=use_parallel,
         n_workers=args.workers, runs=args.runs,
         pop=args.pop, gen=args.gen,
         quiet=args.quiet,
         pareto_quick=not args.full_pareto,
         compute_sigma=not args.no_sigma,
         include_external=not args.no_external,
         external_mode=args.external_mode,
         skip_main_runs=args.resume,
         skip_igd_gd=args.no_igd_gd,
         pareto_only=False,
         pareto_problems=None,
         pareto_algos='all',
         pareto_force=False)


if __name__ == "__main__":
    _cli_entry()