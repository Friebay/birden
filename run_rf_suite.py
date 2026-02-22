#!/usr/bin/env python3
"""
RF/ET/Imbalanced-Ensembles experiment runner (HPC-friendly)

What this script does
- Loads a single classification CSV with feature columns like c110_*, d110_*, err_* / errors_*, snr_*, etc.
- Uses a split manifest (parquet) to obtain train/val/test indices for a chosen fold.
- Evaluates 4 tree-ensemble families with multiple hyperparameter samples:
    1) RandomForestClassifier
    2) ExtraTreesClassifier
    3) BalancedRandomForestClassifier (imblearn)
    4) BalancedBaggingClassifier (imblearn) with ExtraTree base estimator
- For each (feature_set, model_family, param_sample):
    runs N_BOOTSTRAP stratified bootstraps on TRAIN,
    predicts on VAL/TEST,
    picks thresholds on VAL (Youden, F1) and applies them to TEST,
    computes threshold metrics (Sensitivity, Specificity, Precision) and
    probability metrics (ROC AUC, PR AUC, Brier, LogLoss) on TEST.
- Writes RAW per-bootstrap rows (so nothing is lost) + a compact summary table.

Designed for SLURM array usage:
- create a task table (one row = one (feature_set, model, param_idx))
- run a block with --task-block START:COUNT (recommended)
- checkpointing: each task writes its own parquet; re-runs skip completed tasks.

Requirements
- numpy, pandas
- scikit-learn
- imbalanced-learn (for BRF and BalancedBagging)
- pyarrow (recommended) for parquet I/O

Notes
- No truncation experiment here (kept for a separate stage).
- Default bootstrap count is 50 (as requested).
"""

from __future__ import annotations


def _sanitize_params(params: dict) -> dict:
    """Cast numeric strings to numbers for sklearn param validation (e.g. '0.3' -> 0.3)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, str):
            s = v.strip()
            # int?
            if re.fullmatch(r"[+-]?\d+", s):
                try:
                    out[k] = int(s)
                    continue
                except Exception:
                    pass
            # float?
            if re.fullmatch(r"[+-]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][+-]?\d+)?", s):
                try:
                    out[k] = float(s)
                    continue
                except Exception:
                    pass
        out[k] = v
    return out

def _sanitize_params(params: dict) -> dict:
    """Cast numeric strings to numbers for sklearn param validation (e.g. '0.3' -> 0.3)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, str):
            s = v.strip()
            # int?
            if re.fullmatch(r"[+-]?\d+", s):
                try:
                    out[k] = int(s)
                    continue
                except Exception:
                    pass
            # float?
            if re.fullmatch(r"[+-]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][+-]?\d+)?", s):
                try:
                    out[k] = float(s)
                    continue
                except Exception:
                    pass
        out[k] = v
    return out
import argparse
import re
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import ExtraTreeClassifier

# imblearn is required for two of the four methods:
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier

# ----------------------------
# Resource-aware helpers (HPC-friendly)
# ----------------------------
def _effective_n_jobs(default: int = -1) -> int:
    """Return a safe n_jobs value that respects SLURM allocations when present."""
    # Prefer explicit SLURM allocation (prevents oversubscription on shared nodes)
    slurm = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm:
        try:
            n = int(slurm)
            if n > 0:
                return n
        except Exception:
            pass

    # Fall back to common thread caps if set
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        v = os.environ.get(k)
        if v:
            try:
                n = int(v)
                if n > 0:
                    return n
            except Exception:
                pass

    return default


# Simple in-process caches to avoid re-reading large files for each task in a SLURM array block.
_DF_CACHE: Dict[str, pd.DataFrame] = {}
_SPLIT_MANIFEST_CACHE: Dict[str, pd.DataFrame] = {}

# tqdm is optional; if missing, we fall back to plain loops
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ----------------------------
# Utilities
# ----------------------------
def expand_path(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def stable_hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _make_hashable(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return tuple(_make_hashable(v) for v in x)
    if isinstance(x, dict):
        return tuple((k, _make_hashable(v)) for k, v in sorted(x.items()))
    return x


# ----------------------------
# Data handling
# ----------------------------
def infer_feature_cols(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        for p in prefixes:
            if c.startswith(p):
                cols.append(c)
                break
    if len(cols) == 0:
        raise ValueError(f"No feature columns matched prefixes={prefixes}. Available columns head={list(df.columns)[:20]}")
    return cols


def build_Xy(df: pd.DataFrame, id_col: str, y_col: str, ids: np.ndarray, prefixes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    feat_cols = infer_feature_cols(df, prefixes)
    sub = df.loc[df[id_col].isin(ids), [id_col, y_col] + feat_cols].copy()
    # preserve split ordering based on ids
    sub = sub.set_index(id_col).loc[ids]
    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub[y_col].to_numpy()
    return X, y


def load_split_ids(cfg: dict, fold: int) -> Dict[str, np.ndarray]:
    """
    Expects a split manifest parquet with at least:
        - id_col (e.g., source_id)
        - split in {"train","val","test"}
        - fold (int) OR a single fold file (if no fold col, we take all rows)
    """
    data = cfg["data"]
    id_col = data["id_col"]

    p = expand_path(data["split_manifest"])
    p_key = str(p)
    if p_key in _SPLIT_MANIFEST_CACHE:
        man = _SPLIT_MANIFEST_CACHE[p_key]
    else:
        man = pd.read_parquet(p)
        _SPLIT_MANIFEST_CACHE[p_key] = man

    cols_l = {c.lower(): c for c in man.columns}
    split_col = cols_l.get("split", cols_l.get("subset", None))
    fold_col = cols_l.get("fold", cols_l.get("kfold", None))

    if split_col is None:
        raise ValueError(f"Split manifest must contain a 'split' column. Columns: {list(man.columns)}")

    if fold_col is not None:
        man = man.loc[man[fold_col] == fold]

    def _ids_for(name: str) -> np.ndarray:
        x = man.loc[man[split_col].str.lower() == name, id_col].to_numpy()
        return x.astype(np.int64, copy=False)

    out = {"train": _ids_for("train"), "val": _ids_for("val"), "test": _ids_for("test")}
    for k in out:
        if len(out[k]) == 0:
            raise ValueError(f"Split '{k}' is empty for fold={fold}. Check split manifest: {p}")
    return out


# ----------------------------
# Bootstrap sampling
# ----------------------------
def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Stratified bootstrap: sample each class separately with replacement,
    keeping the same class counts as in y.
    """
    idx = np.arange(len(y))
    out_parts = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        out_parts.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    return np.concatenate(out_parts)


# ----------------------------
# Threshold selection + metrics
# ----------------------------
def pick_threshold(y_true: np.ndarray, p: np.ndarray, criterion: str, grid_size: int = 600) -> Dict[str, float]:
    eps = 1e-12
    # threshold candidates: quantile grid + extremes
    qs = np.linspace(0.0, 1.0, grid_size)
    thrs = np.unique(np.quantile(p, qs))
    thrs = np.clip(thrs, eps, 1.0 - eps)

    best = {"thr": 0.5, "score": -np.inf}
    for thr in thrs:
        y_hat = (p >= thr).astype(int)
        tp = int(((y_true == 1) & (y_hat == 1)).sum())
        tn = int(((y_true == 0) & (y_hat == 0)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())
        fn = int(((y_true == 1) & (y_hat == 0)).sum())

        sens = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)
        prec = tp / (tp + fp + eps)
        f1 = 2 * prec * sens / (prec + sens + eps)
        youden = sens + spec - 1.0

        score = youden if criterion == "youden" else f1
        if score > best["score"]:
            best = {"thr": float(thr), "score": float(score), "sens": float(sens), "spec": float(spec), "prec": float(prec), "f1": float(f1)}
    return best


def metrics_from_proba(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    eps = 1e-12
    y_hat = (p >= thr).astype(int)

    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())

    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    prec = tp / (tp + fp + eps)

    out = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
    }
    return out


def prob_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    # guard against degenerate y_true
    out = {}
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["pr_auc"] = float(average_precision_score(y_true, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    out["brier"] = float(brier_score_loss(y_true, p))
    out["logloss"] = float(log_loss(y_true, np.vstack([1 - p, p]).T, labels=[0, 1]))
    return out


# ----------------------------
# Hyperparameter grids
# ----------------------------
def sample_param_space(space: Dict[str, List[Any]], n_samples: int, seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    keys = list(space.keys())
    seen = set()
    out = []
    tries = 0
    while len(out) < n_samples and tries < n_samples * 100:
        tries += 1
        d = {k: rng.choice(space[k]) for k in keys}
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, np.generic):
                d[k] = v.item()
        key = tuple((k, _make_hashable(d[k])) for k in keys)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


# ----------------------------
# Model fitting
# ----------------------------
def fit_predict_proba(model_key: str, params: dict, X_tr: np.ndarray, y_tr: np.ndarray, X_eval: np.ndarray, seed: int) -> np.ndarray:
    if model_key == "rf":
        mdl = RandomForestClassifier(**params, n_jobs=_effective_n_jobs(-1), random_state=seed)
        mdl.fit(X_tr, y_tr)
        return mdl.predict_proba(X_eval)[:, 1]

    if model_key == "et":
        
        # sklearn constraint: max_samples is only valid when bootstrap=True
        if params.get("bootstrap") is False:
            params["max_samples"] = None
        mdl = ExtraTreesClassifier(**params, n_jobs=_effective_n_jobs(-1), random_state=seed)
        mdl.fit(X_tr, y_tr)
        return mdl.predict_proba(X_eval)[:, 1]

    if model_key == "brf":
        mdl = BalancedRandomForestClassifier(**params, n_jobs=_effective_n_jobs(-1), random_state=seed)
        mdl.fit(X_tr, y_tr)
        return mdl.predict_proba(X_eval)[:, 1]

    if model_key == "bbag":
        # base estimator: fast random-split ExtraTree
        base = ExtraTreeClassifier(
            splitter="random",
            random_state=seed,
            max_features=params.pop("base_max_features", "sqrt"),
            max_depth=params.pop("base_max_depth", None),
            min_samples_leaf=int(params.pop("base_min_samples_leaf", 1)),
            min_samples_split=int(params.pop("base_min_samples_split", 2)),
        )
        mdl = BalancedBaggingClassifier(
            estimator=base,
            n_estimators=int(params["n_estimators"]),
            max_samples=float(params["max_samples"]),
            max_features=float(params["max_features"]),
            bootstrap=bool(params["bootstrap"]),
            sampling_strategy=params.get("sampling_strategy", "auto"),
            replacement=bool(params.get("replacement", False)),
            n_jobs=_effective_n_jobs(-1),
            random_state=seed,
        )
        mdl.fit(X_tr, y_tr)
        return mdl.predict_proba(X_eval)[:, 1]

    raise ValueError(f"Unknown model_key={model_key}")


# ----------------------------
# Task table
# ----------------------------
@dataclass(frozen=True)
class Task:
    task_id: int
    feature_set: str
    model_key: str
    param_index: int
    param_hash: str


def build_tasks(cfg: dict) -> List[Task]:
    exp = cfg["experiment"]
    n_param = int(cfg.get("models", {}).get("n_param_samples", 24))
    seed = int(cfg.get("seed", 1234))

    # Compact but “push-to-limits” spaces (expand later if needed)
    RF_SPACE = {
        "n_estimators": [800, 1500, 2500, 4000],
        "max_features": ["sqrt", 0.2, 0.3, 0.5],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True],
        "class_weight": [None, "balanced"],
    }
    ET_SPACE = {
        "n_estimators": [800, 1500, 2500, 4000],
        "max_features": ["sqrt", 0.2, 0.3, 0.5],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [False, True],
        "max_samples": [None, 0.5, 0.8, 1.0],
    }
    BRF_SPACE = {
        "n_estimators": [800, 1500, 2500, 4000],
        "max_features": ["sqrt", 0.2, 0.3, 0.5],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "sampling_strategy": ["auto"],
        "replacement": [False, True],
    }
    BBAG_SPACE = {
        "n_estimators": [200, 500, 800, 1200],
        "max_samples": [0.5, 0.7, 0.9, 1.0],
        "max_features": [0.5, 0.7, 0.9, 1.0],
        "bootstrap": [True],
        "sampling_strategy": ["auto"],
        "replacement": [False, True],
        # base estimator knobs:
        "base_max_features": ["sqrt", 0.3, 0.5],
        "base_max_depth": [None, 10, 20],
        "base_min_samples_leaf": [1, 2, 4],
        "base_min_samples_split": [2, 4, 8],
    }

    grids = {
        "rf": sample_param_space(RF_SPACE, n_param, seed=seed + 1),
        "et": sample_param_space(ET_SPACE, n_param, seed=seed + 2),
        "brf": sample_param_space(BRF_SPACE, n_param, seed=seed + 3),
        "bbag": sample_param_space(BBAG_SPACE, n_param, seed=seed + 4),
    }
    cfg["_RUNTIME_GRIDS"] = grids

    families = ["rf", "et", "brf", "bbag"]

    tasks = []
    task_id = 0
    for fs_name in exp["feature_sets"].keys():
        for model_key in families:
            for pidx in range(n_param):
                ph = stable_hash_dict(grids[model_key][pidx])
                tasks.append(Task(task_id=task_id, feature_set=fs_name, model_key=model_key, param_index=pidx, param_hash=ph))
                task_id += 1
    return tasks


# ----------------------------
# Output paths
# ----------------------------
def output_base(cfg: dict) -> Path:
    workdir = expand_path(cfg["project"]["workdir"])
    subdir = cfg["data"].get("results_subdir", "results_rf_suite")
    return workdir / subdir


def task_output_paths(cfg: dict, task: Task) -> Tuple[Path, Path, Path]:
    out_dir = output_base(cfg) / f"fold={int(cfg.get('_FOLD',0))}"
    ensure_dir(out_dir)
    raw_path = out_dir / f"raw_task{task.task_id:06d}_{task.feature_set}_{task.model_key}_p{task.param_index:03d}_{task.param_hash}.parquet"
    summary_path = out_dir / f"summary_task{task.task_id:06d}_{task.feature_set}_{task.model_key}_p{task.param_index:03d}_{task.param_hash}.parquet"
    meta_path = out_dir / f"meta_task{task.task_id:06d}_{task.feature_set}_{task.model_key}_p{task.param_index:03d}_{task.param_hash}.json"
    return raw_path, summary_path, meta_path


# ----------------------------
# Main task runner
# ----------------------------
def run_task(cfg: dict, task: Task) -> None:
    raw_path, summary_path, meta_path = task_output_paths(cfg, task)
    if raw_path.exists() and meta_path.exists():
        return  # checkpoint hit

    t0 = time.time()
    data = cfg["data"]
    exp = cfg["experiment"]

    id_col = data["id_col"]
    y_col = data["y_col"]

    # Load main data (per process) — cached to avoid repeated I/O inside a SLURM array block.
    csv_path = expand_path(data["input_csv"])
    csv_key = str(csv_path)
    if csv_key in _DF_CACHE:
        df = _DF_CACHE[csv_key]
    else:
        df = pd.read_csv(csv_path)
        _DF_CACHE[csv_key] = df

    # Splits
    splits = load_split_ids(cfg, fold=int(cfg.get("_FOLD", 0)))
    prefixes = exp["feature_sets"][task.feature_set]["prefixes"]

    X_tr, y_tr = build_Xy(df, id_col, y_col, splits["train"], prefixes)
    X_va, y_va = build_Xy(df, id_col, y_col, splits["val"], prefixes)
    X_te, y_te = build_Xy(df, id_col, y_col, splits["test"], prefixes)

    # Params
    params = dict(cfg["_RUNTIME_GRIDS"][task.model_key][task.param_index])

    n_boot = int(exp.get("n_bootstrap", 50))
    thr_grid = int(exp.get("thr_grid_size", 600))
    seed0 = int(cfg.get("seed", 1234)) + 1000 * task.task_id

    iterator = range(n_boot)
    if _HAS_TQDM and cfg.get("show_progress", True):
        iterator = tqdm(iterator, desc=f"{task.feature_set}|{task.model_key}|p{task.param_index}", leave=False)

    rows = []
    rng_master = np.random.default_rng(seed0)

    for b in iterator:
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)

        boot_idx = stratified_bootstrap_indices(y_tr, rng=rng)
        Xb, yb = X_tr[boot_idx], y_tr[boot_idx]

        p_va = fit_predict_proba(task.model_key, _sanitize_params(dict(params)), Xb, yb, X_va, seed=seed)
        p_te = fit_predict_proba(task.model_key, _sanitize_params(dict(params)), Xb, yb, X_te, seed=seed)

        pick_y = pick_threshold(y_va, p_va, criterion="youden", grid_size=thr_grid)
        pick_f = pick_threshold(y_va, p_va, criterion="f1", grid_size=thr_grid)

        met_va_y = metrics_from_proba(y_va, p_va, thr=pick_y["thr"])
        met_te_y = metrics_from_proba(y_te, p_te, thr=pick_y["thr"])

        met_va_f = metrics_from_proba(y_va, p_va, thr=pick_f["thr"])
        met_te_f = metrics_from_proba(y_te, p_te, thr=pick_f["thr"])

        prob_te = prob_metrics(y_te, p_te)

        # store both threshold criteria as separate rows
        rows.append({
            "task_id": task.task_id,
            "feature_set": task.feature_set,
            "model": task.model_key,
            "param_index": task.param_index,
            "param_hash": task.param_hash,
            "bootstrap": b,
            "criterion": "youden",
            "seed": seed,
            "val_thr": pick_y["thr"],
            "val_score": pick_y["score"],
            "val_Sensitivity": met_va_y["sensitivity"],
            "val_Specificity": met_va_y["specificity"],
            "val_Precision": met_va_y["precision"],
            "test_Sensitivity": met_te_y["sensitivity"],
            "test_Specificity": met_te_y["specificity"],
            "test_Precision": met_te_y["precision"],
            "test_ROC_AUC": prob_te["roc_auc"],
            "test_PR_AUC": prob_te["pr_auc"],
            "test_Brier": prob_te["brier"],
            "test_LogLoss": prob_te["logloss"],
        })
        rows.append({
            "task_id": task.task_id,
            "feature_set": task.feature_set,
            "model": task.model_key,
            "param_index": task.param_index,
            "param_hash": task.param_hash,
            "bootstrap": b,
            "criterion": "f1",
            "seed": seed,
            "val_thr": pick_f["thr"],
            "val_score": pick_f["score"],
            "val_Sensitivity": met_va_f["sensitivity"],
            "val_Specificity": met_va_f["specificity"],
            "val_Precision": met_va_f["precision"],
            "test_Sensitivity": met_te_f["sensitivity"],
            "test_Specificity": met_te_f["specificity"],
            "test_Precision": met_te_f["precision"],
            "test_ROC_AUC": prob_te["roc_auc"],
            "test_PR_AUC": prob_te["pr_auc"],
            "test_Brier": prob_te["brier"],
            "test_LogLoss": prob_te["logloss"],
        })

    raw_df = pd.DataFrame(rows)

    # compact summary across bootstraps
    group_cols = ["feature_set", "model", "param_index", "param_hash", "criterion"]
    metric_cols = [
        "val_score",
        "val_thr",
        "val_Sensitivity", "val_Specificity", "val_Precision",
        "test_Sensitivity", "test_Specificity", "test_Precision",
        "test_ROC_AUC", "test_PR_AUC", "test_Brier", "test_LogLoss",
    ]
    agg = {}
    for c in metric_cols:
        agg[c + "_mean"] = (c, "mean")
        agg[c + "_std"] = (c, "std")

    summary_df = (
        raw_df
        .groupby(group_cols, as_index=False)
        .agg(**agg)
        .sort_values(["feature_set", "model", "criterion"])
        .reset_index(drop=True)
    )

    meta = {
        "task_id": task.task_id,
        "feature_set": task.feature_set,
        "model": task.model_key,
        "param_index": task.param_index,
        "param_hash": task.param_hash,
        "params": cfg["_RUNTIME_GRIDS"][task.model_key][task.param_index],
        "n_bootstrap": n_boot,
        "thr_grid_size": thr_grid,
        "runtime_s": float(time.time() - t0),
        "summary_preview": summary_df.head(8).to_dict(orient="records"),
    }

    _atomic_write_parquet(raw_path, raw_df)
    _atomic_write_parquet(summary_path, summary_df)
    _atomic_write_json(meta_path, meta)


# ----------------------------
# CLI
# ----------------------------
def parse_task_block(s: str) -> Tuple[int, int]:
    if ":" not in s:
        raise ValueError("task-block must be START:COUNT")
    a, b = s.split(":")
    return int(a), int(b)


def load_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def write_task_table(cfg: dict, tasks: List[Task]) -> None:
    base = output_base(cfg).parent / "tasks"
    ensure_dir(base)
    tdf = pd.DataFrame([{
        "task_id": t.task_id,
        "feature_set": t.feature_set,
        "model": t.model_key,
        "param_index": t.param_index,
        "param_hash": t.param_hash,
    } for t in tasks])
    tpath = base / "tasks.parquet"
    mpath = base / "tasks_meta.json"
    _atomic_write_parquet(tpath, tdf)
    _atomic_write_json(mpath, {
        "n_tasks": int(len(tasks)),
        "n_param_samples": int(cfg.get("models", {}).get("n_param_samples", 24)),
        "feature_sets": list(cfg["experiment"]["feature_sets"].keys()),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--task-block", type=str, default=None, help="START:COUNT")
    ap.add_argument("--show-progress", action="store_true", default=False)
    ap.add_argument("--write-task-table", action="store_true", default=False)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    cfg["_FOLD"] = int(args.fold)
    cfg["show_progress"] = bool(args.show_progress)

    tasks = build_tasks(cfg)

    if args.write_task_table:
        write_task_table(cfg, tasks)

    if args.task_id is not None:
        run_task(cfg, tasks[int(args.task_id)])
        return

    if args.task_block is not None:
        start, count = parse_task_block(args.task_block)
        end = min(len(tasks), start + count)
        for tid in range(start, end):
            run_task(cfg, tasks[tid])
        return

    # default: run everything in-process (not recommended on HPC)
    for t in tasks:
        run_task(cfg, t)


if __name__ == "__main__":
    main()
