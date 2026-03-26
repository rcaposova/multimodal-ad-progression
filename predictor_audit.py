# predictor_audit.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None


# ========================
# User-configurable paths
# ========================
PROJECT_DIR = Path(".")
ARTIFACT_DIR = PROJECT_DIR / "preprocessing_artifacts"
OUT_DIR = PROJECT_DIR / "predictor_audit_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# serialized objects.
ALL_SPLITS_PATHS = [
    PROJECT_DIR / "all_splits.joblib",
    PROJECT_DIR / "all_splits.pkl",
]

# save for a richer audit.
OPTIONAL_OBJECT_PATHS = {
    "raw_data": [
        PROJECT_DIR / "raw_data.joblib",
        PROJECT_DIR / "raw_data.pkl",
    ],
    "cleaned_data": [
        PROJECT_DIR / "cleaned_data.joblib",
        PROJECT_DIR / "cleaned_data.pkl",
    ],
    "merged_data_raw": [
        PROJECT_DIR / "merged_data_raw.joblib",
        PROJECT_DIR / "merged_data_raw.pkl",
    ],
    "baseline_static_raw": [
        PROJECT_DIR / "baseline_static_raw.joblib",
        PROJECT_DIR / "baseline_static_raw.pkl",
    ],
    "mri_raw_visits": [
        PROJECT_DIR / "mri_raw_visits.joblib",
        PROJECT_DIR / "mri_raw_visits.pkl",
    ],
    "pet_raw_visits": [
        PROJECT_DIR / "pet_raw_visits.joblib",
        PROJECT_DIR / "pet_raw_visits.pkl",
    ],
    "tau_master_raw": [
        PROJECT_DIR / "tau_master_raw.joblib",
        PROJECT_DIR / "tau_master_raw.pkl",
    ],
}

LEAKAGE_EXCLUDE_COLS = [
    "DIAGNOSIS",
    "DIAGNOSIS_ENC",
    "ADAS_11",
    "ADAS_13",
    "MMSCORE",
    "FAQTOTAL",
    "CDRSB",
    "DATE",
    "EXAMDATE",
    "VISDATE",
]

TARGET_COLS = ["ADAS_13", "MMSCORE", "FAQTOTAL", "CDRSB"]

DIRECT_FORBIDDEN: Set[str] = set(LEAKAGE_EXCLUDE_COLS) | set(TARGET_COLS)

# Terms that may indicate target-adjacent or clinically overlapping predictors.
# This is intentionally broad enough to catch obvious issues.
SUSPICIOUS_PATTERNS = [
    r"\bdiag",
    r"\bdiagnosis\b",
    r"\bcdr\b",
    r"\bcdrsb\b",
    r"\bmmse\b",
    r"\bmmscore\b",
    r"\bfaq\b",
    r"\bfaqtotal\b",
    r"\badas\b",
    r"\badas_?11\b",
    r"\badas_?13\b",
    r"\bscore\b",
    r"\btotal\b",
    r"\bglobal\b",
    r"\bseverity\b",
    r"\bdement",
    r"\bcogn",
    r"\bmemory\b",
    r"\bimpair",
    r"\bdecline\b",
    r"\bfunction",
    r"\bexecutive\b",
]
SUSPICIOUS_REGEX = re.compile("|".join(SUSPICIOUS_PATTERNS), flags=re.IGNORECASE)


# ===========
# IO helpers
# ===========
def load_serialized(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".joblib":
        if joblib is None:
            raise ImportError("joblib is not installed, cannot read .joblib")
        return joblib.load(path)

    if suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Unsupported file type: {path}")


def try_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================
# Normalization helpers
# ======================
def normalize_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x.upper()


def normalize_names(cols: Iterable[str]) -> List[str]:
    return [normalize_name(c) for c in cols]


def is_suspicious_name(col: str) -> bool:
    return bool(SUSPICIOUS_REGEX.search(str(col)))


# ============================================
# Audit from split_preproc logs / all_splits
# ============================================
def extract_split_log_rows_from_artifacts(artifact_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(artifact_dir.glob("split_*_preprocessing_log.json")):
        payload = read_json(path)
        seed = payload.get("seed")

        def add_branch(branch_name: str, info: Dict[str, Any]):
            fit_cols = info.get("fit_cols_before_const_drop", None)
            keep_cols = info.get("keep_cols", None)

            rows.append({
                "seed": seed,
                "branch": branch_name,
                "fit_cols_before_const_drop_n": len(fit_cols) if isinstance(fit_cols, list) else np.nan,
                "keep_cols_n": len(keep_cols) if isinstance(keep_cols, list) else np.nan,
                "fit_cols_before_const_drop": fit_cols if isinstance(fit_cols, list) else None,
                "keep_cols": keep_cols if isinstance(keep_cols, list) else None,
            })

        if "stats" in payload:
            add_branch("stats", payload["stats"])
        if "static" in payload:
            add_branch("static", payload["static"])
        if "mri" in payload:
            add_branch("mri", payload["mri"])
        if "pet" in payload:
            add_branch("pet", payload["pet"])
        if "tau" in payload:
            add_branch("tau", payload["tau"])

    return pd.DataFrame(rows)


def expand_artifact_branch_inventory(artifact_branch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if artifact_branch_df.empty:
        return pd.DataFrame()

    for _, r in artifact_branch_df.iterrows():
        seed = r.get("seed", np.nan)
        branch = r.get("branch", None)

        fit_cols = r.get("fit_cols_before_const_drop", None)
        keep_cols = r.get("keep_cols", None)

        if isinstance(fit_cols, list):
            for c in fit_cols:
                rows.append({
                    "seed": seed,
                    "branch": branch,
                    "stage": "fit_cols_before_const_drop",
                    "column_original": c,
                    "column_norm": normalize_name(c),
                })

        if isinstance(keep_cols, list):
            for c in keep_cols:
                rows.append({
                    "seed": seed,
                    "branch": branch,
                    "stage": "keep_cols",
                    "column_original": c,
                    "column_norm": normalize_name(c),
                })

    return pd.DataFrame(rows)


def extract_branch_predictors_from_all_splits(all_splits: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for split in all_splits:
        seed = split.get("seed", np.nan)
        log = split.get("split_preproc_log", {})

        for branch in ["stats", "static", "mri", "pet", "tau"]:
            info = log.get(branch, {})
            fit_cols = info.get("fit_cols_before_const_drop", [])
            keep_cols = info.get("keep_cols", [])

            if isinstance(fit_cols, list):
                for c in fit_cols:
                    rows.append({
                        "seed": seed,
                        "branch": branch,
                        "stage": "fit_cols_before_const_drop",
                        "column_original": c,
                        "column_norm": normalize_name(c),
                    })

            if isinstance(keep_cols, list):
                for c in keep_cols:
                    rows.append({
                        "seed": seed,
                        "branch": branch,
                        "stage": "keep_cols",
                        "column_original": c,
                        "column_norm": normalize_name(c),
                    })

    return pd.DataFrame(rows)


# =========================================================
# richer audit from raw DataFrames if available
# =========================================================
def infer_numeric_predictor_columns(
    df: pd.DataFrame,
    exclude: Iterable[str],
) -> List[str]:
    exclude_norm = set(normalize_names(exclude))
    out = []
    for c in df.columns:
        cn = normalize_name(c)
        if cn in exclude_norm:
            continue
        if cn in {"RID", "VISCODE"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def dataframe_inventory(
    objects: Dict[str, Any],
    exclude: Iterable[str],
) -> pd.DataFrame:
    rows = []
    forbidden_norm = set(normalize_names(DIRECT_FORBIDDEN))

    for name, obj in objects.items():
        if not isinstance(obj, pd.DataFrame):
            continue

        pred_cols = infer_numeric_predictor_columns(obj, exclude)
        for c in pred_cols:
            rows.append({
                "dataset_object": name,
                "column_original": c,
                "column_norm": normalize_name(c),
                "dtype": str(obj[c].dtype),
                "is_suspicious_name": is_suspicious_name(c),
                "is_direct_forbidden": normalize_name(c) in forbidden_norm,
            })
    return pd.DataFrame(rows)


# ============
# Core audits
# ============
def direct_forbidden_audit(predictor_df: pd.DataFrame) -> pd.DataFrame:
    forbidden_norm = set(normalize_names(DIRECT_FORBIDDEN))
    out = predictor_df.copy()
    out["is_direct_forbidden"] = out["column_norm"].isin(forbidden_norm)
    return out[out["is_direct_forbidden"]].copy()


def suspicious_name_audit(predictor_df: pd.DataFrame) -> pd.DataFrame:
    out = predictor_df.copy()
    out["is_suspicious_name"] = out["column_original"].map(is_suspicious_name)
    return out[out["is_suspicious_name"]].copy()


def predictor_summary(predictor_df: pd.DataFrame) -> pd.DataFrame:
    if predictor_df.empty:
        return pd.DataFrame()

    grp_cols = [c for c in ["branch", "stage", "column_norm"] if c in predictor_df.columns]
    agg_spec = {
        "example_name": ("column_original", "first"),
    }

    if "seed" in predictor_df.columns:
        agg_spec["n_splits"] = ("seed", "nunique")

    summary = (
        predictor_df
        .groupby(grp_cols, as_index=False)
        .agg(**agg_spec)
        .sort_values(grp_cols)
        .reset_index(drop=True)
    )
    return summary


def branch_inventory_table(predictor_df: pd.DataFrame) -> pd.DataFrame:
    if predictor_df.empty:
        return pd.DataFrame()

    if not {"branch", "stage"}.issubset(predictor_df.columns):
        return pd.DataFrame()

    agg_spec = {
        "n_unique_predictors": ("column_norm", "nunique"),
    }
    if "seed" in predictor_df.columns:
        agg_spec["n_splits"] = ("seed", "nunique")

    grp = (
        predictor_df
        .groupby(["branch", "stage"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["branch", "stage"])
        .reset_index(drop=True)
    )
    return grp


def unique_predictor_name_table(predictor_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-row-per-normalized-predictor export for quick manual review.
    This is the main missing practical step for the rebuttal/manuscript.
    """
    if predictor_df.empty:
        return pd.DataFrame(columns=[
            "column_norm",
            "example_name",
            "is_direct_forbidden",
            "is_suspicious_name",
            "branches_seen",
            "stages_seen",
            "n_rows",
            "n_splits",
        ])

    work = predictor_df.copy()
    work["is_direct_forbidden"] = work["column_norm"].isin(set(normalize_names(DIRECT_FORBIDDEN)))
    work["is_suspicious_name"] = work["column_original"].map(is_suspicious_name)

    def join_unique(vals: pd.Series) -> str:
        vals = [str(v) for v in vals.dropna().unique().tolist() if str(v) != ""]
        vals = sorted(vals)
        return "; ".join(vals)

    agg_dict = {
        "example_name": ("column_original", "first"),
        "is_direct_forbidden": ("is_direct_forbidden", "max"),
        "is_suspicious_name": ("is_suspicious_name", "max"),
        "branches_seen": ("branch", join_unique) if "branch" in work.columns else ("column_norm", lambda s: ""),
        "stages_seen": ("stage", join_unique) if "stage" in work.columns else ("column_norm", lambda s: ""),
        "n_rows": ("column_norm", "size"),
    }

    if "seed" in work.columns:
        agg_dict["n_splits"] = ("seed", "nunique")
    else:
        agg_dict["n_splits"] = ("column_norm", lambda s: np.nan)

    out = (
        work
        .groupby("column_norm", as_index=False)
        .agg(**agg_dict)
        .sort_values(["is_direct_forbidden", "is_suspicious_name", "column_norm"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return out


def manual_review_candidates(unique_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact table of names to manually inspect.
    Even if empty, it is useful evidence that the screen was run.
    """
    if unique_df.empty:
        return pd.DataFrame(columns=[
            "column_norm",
            "example_name",
            "is_direct_forbidden",
            "is_suspicious_name",
            "branches_seen",
            "stages_seen",
            "n_splits",
        ])

    cols = [
        "column_norm",
        "example_name",
        "is_direct_forbidden",
        "is_suspicious_name",
        "branches_seen",
        "stages_seen",
        "n_splits",
    ]
    cols = [c for c in cols if c in unique_df.columns]

    out = unique_df[
        unique_df["is_direct_forbidden"] | unique_df["is_suspicious_name"]
    ][cols].copy()

    return out.sort_values(["is_direct_forbidden", "is_suspicious_name", "column_norm"], ascending=[False, False, True]).reset_index(drop=True)


def coverage_table(predictor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional descriptive table for the rebuttal:
    how many unique predictors per branch/stage and how consistently they appear across splits.
    """
    if predictor_df.empty or not {"branch", "stage", "column_norm"}.issubset(predictor_df.columns):
        return pd.DataFrame()

    if "seed" not in predictor_df.columns:
        out = (
            predictor_df
            .groupby(["branch", "stage"], as_index=False)
            .agg(n_unique_predictors=("column_norm", "nunique"))
            .sort_values(["branch", "stage"])
            .reset_index(drop=True)
        )
        return out

    per_predictor = (
        predictor_df
        .groupby(["branch", "stage", "column_norm"], as_index=False)
        .agg(n_splits_present=("seed", "nunique"))
    )

    out = (
        per_predictor
        .groupby(["branch", "stage"], as_index=False)
        .agg(
            n_unique_predictors=("column_norm", "nunique"),
            min_splits_present=("n_splits_present", "min"),
            median_splits_present=("n_splits_present", "median"),
            max_splits_present=("n_splits_present", "max"),
        )
        .sort_values(["branch", "stage"])
        .reset_index(drop=True)
    )
    return out


# ====
# Main
# ====
def main() -> None:
    print("Running predictor audit...")
    print(f"Project dir: {PROJECT_DIR.resolve()}")
    print(f"Artifact dir: {ARTIFACT_DIR.resolve()}")
    print(f"Output dir: {OUT_DIR.resolve()}")

    all_splits = None
    all_splits_path = try_first_existing(ALL_SPLITS_PATHS)
    if all_splits_path is not None:
        print(f"Loading all_splits from: {all_splits_path}")
        all_splits = load_serialized(all_splits_path)
    else:
        print("No serialized all_splits found. Will rely on artifact logs where possible.")

    optional_objects: Dict[str, Any] = {}
    for name, paths in OPTIONAL_OBJECT_PATHS.items():
        p = try_first_existing(paths)
        if p is not None:
            print(f"Loading optional object '{name}' from: {p}")
            optional_objects[name] = load_serialized(p)

    # 1) Branch/predictor extraction
    predictor_rows = []

    if all_splits is not None:
        split_predictor_df = extract_branch_predictors_from_all_splits(all_splits)
        if not split_predictor_df.empty:
            predictor_rows.append(split_predictor_df)

    artifact_branch_df = extract_split_log_rows_from_artifacts(ARTIFACT_DIR)
    artifact_branch_df.to_csv(OUT_DIR / "artifact_branch_inventory_raw.csv", index=False)

    artifact_predictor_df = expand_artifact_branch_inventory(artifact_branch_df)
    if not artifact_predictor_df.empty:
        predictor_rows.append(artifact_predictor_df)

    if optional_objects:
        df_inventory = dataframe_inventory(optional_objects, exclude=LEAKAGE_EXCLUDE_COLS)
        if not df_inventory.empty:
            predictor_rows.append(df_inventory)

    if predictor_rows:
        predictor_df = pd.concat(predictor_rows, ignore_index=True, sort=False)
        predictor_df.to_csv(OUT_DIR / "predictor_inventory_long.csv", index=False)
    else:
        predictor_df = pd.DataFrame()
        pd.DataFrame().to_csv(OUT_DIR / "predictor_inventory_long.csv", index=False)

    # 2) Summaries
    summary_df = predictor_summary(predictor_df) if not predictor_df.empty else pd.DataFrame()
    summary_df.to_csv(OUT_DIR / "predictor_inventory_summary.csv", index=False)

    branch_inventory_df = branch_inventory_table(predictor_df) if not predictor_df.empty else pd.DataFrame()
    branch_inventory_df.to_csv(OUT_DIR / "branch_inventory_summary.csv", index=False)

    coverage_df = coverage_table(predictor_df) if not predictor_df.empty else pd.DataFrame()
    coverage_df.to_csv(OUT_DIR / "predictor_coverage_summary.csv", index=False)

    # 3) Direct forbidden audit
    forbidden_df = direct_forbidden_audit(predictor_df) if not predictor_df.empty else pd.DataFrame()
    forbidden_df.to_csv(OUT_DIR / "direct_forbidden_predictors_found.csv", index=False)

    # 4) Suspicious-name audit
    suspicious_df = suspicious_name_audit(predictor_df) if not predictor_df.empty else pd.DataFrame()
    suspicious_df.to_csv(OUT_DIR / "suspicious_name_predictors_found.csv", index=False)

    # 5) Unique predictor list for manual review
    unique_name_df = unique_predictor_name_table(predictor_df) if not predictor_df.empty else pd.DataFrame()
    unique_name_df.to_csv(OUT_DIR / "unique_predictor_names.csv", index=False)

    manual_review_df = manual_review_candidates(unique_name_df) if not unique_name_df.empty else pd.DataFrame()
    manual_review_df.to_csv(OUT_DIR / "manual_review_candidates.csv", index=False)

    # 6) High-level report
    report = {
        "project_dir": str(PROJECT_DIR.resolve()),
        "artifact_dir": str(ARTIFACT_DIR.resolve()),
        "used_serialized_all_splits": str(all_splits_path.resolve()) if all_splits_path is not None else None,
        "used_optional_objects": {k: True for k in optional_objects.keys()},
        "n_predictor_rows_total": int(len(predictor_df)),
        "n_unique_predictor_names": int(predictor_df["column_norm"].nunique()) if not predictor_df.empty else 0,
        "n_direct_forbidden_rows_found": int(len(forbidden_df)),
        "n_direct_forbidden_unique_found": int(forbidden_df["column_norm"].nunique()) if not forbidden_df.empty else 0,
        "n_suspicious_rows_found": int(len(suspicious_df)),
        "n_suspicious_unique_found": int(suspicious_df["column_norm"].nunique()) if not suspicious_df.empty else 0,
        "n_manual_review_candidates": int(len(manual_review_df)),
        "direct_forbidden_names": sorted(forbidden_df["column_norm"].unique().tolist()) if not forbidden_df.empty else [],
        "suspicious_names": sorted(suspicious_df["column_norm"].unique().tolist()) if not suspicious_df.empty else [],
        "note": (
            "This audit checks split-specific predictor inventories for direct forbidden target columns "
            "and for obvious target-adjacent names. It is strong evidence against inclusion of direct targets "
            "or obvious name-level duplicates, but it is not a guarantee against every possible semantic surrogate."
        ),
        "recommended_next_step": (
            "Open unique_predictor_names.csv and manually skim the predictor names once. "
            "If manual_review_candidates.csv is empty, the name-based screen found no obvious target-adjacent predictors."
        ),
    }

    with open(OUT_DIR / "predictor_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 7) Friendly console summary
    print("\nAudit finished.")
    print(f"Total predictor rows: {report['n_predictor_rows_total']}")
    print(f"Unique predictor names: {report['n_unique_predictor_names']}")
    print(f"Direct forbidden rows found: {report['n_direct_forbidden_rows_found']}")
    print(f"Suspicious-name rows found: {report['n_suspicious_rows_found']}")
    print(f"Manual review candidates: {report['n_manual_review_candidates']}")

    if report["direct_forbidden_names"]:
        print("\nDirect forbidden predictors found:")
        for name in report["direct_forbidden_names"]:
            print(" -", name)
    else:
        print("\nNo direct forbidden predictors were found in the audited inventories.")

    if report["suspicious_names"]:
        print("\nSuspicious predictor names found:")
        for name in report["suspicious_names"]:
            print(" -", name)
    else:
        print("\nNo suspicious predictor names were found in the audited inventories.")

    print(f"\nOutputs written to: {OUT_DIR.resolve()}")
    print("Most useful files:")
    print(f" - {OUT_DIR / 'predictor_audit_report.json'}")
    print(f" - {OUT_DIR / 'unique_predictor_names.csv'}")
    print(f" - {OUT_DIR / 'manual_review_candidates.csv'}")


if __name__ == "__main__":
    main()