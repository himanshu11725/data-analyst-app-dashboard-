"""Cleaning and profiling engine. Pure pandas, no UI code, so it is easy to test."""
from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_MB = 20
MAX_ROWS = 200_000

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "#n/a"}
ID_LIKE = re.compile(r"(^|_)(id|zip|pin|pincode|phone|mobile|code|sku)($|_)", re.I)


@dataclass
class CleanOptions:
    drop_blank_rows: bool = True
    drop_blank_cols: bool = True
    drop_duplicates: bool = True
    trim_text: bool = True
    unify_text_case: bool = True
    fix_nulls: bool = True
    fix_numbers: bool = True
    fix_dates: bool = True
    day_first: bool = True  # 05/03/2024 means 5 March (common in India/UK)
    snake_case_columns: bool = True


@dataclass
class CleanResult:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)


class FileError(ValueError):
    """Raised for problems the user should see (bad type, too big, unreadable)."""


def load_file(raw: bytes, filename: str) -> pd.DataFrame:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise FileError("Unsupported file type. Please upload a .csv, .xlsx or .xls file.")
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise FileError(f"File is larger than {MAX_FILE_MB} MB. Please upload a smaller file.")

    try:
        if ext == ".csv":
            df = None
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    # sep=None lets pandas sniff comma, semicolon, tab, etc.
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise FileError("Could not read the file encoding.")
        else:
            df = pd.read_excel(io.BytesIO(raw))
    except FileError:
        raise
    except Exception as exc:  # noqa: BLE001 - show a friendly message, not a stack trace
        raise FileError(f"Could not read this file. Is it a valid spreadsheet? ({type(exc).__name__})") from exc

    if df.empty:
        raise FileError("The file has no data rows.")
    if len(df) > MAX_ROWS:
        raise FileError(f"The file has more than {MAX_ROWS:,} rows. Please upload a smaller file.")
    return df


def _snake(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    return name or "column"


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def _parse_dates(s: pd.Series, day_first: bool) -> pd.Series:
    """Parse mixed date text. ISO-style dates (2024-03-06) are always read year-first."""
    txt = s.dropna().astype(str)
    iso = txt.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if iso.any():
            parts.append(pd.to_datetime(txt[iso], errors="coerce", dayfirst=False, format="mixed"))
        if (~iso).any():
            parts.append(pd.to_datetime(txt[~iso], errors="coerce", dayfirst=day_first, format="mixed"))
    if not parts:
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    return pd.concat(parts).reindex(s.index)


def _is_text(s: pd.Series) -> bool:
    """True for text columns in both older pandas (object) and newer pandas (str dtype)."""
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype)


def clean(df: pd.DataFrame, opts: CleanOptions | None = None) -> CleanResult:
    opts = opts or CleanOptions()
    df = df.copy()
    log: list[str] = []

    # --- column names -------------------------------------------------
    if opts.snake_case_columns:
        new_cols = _unique([_snake(c) for c in df.columns])
        changed = sum(1 for a, b in zip(df.columns, new_cols) if str(a) != b)
        df.columns = new_cols
        if changed:
            log.append(f"Standardized {changed} column name(s) to snake_case.")
    else:
        df.columns = _unique([str(c).strip() for c in df.columns])

    text_cols = [c for c in df.columns if _is_text(df[c])]

    # --- whitespace ---------------------------------------------------
    if opts.trim_text:
        fixed = 0
        for c in text_cols:
            s = df[c]
            cleaned = s.where(s.isna(), s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
            fixed += int((cleaned.fillna("") != s.fillna("")).sum())
            df[c] = cleaned
        if fixed:
            log.append(f"Trimmed extra spaces in {fixed} cell(s).")

    # --- placeholder nulls -> real blanks ------------------------------
    if opts.fix_nulls:
        fixed = 0
        for c in text_cols:
            mask = df[c].notna() & df[c].astype(str).str.strip().str.lower().isin(NULL_TOKENS)
            fixed += int(mask.sum())
            df.loc[mask, c] = np.nan
        if fixed:
            log.append(f"Converted {fixed} placeholder value(s) like 'N/A' or '-' into real blanks.")

    # --- blank rows / columns -----------------------------------------
    if opts.drop_blank_rows:
        before = len(df)
        df = df.dropna(how="all")
        if before - len(df):
            log.append(f"Removed {before - len(df)} completely blank row(s).")
    if opts.drop_blank_cols:
        before = df.shape[1]
        df = df.dropna(axis=1, how="all")
        if before - df.shape[1]:
            log.append(f"Removed {before - df.shape[1]} completely blank column(s).")

    # --- numbers stored as text ---------------------------------------
    if opts.fix_numbers:
        for c in [c for c in df.columns if _is_text(df[c])]:
            if ID_LIKE.search(c):
                continue
            s = df[c].dropna().astype(str)
            if s.empty:
                continue
            stripped = s.str.replace(r"[,\s$€£₹%]", "", regex=True)
            stripped = stripped.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (500) -> -500
            nums = pd.to_numeric(stripped, errors="coerce")
            # Only convert when nearly everything parses and no leading zeros (e.g. 007)
            if nums.notna().mean() >= 0.9 and not s.str.match(r"^0\d+").any():
                full = df[c].astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True)
                full = full.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                df[c] = pd.to_numeric(full, errors="coerce")
                log.append(f"Column '{c}': converted text to numbers.")

    # --- dates stored as text -----------------------------------------
    if opts.fix_dates:
        for c in [c for c in df.columns if _is_text(df[c])]:
            s = df[c].dropna().astype(str)
            if s.empty or not s.str.contains(r"\d").all():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = _parse_dates(s, opts.day_first)
            if parsed.notna().mean() >= 0.8:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[c] = _parse_dates(df[c], opts.day_first)
                bad = int(df[c].isna().sum() - (len(df) - len(s)))
                log.append(
                    f"Column '{c}': standardized dates to YYYY-MM-DD"
                    + (f" ({bad} unreadable value(s) left blank)." if bad > 0 else ".")
                )

    # --- inconsistent text (Delhi / delhi / DELHI) --------------------
    if opts.unify_text_case:
        for c in [c for c in df.columns if _is_text(df[c])]:
            s = df[c].dropna().astype(str)
            if s.empty:
                continue
            key = s.str.lower()
            canon = s.groupby(key).agg(lambda x: x.value_counts().index[0])
            mapped = df[c].map(lambda v: canon.get(str(v).lower(), v) if pd.notna(v) else v)
            changed = int((mapped.fillna("") != df[c].fillna("")).sum())
            if changed:
                df[c] = mapped
                log.append(f"Column '{c}': unified spelling/case in {changed} cell(s).")

    # --- whole numbers that became decimals (1001.0 -> 1001) -----------
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            s = df[c].dropna()
            if not s.empty and (s % 1 == 0).all():
                df[c] = df[c].astype("Int64")

    # --- duplicates ---------------------------------------------------
    if opts.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        if before - len(df):
            log.append(f"Removed {before - len(df)} duplicate row(s).")

    df = df.reset_index(drop=True)
    if not log:
        log.append("No problems found. Your data was already clean.")
    return CleanResult(df=df, log=log)


def profile(df: pd.DataFrame) -> dict:
    """Summary numbers for the dashboard."""
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isna().sum()
    cols = []
    for c in df.columns:
        info = {
            "column": c,
            "type": "date" if pd.api.types.is_datetime64_any_dtype(df[c])
            else "number" if pd.api.types.is_numeric_dtype(df[c]) else "text",
            "missing": int(missing[c]),
            "missing_%": round(missing[c] / len(df) * 100, 1) if len(df) else 0.0,
            "unique": int(df[c].nunique(dropna=True)),
        }
        if info["type"] == "number":
            s = df[c].dropna()
            if len(s) >= 4:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                info["outliers"] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            else:
                info["outliers"] = 0
        else:
            info["outliers"] = None
        cols.append(info)

    numeric = df.select_dtypes(include="number")
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(missing.sum()),
        "missing_pct": round(missing.sum() / total_cells * 100, 1) if total_cells else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_table": pd.DataFrame(cols),
        "numeric_summary": numeric.describe().T.round(2) if not numeric.empty else None,
        "correlation": numeric.corr().round(2) if numeric.shape[1] >= 2 else None,
    }
