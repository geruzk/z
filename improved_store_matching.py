"""Improved bidirectional store matching.

Key upgrades:
- Generic Arabic retail tokens are ignored/downweighted for name/address scoring.
- Confirmed matching uses strict distance <= 30m.
- Weighted score uses: Name 15%, Location 60%, Address 15%, New Address 10%.
- Global one-to-one deduplication per comparison via greedy max-score assignment.
"""

import io
import math
import os
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.neighbors import BallTree


# =========================
# Configuration
# =========================
CANDIDATE_RADIUS_M = 120
CONFIRMED_DISTANCE_M = 30
STRONG_POSSIBLE_DISTANCE_M = 60
POSSIBLE_DISTANCE_M = 100
HARD_VETO_DISTANCE_M = 200
HARD_VETO_NAME_MIN = 5
MAX_CANDIDATES = 60

# Requested weights
NAME_WEIGHT = 0.15
LOCATION_WEIGHT = 0.60
ADDRESS_WEIGHT = 0.15
NEW_ADDRESS_WEIGHT = 0.10

CONFIRMED_NAME_THRESHOLD = 65
STRONG_POSSIBLE_NAME_THRESHOLD = 55
POSSIBLE_NAME_THRESHOLD = 50

ERROR_MARKERS = ["failed after retries", "error", "timeout", "no address found", "http error", "exception", "❌"]

GENERIC_RETAIL_TOKENS: Set[str] = {
    "ماركت", "سوبر", "هايبر", "ميني", "بقالة", "بقاله", "بازار", "اسواق", "سوق", "ستور", "market", "store",
    "shop", "center", "سنتر", "محمصة", "محمصه", "عطارة", "عطاره", "كشك",
}

ARABIC_DIACRITIC_RE = re.compile("[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]")
TATWEEL_RE = re.compile("\u0640")
PUNCT_RE = re.compile(r"[^0-9a-zء-ي\s]")


@dataclass
class Candidate:
    a_idx: int
    b_idx: int
    name_score: float
    address_score: Optional[float]
    new_address_score: Optional[float]
    location_score: float
    distance_m: Optional[float]
    weighted_score: float
    confidence: str
    shared_generic_tokens: str
    shared_distinctive_tokens: str
    reason_code: str


# =========================
# Text helpers
# =========================
def _safe_str(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if s.lower() in ("nan", "none", "null", "#n/a", "n/a", "#ref!", ""):
        return ""
    return unicodedata.normalize("NFC", s)


def _is_error_message(s: str) -> bool:
    lower = _safe_str(s).lower()
    return bool(lower) and any(marker in lower for marker in ERROR_MARKERS)


def _normalize_arabic_letters(s: str) -> str:
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    s = s.replace("ة", "ه")
    s = s.replace("گ", "ك").replace("پ", "ب").replace("چ", "ج").replace("ژ", "ز")
    return s


def normalize_text(raw: object) -> str:
    s = _safe_str(raw)
    if not s:
        return ""
    s = ARABIC_DIACRITIC_RE.sub("", s)
    s = TATWEEL_RE.sub("", s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s).lower()
    s = _normalize_arabic_letters(s)
    s = PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text: str) -> List[str]:
    txt = normalize_text(text)
    return [t for t in txt.split() if t]


def split_tokens(text: str) -> Tuple[List[str], List[str]]:
    all_tokens = tokenize(text)
    distinctive = [t for t in all_tokens if t not in GENERIC_RETAIL_TOKENS and not t.isdigit()]
    return all_tokens, distinctive


# =========================
# Numeric / geo helpers
# =========================
def safe_float(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def is_valid_coordinate(lat: Optional[float], lng: Optional[float]) -> bool:
    return lat is not None and lng is not None and np.isfinite(lat) and np.isfinite(lng) and not (lat == 0.0 and lng == 0.0)


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000.0
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def location_score(distance_m: Optional[float]) -> float:
    if distance_m is None:
        return 0.0
    if distance_m <= CONFIRMED_DISTANCE_M:
        return round(max(0.0, 100.0 - (distance_m / CONFIRMED_DISTANCE_M) * 20.0), 2)
    if distance_m <= STRONG_POSSIBLE_DISTANCE_M:
        return round(max(0.0, 80.0 - ((distance_m - CONFIRMED_DISTANCE_M) / (STRONG_POSSIBLE_DISTANCE_M - CONFIRMED_DISTANCE_M)) * 30.0), 2)
    if distance_m <= POSSIBLE_DISTANCE_M:
        return round(max(0.0, 50.0 - ((distance_m - STRONG_POSSIBLE_DISTANCE_M) / (POSSIBLE_DISTANCE_M - STRONG_POSSIBLE_DISTANCE_M)) * 50.0), 2)
    return 0.0


# =========================
# Scoring
# =========================
def name_similarity(a_name: str, b_name: str) -> Tuple[float, bool, str, str]:
    a_all, a_dist = split_tokens(a_name)
    b_all, b_dist = split_tokens(b_name)

    if not a_all or not b_all:
        return 0.0, True, "", ""

    shared_all = set(a_all) & set(b_all)
    shared_generic = sorted(t for t in shared_all if t in GENERIC_RETAIL_TOKENS)
    shared_distinctive = sorted(set(a_dist) & set(b_dist))

    generic_only_overlap = bool(shared_generic) and not shared_distinctive

    # Decision score uses distinctive tokens where possible.
    a_text = " ".join(a_dist) if a_dist else ""
    b_text = " ".join(b_dist) if b_dist else ""

    if a_text and b_text:
        token_set = fuzz.token_set_ratio(a_text, b_text)
        ratio = fuzz.ratio(a_text, b_text)
        partial = fuzz.partial_ratio(a_text, b_text)
        score = 0.5 * token_set + 0.3 * ratio + 0.2 * partial
    else:
        # If no distinctive tokens, fall back to full name with penalty.
        token_set = fuzz.token_set_ratio(normalize_text(a_name), normalize_text(b_name))
        ratio = fuzz.ratio(normalize_text(a_name), normalize_text(b_name))
        partial = fuzz.partial_ratio(normalize_text(a_name), normalize_text(b_name))
        score = (0.5 * token_set + 0.3 * ratio + 0.2 * partial) * 0.4

    if generic_only_overlap:
        score *= 0.35

    return round(score, 2), generic_only_overlap, ", ".join(shared_generic), ", ".join(shared_distinctive)


def address_similarity(a_addr: str, b_addr: str) -> Optional[float]:
    a_dist = " ".join(split_tokens(a_addr)[1])
    b_dist = " ".join(split_tokens(b_addr)[1])

    a_empty = len(a_dist) == 0
    b_empty = len(b_dist) == 0

    if a_empty and b_empty:
        return None
    if a_empty or b_empty:
        return 30.0
    if _is_error_message(a_addr) or _is_error_message(b_addr):
        return 0.0

    return round(float(fuzz.token_set_ratio(a_dist, b_dist)), 2)


def weighted_score(name: float, loc: float, addr: Optional[float], new_addr: Optional[float]) -> float:
    addr_score = 40.0 if addr is None else addr
    new_addr_score = 40.0 if new_addr is None else new_addr
    score = name * NAME_WEIGHT + loc * LOCATION_WEIGHT + addr_score * ADDRESS_WEIGHT + new_addr_score * NEW_ADDRESS_WEIGHT
    return round(score, 2)


# =========================
# Matching core
# =========================
def resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    lower = {re.sub(r"[\s_\-]+", "", str(c)).lower(): c for c in df.columns}

    def pick(cands: Sequence[str]) -> Optional[str]:
        for c in cands:
            k = re.sub(r"[\s_\-]+", "", c).lower()
            if k in lower:
                return lower[k]
        return None

    return {
        "code": pick(["customercode", "customer code", "custcode", "code", "customerid"]),
        "name": pick(["storename", "store_name", "store name", "name"]),
        "address": pick(["storeaddress", "store_address", "store address", "address"]),
        "new_address": pick(["newaddress", "new_address", "new address", "address_english"]),
        "lat": pick(["latitude", "lat"]),
        "lng": pick(["longitude", "lng", "lon", "long"]),
    }


def build_balltree(df: pd.DataFrame, latcol: str, lngcol: str) -> Tuple[Optional[BallTree], List[int]]:
    coords: List[Tuple[float, float]] = []
    idx_map: List[int] = []
    for i, row in df.iterrows():
        lat = safe_float(row.get(latcol))
        lng = safe_float(row.get(lngcol))
        if not is_valid_coordinate(lat, lng):
            continue
        coords.append((math.radians(lat), math.radians(lng)))
        idx_map.append(i)
    if not coords:
        return None, []
    return BallTree(np.array(coords), metric="haversine"), idx_map


def nearby_candidates(tree: Optional[BallTree], idx_map: List[int], lat: float, lng: float) -> List[int]:
    if tree is None or not is_valid_coordinate(lat, lng):
        return []
    q = np.array([[math.radians(lat), math.radians(lng)]])
    hits = tree.query_radius(q, r=CANDIDATE_RADIUS_M / 6371000.0)[0]
    if len(hits) == 0:
        return []
    if len(hits) > MAX_CANDIDATES:
        dists, inds = tree.query(q, k=MAX_CANDIDATES)
        return [idx_map[i] for i in inds[0].tolist()]
    return [idx_map[i] for i in hits.tolist()]


def classify_pair(name_score_val: float, dist_m: Optional[float], generic_only_overlap: bool) -> Tuple[Optional[str], str]:
    if dist_m is None:
        return None, "NO_DISTANCE"
    if dist_m > HARD_VETO_DISTANCE_M:
        return None, "VETO_DISTANCE"
    if name_score_val <= HARD_VETO_NAME_MIN:
        return None, "VETO_NAME"
    if generic_only_overlap:
        return None, "GENERIC_ONLY_NAME_BLOCKED"

    if dist_m <= CONFIRMED_DISTANCE_M and name_score_val >= CONFIRMED_NAME_THRESHOLD:
        return "✓ Confirmed", "CONFIRMED"
    if dist_m <= STRONG_POSSIBLE_DISTANCE_M and name_score_val >= STRONG_POSSIBLE_NAME_THRESHOLD:
        return "⭐ Strong Possible (Priority)", "STRONG_POSSIBLE"
    if dist_m <= POSSIBLE_DISTANCE_M and name_score_val >= POSSIBLE_NAME_THRESHOLD:
        return "⚠ Possible", "POSSIBLE"
    return None, "BELOW_THRESHOLD"


def greedy_one_to_one(candidates: List[Candidate]) -> List[Candidate]:
    # Deterministic high-score-first assignment
    ranked = sorted(candidates, key=lambda c: (c.weighted_score, c.name_score, -(c.distance_m or 1e9)), reverse=True)
    used_a: Set[int] = set()
    used_b: Set[int] = set()
    out: List[Candidate] = []
    for c in ranked:
        if c.a_idx in used_a or c.b_idx in used_b:
            continue
        used_a.add(c.a_idx)
        used_b.add(c.b_idx)
        out.append(c)
    return out


def run_phase(df_a: pd.DataFrame, df_b: pd.DataFrame, label: str) -> Tuple[List[Candidate], List[Candidate], List[Candidate], Dict[str, str], Dict[str, str]]:
    cols_a = resolve_columns(df_a)
    cols_b = resolve_columns(df_b)
    tree_b, idx_map_b = build_balltree(df_b, cols_b["lat"] or "", cols_b["lng"] or "")

    confirmed: List[Candidate] = []
    strong: List[Candidate] = []
    possible: List[Candidate] = []

    for a_idx, a_row in df_a.iterrows():
        lat_a = safe_float(a_row.get(cols_a["lat"] or ""))
        lng_a = safe_float(a_row.get(cols_a["lng"] or ""))
        b_candidates = nearby_candidates(tree_b, idx_map_b, lat_a, lng_a)
        if not b_candidates:
            continue

        for b_idx in b_candidates:
            b_row = df_b.iloc[b_idx]
            lat_b = safe_float(b_row.get(cols_b["lat"] or ""))
            lng_b = safe_float(b_row.get(cols_b["lng"] or ""))
            if not (is_valid_coordinate(lat_a, lng_a) and is_valid_coordinate(lat_b, lng_b)):
                continue

            n_score, generic_only, shared_generic, shared_distinctive = name_similarity(
                a_row.get(cols_a["name"] or "", ""), b_row.get(cols_b["name"] or "", "")
            )
            dist = haversine_m(lat_a, lng_a, lat_b, lng_b)
            cls, reason = classify_pair(n_score, dist, generic_only)
            if cls is None:
                continue

            addr_score = address_similarity(a_row.get(cols_a["address"] or "", ""), b_row.get(cols_b["address"] or "", ""))
            naddr_score = address_similarity(a_row.get(cols_a["new_address"] or "", ""), b_row.get(cols_b["new_address"] or "", ""))
            loc_score = location_score(dist)
            w_score = weighted_score(n_score, loc_score, addr_score, naddr_score)

            cand = Candidate(
                a_idx=a_idx,
                b_idx=b_idx,
                name_score=n_score,
                address_score=addr_score,
                new_address_score=naddr_score,
                location_score=loc_score,
                distance_m=round(dist, 2),
                weighted_score=w_score,
                confidence=cls,
                shared_generic_tokens=shared_generic,
                shared_distinctive_tokens=shared_distinctive,
                reason_code=reason,
            )

            if cls.startswith("✓"):
                confirmed.append(cand)
            elif cls.startswith("⭐"):
                strong.append(cand)
            else:
                possible.append(cand)

    return greedy_one_to_one(confirmed), greedy_one_to_one(strong), greedy_one_to_one(possible), cols_a, cols_b


def read_input_files() -> Dict[str, pd.DataFrame]:
    files_data: Dict[str, pd.DataFrame] = {}
    for name in ["File A", "File B", "File C", "File D", "File E"]:
        for ext in [".xlsx", ".csv"]:
            local = f"{name}{ext}"
            colab = f"/content/{name}{ext}"
            path = local if os.path.exists(local) else colab
            if os.path.exists(path):
                files_data[name] = pd.read_excel(path, dtype=str) if ext == ".xlsx" else pd.read_csv(path, dtype=str)
                break
    if files_data:
        return files_data

    try:
        from google.colab import files
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("No input files found and not running in Colab upload context.") from exc

    uploaded = files.upload()
    for filename in sorted(uploaded.keys()):
        bio = io.BytesIO(uploaded[filename])
        if filename.lower().endswith((".xlsx", ".xls")):
            files_data[filename] = pd.read_excel(bio, dtype=str)
        else:
            files_data[filename] = pd.read_csv(bio, dtype=str)
    return files_data


def build_output_rows(cands: List[Candidate], df_a: pd.DataFrame, df_b: pd.DataFrame, file_a: str, file_b: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for c in cands:
        a_row = df_a.iloc[c.a_idx]
        b_row = df_b.iloc[c.b_idx]
        row: Dict[str, object] = {f"A_{col}": a_row[col] for col in a_row.index}
        row.update({f"B_{col}": b_row[col] for col in b_row.index})
        row.update(
            {
                "Comparison": f"{file_a} → {file_b}",
                "Name Score (%)": c.name_score,
                "Address Score (%)": c.address_score,
                "New Address Score (%)": c.new_address_score,
                "Location Score (%)": c.location_score,
                "Distance (m)": c.distance_m,
                "Weighted Score (%)": c.weighted_score,
                "Shared_Generic_Tokens": c.shared_generic_tokens,
                "Shared_Distinctive_Tokens": c.shared_distinctive_tokens,
                "Reason_Code": c.reason_code,
                "Confidence": c.confidence,
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    files_data = read_input_files()
    if len(files_data) < 2:
        raise RuntimeError("Need at least 2 files.")

    file_names = sorted(files_data.keys())
    all_confirmed: List[Dict[str, object]] = []
    all_strong: List[Dict[str, object]] = []
    all_possible: List[Dict[str, object]] = []
    matched_store_ids: Set[Tuple[str, int]] = set()

    for i, file_a in enumerate(file_names):
        for j, file_b in enumerate(file_names):
            if i == j:
                continue
            df_a = files_data[file_a].copy()
            df_b = files_data[file_b].copy()
            confirmed, strong, possible, _, _ = run_phase(df_a, df_b, f"{file_a}→{file_b}")

            all_confirmed.extend(build_output_rows(confirmed, df_a, df_b, file_a, file_b))
            all_strong.extend(build_output_rows(strong, df_a, df_b, file_a, file_b))
            all_possible.extend(build_output_rows(possible, df_a, df_b, file_a, file_b))

            for c in confirmed + strong + possible:
                matched_store_ids.add((file_a, c.a_idx))
                matched_store_ids.add((file_b, c.b_idx))

    unique_rows: List[Dict[str, object]] = []
    for file_name in file_names:
        df = files_data[file_name]
        for idx, row in df.iterrows():
            if (file_name, idx) in matched_store_ids:
                continue
            rec = {col: row[col] for col in row.index}
            rec["Source_File"] = file_name
            rec["Status"] = f"✗ Unique to {file_name}"
            unique_rows.append(rec)

    default_out = "/content/improved_bidirectional_matching.xlsx" if os.path.isdir("/content") else "improved_bidirectional_matching.xlsx"
    out_path = os.environ.get("MATCH_OUTPUT_PATH", default_out)
    out_path = str(Path(out_path).expanduser().resolve())

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "Metric": [
                    "Generated",
                    "Files",
                    "Confirmed Pairs",
                    "Strong Possible Pairs",
                    "Possible Pairs",
                    "Unique Stores",
                    "Name Weight",
                    "Location Weight",
                    "Address Weight",
                    "New Address Weight",
                    "Confirmed Distance Gate",
                ],
                "Value": [
                    datetime.now().isoformat(timespec="seconds"),
                    ", ".join(file_names),
                    len(all_confirmed),
                    len(all_strong),
                    len(all_possible),
                    len(unique_rows),
                    NAME_WEIGHT,
                    LOCATION_WEIGHT,
                    ADDRESS_WEIGHT,
                    NEW_ADDRESS_WEIGHT,
                    f"<= {CONFIRMED_DISTANCE_M}m",
                ],
            }
        ).to_excel(writer, index=False, sheet_name="Summary")
        pd.DataFrame(all_confirmed).to_excel(writer, index=False, sheet_name="Confirmed_Matches")
        pd.DataFrame(all_strong).to_excel(writer, index=False, sheet_name="Strong_Possible_Matches")
        pd.DataFrame(all_possible).to_excel(writer, index=False, sheet_name="Possible_Matches")
        pd.DataFrame(unique_rows).to_excel(writer, index=False, sheet_name="Unique_Stores")

    print(f"Saved: {out_path}")

    # Auto-download when running in Colab so users can access the file immediately.
    try:
        from google.colab import files as colab_files

        colab_files.download(out_path)
        print("Download started in Colab.")
    except Exception:
        print("If running locally, open the saved path above.")


if __name__ == "__main__":
    main()
