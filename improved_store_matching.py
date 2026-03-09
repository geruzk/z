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
from difflib import SequenceMatcher
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


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio() * 100.0


def _token_set_ratio(a: str, b: str) -> float:
    a_tokens = set(a.split()) if a else set()
    b_tokens = set(b.split()) if b else set()
    if not a_tokens and not b_tokens:
        return 100.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    a_only = a_tokens - inter
    b_only = b_tokens - inter

    def join_tokens(tokens):
        return " ".join(sorted(tokens)).strip()

    s_inter = join_tokens(inter)
    s_a = join_tokens(inter | a_only)
    s_b = join_tokens(inter | b_only)

    # Similar spirit to token-set behavior
    return max(_ratio(s_inter, s_a), _ratio(s_inter, s_b), _ratio(s_a, s_b))


def _partial_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0

    # Slide the shorter string over the longer one and pick best local ratio
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    window = len(short)
    if window == 0:
        return 0.0

    best = 0.0
    for i in range(0, len(long_) - window + 1):
        cand = long_[i : i + window]
        best = max(best, _ratio(short, cand))
        if best >= 100.0:
            break
    return best


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
        token_set = _token_set_ratio(a_text, b_text)
        ratio = _ratio(a_text, b_text)
        partial = _partial_ratio(a_text, b_text)
        score = 0.5 * token_set + 0.3 * ratio + 0.2 * partial
    else:
        # If no distinctive tokens, fall back to full name with penalty.
        token_set = _token_set_ratio(normalize_text(a_name), normalize_text(b_name))
        ratio = _ratio(normalize_text(a_name), normalize_text(b_name))
        partial = _partial_ratio(normalize_text(a_name), normalize_text(b_name))
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

    return round(float(_token_set_ratio(a_dist, b_dist)), 2)


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
        row: Dict[str, object] = {
            "A_File": file_a,
            "A_Idx": int(c.a_idx),
            "B_File": file_b,
            "B_Idx": int(c.b_idx),
        }
        row.update({f"A_{col}": a_row[col] for col in a_row.index})
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


def dedupe_match_rows(rows: List[Dict[str, object]], priority: int) -> Dict[Tuple[Tuple[str, int], Tuple[str, int]], Dict[str, object]]:
    deduped: Dict[Tuple[Tuple[str, int], Tuple[str, int]], Dict[str, object]] = {}
    for r in rows:
        a = (str(r["A_File"]), int(r["A_Idx"]))
        b = (str(r["B_File"]), int(r["B_Idx"]))
        key = tuple(sorted([a, b]))
        if key not in deduped:
            rr = dict(r)
            rr["_priority"] = priority
            deduped[key] = rr
            continue

        old = deduped[key]
        old_pr = int(old.get("_priority", 99))
        if priority < old_pr:
            rr = dict(r)
            rr["_priority"] = priority
            deduped[key] = rr
        elif priority == old_pr and float(r.get("Weighted Score (%)", 0) or 0) > float(old.get("Weighted Score (%)", 0) or 0):
            rr = dict(r)
            rr["_priority"] = priority
            deduped[key] = rr
    return deduped


def one_to_one_global(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked = sorted(rows, key=lambda r: (float(r.get("Weighted Score (%)", 0) or 0), float(r.get("Name Score (%)", 0) or 0), -float(r.get("Distance (m)", 1e9) or 1e9)), reverse=True)
    used: Set[Tuple[str, int]] = set()
    out: List[Dict[str, object]] = []
    for r in ranked:
        a = (str(r["A_File"]), int(r["A_Idx"]))
        b = (str(r["B_File"]), int(r["B_Idx"]))
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        out.append(r)
    return out



def apply_excel_colors(path: str) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font

    wb = load_workbook(path)

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    confirmed_fill = PatternFill("solid", fgColor="C6EFCE")
    possible_fill = PatternFill("solid", fgColor="FFF2CC")
    unique_fill = PatternFill("solid", fgColor="E7E6E6")

    for ws in wb.worksheets:
        if ws.max_row == 0:
            continue

        if ws.title == "Summary":
            # Dashboard header row
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
            # Dashboard first column labels and body highlight (requested matrix look)
            summary_accent = PatternFill("solid", fgColor="8EC3D3")
            for r in range(2, 8):
                for c in range(1, ws.max_column + 1):
                    ws.cell(r, c).fill = summary_accent
            continue

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font

        fill = None
        if ws.title == "Confirmed_Matches":
            fill = confirmed_fill
        elif ws.title == "Possible_Matches":
            fill = possible_fill
        elif ws.title == "Unique_Stores":
            fill = unique_fill

        if fill is not None:
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
                for c in row:
                    c.fill = fill

    wb.save(path)


def store_output_row(store_id: Tuple[str, int], files_data: Dict[str, pd.DataFrame], tab_status: str, matched: bool, partner: Optional[Tuple[str, int]], reason: str) -> Dict[str, object]:
    file_name, idx = store_id
    row = files_data[file_name].iloc[idx]
    out: Dict[str, object] = {col: row[col] for col in row.index}
    out["Source_File"] = file_name
    out["Source_Idx"] = int(idx)
    out["Status"] = tab_status
    out["Matched"] = "Yes" if matched else "No"
    out["Reason_Code"] = reason
    if partner is None:
        out["Partner_File"] = ""
        out["Partner_Idx"] = ""
    else:
        out["Partner_File"] = partner[0]
        out["Partner_Idx"] = int(partner[1])
    return out


def build_possible_pairs_from_unmatched(files_data: Dict[str, pd.DataFrame], unmatched: Set[Tuple[str, int]], new_addr_threshold: float = 60.0) -> List[Tuple[Tuple[str, int], Tuple[str, int], float, float]]:
    """Find extra possible pairs for unmatched stores:
    distance <= 30m and new address similarity >= threshold, even if name does not match.
    Returns one-to-one pairs as (a_id, b_id, new_addr_score, distance_m).
    """
    file_names = sorted(files_data.keys())
    candidates: List[Tuple[Tuple[str, int], Tuple[str, int], float, float]] = []

    for i, file_a in enumerate(file_names):
        df_a = files_data[file_a]
        cols_a = resolve_columns(df_a)
        for j, file_b in enumerate(file_names):
            if i >= j:
                continue
            df_b = files_data[file_b]
            cols_b = resolve_columns(df_b)

            tree_b, idx_map_b = build_balltree(df_b, cols_b["lat"] or "", cols_b["lng"] or "")
            if tree_b is None:
                continue

            for a_idx, a_row in df_a.iterrows():
                a_id = (file_a, int(a_idx))
                if a_id not in unmatched:
                    continue

                lat_a = safe_float(a_row.get(cols_a["lat"] or ""))
                lng_a = safe_float(a_row.get(cols_a["lng"] or ""))
                if not is_valid_coordinate(lat_a, lng_a):
                    continue

                b_idx_list = nearby_candidates(tree_b, idx_map_b, lat_a, lng_a)
                for b_idx in b_idx_list:
                    b_id = (file_b, int(b_idx))
                    if b_id not in unmatched:
                        continue

                    b_row = df_b.iloc[b_idx]
                    lat_b = safe_float(b_row.get(cols_b["lat"] or ""))
                    lng_b = safe_float(b_row.get(cols_b["lng"] or ""))
                    if not is_valid_coordinate(lat_b, lng_b):
                        continue

                    dist = haversine_m(lat_a, lng_a, lat_b, lng_b)
                    if dist > CONFIRMED_DISTANCE_M:
                        continue

                    naddr = address_similarity(a_row.get(cols_a["new_address"] or "", ""), b_row.get(cols_b["new_address"] or "", ""))
                    naddr_score = 0.0 if naddr is None else float(naddr)
                    if naddr_score < new_addr_threshold:
                        continue

                    candidates.append((a_id, b_id, naddr_score, float(dist)))

    # One-to-one greedy on strongest new-address then closest distance
    candidates.sort(key=lambda x: (x[2], -x[3]), reverse=True)
    used: Set[Tuple[str, int]] = set()
    final_pairs: List[Tuple[Tuple[str, int], Tuple[str, int], float, float]] = []
    for a_id, b_id, s, d in candidates:
        if a_id in used or b_id in used:
            continue
        used.add(a_id)
        used.add(b_id)
        final_pairs.append((a_id, b_id, s, d))
    return final_pairs


def _build_store_level_rows_from_pair_rows(pair_rows: List[Dict[str, object]], files_data: Dict[str, pd.DataFrame], tab_status: str) -> List[Dict[str, object]]:
    """Convert pair rows into store-level rows (each store appears once max in a tab)."""
    out: List[Dict[str, object]] = []
    seen: Set[Tuple[str, int]] = set()

    for pr in pair_rows:
        a_file, a_idx = str(pr["A_File"]), int(pr["A_Idx"])
        b_file, b_idx = str(pr["B_File"]), int(pr["B_Idx"])

        # Row for A store
        a_id = (a_file, a_idx)
        if a_id not in seen:
            src = files_data[a_file].iloc[a_idx]
            rec_a: Dict[str, object] = {col: src[col] for col in src.index}
            rec_a.update(
                {
                    "Source_File": a_file,
                    "Source_Idx": a_idx,
                    "Matched_With_File": b_file,
                    "Matched_With_Idx": b_idx,
                    "Matched_With_Store_Name": pr.get("B_store_name", ""),
                    "Matched_With_Customer_Code": pr.get("B_Customer Code", ""),
                    "Comparison": pr.get("Comparison", ""),
                    "Status": tab_status,
                    "Name Score (%)": pr.get("Name Score (%)"),
                    "Address Score (%)": pr.get("Address Score (%)"),
                    "New Address Score (%)": pr.get("New Address Score (%)"),
                    "Location Score (%)": pr.get("Location Score (%)"),
                    "Distance (m)": pr.get("Distance (m)"),
                    "Weighted Score (%)": pr.get("Weighted Score (%)"),
                    "Shared_Generic_Tokens": pr.get("Shared_Generic_Tokens", ""),
                    "Shared_Distinctive_Tokens": pr.get("Shared_Distinctive_Tokens", ""),
                    "Reason_Code": pr.get("Reason_Code", ""),
                    "Confidence": pr.get("Confidence", ""),
                }
            )
            out.append(rec_a)
            seen.add(a_id)

        # Row for B store
        b_id = (b_file, b_idx)
        if b_id not in seen:
            src = files_data[b_file].iloc[b_idx]
            rec_b: Dict[str, object] = {col: src[col] for col in src.index}
            rec_b.update(
                {
                    "Source_File": b_file,
                    "Source_Idx": b_idx,
                    "Matched_With_File": a_file,
                    "Matched_With_Idx": a_idx,
                    "Matched_With_Store_Name": pr.get("A_store_name", ""),
                    "Matched_With_Customer_Code": pr.get("A_Customer Code", ""),
                    "Comparison": pr.get("Comparison", ""),
                    "Status": tab_status,
                    "Name Score (%)": pr.get("Name Score (%)"),
                    "Address Score (%)": pr.get("Address Score (%)"),
                    "New Address Score (%)": pr.get("New Address Score (%)"),
                    "Location Score (%)": pr.get("Location Score (%)"),
                    "Distance (m)": pr.get("Distance (m)"),
                    "Weighted Score (%)": pr.get("Weighted Score (%)"),
                    "Shared_Generic_Tokens": pr.get("Shared_Generic_Tokens", ""),
                    "Shared_Distinctive_Tokens": pr.get("Shared_Distinctive_Tokens", ""),
                    "Reason_Code": pr.get("Reason_Code", ""),
                    "Confidence": pr.get("Confidence", ""),
                }
            )
            out.append(rec_b)
            seen.add(b_id)

    return out




def _present_pair_rows(pair_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Make pair output headers explicit and consistent (Store_1 / Store_2)."""
    out: List[Dict[str, object]] = []

    for r in pair_rows:
        a_file = str(r.get("A_File", ""))
        a_idx = int(r.get("A_Idx", -1))
        b_file = str(r.get("B_File", ""))
        b_idx = int(r.get("B_Idx", -1))

        swap = (a_file, a_idx) > (b_file, b_idx)

        rec: Dict[str, object] = {}

        # Keep non A_/B_ fields first
        for k, v in r.items():
            if not k.startswith("A_") and not k.startswith("B_"):
                rec[k] = v

        # Then map store sides with clear naming
        for k, v in r.items():
            if k.startswith("A_"):
                suffix = k[len("A_") :]
                target = "Store_2_" + suffix if swap else "Store_1_" + suffix
                rec[target] = v
            elif k.startswith("B_"):
                suffix = k[len("B_") :]
                target = "Store_1_" + suffix if swap else "Store_2_" + suffix
                rec[target] = v

        rec["Pair_Store_1_ID"] = f"{rec.get('Store_1_File','')}:{rec.get('Store_1_Idx','')}"
        rec["Pair_Store_2_ID"] = f"{rec.get('Store_2_File','')}:{rec.get('Store_2_Idx','')}"
        rec["Pair_Label"] = f"{rec.get('Store_1_File','')} ↔ {rec.get('Store_2_File','')}"

        out.append(rec)

    return out

def main() -> None:
    files_data = read_input_files()
    if len(files_data) < 2:
        raise RuntimeError("Need at least 2 files.")

    file_names = sorted(files_data.keys())

    all_store_ids: Set[Tuple[str, int]] = set()
    for fn in file_names:
        for idx in files_data[fn].index:
            all_store_ids.add((fn, int(idx)))
    total_input_stores = len(all_store_ids)

    raw_confirmed: List[Dict[str, object]] = []
    raw_strong: List[Dict[str, object]] = []
    raw_possible: List[Dict[str, object]] = []

    for i, file_a in enumerate(file_names):
        for j, file_b in enumerate(file_names):
            if i == j:
                continue
            df_a = files_data[file_a].copy()
            df_b = files_data[file_b].copy()
            confirmed, strong, possible, _, _ = run_phase(df_a, df_b, f"{file_a}→{file_b}")

            raw_confirmed.extend(build_output_rows(confirmed, df_a, df_b, file_a, file_b))
            raw_strong.extend(build_output_rows(strong, df_a, df_b, file_a, file_b))
            raw_possible.extend(build_output_rows(possible, df_a, df_b, file_a, file_b))

    # Track stores that matched in ANY iteration (before final dedupe/selection)
    all_iteration_matched_ids: Set[Tuple[str, int]] = set()
    for rr in raw_confirmed + raw_strong + raw_possible:
        all_iteration_matched_ids.add((str(rr["A_File"]), int(rr["A_Idx"])))
        all_iteration_matched_ids.add((str(rr["B_File"]), int(rr["B_Idx"])))

    # Normalize A<->B duplicates, then priority Confirmed > Strong > Possible, then one-to-one
    merged: Dict[Tuple[Tuple[str, int], Tuple[str, int]], Dict[str, object]] = {}
    for key, row in dedupe_match_rows(raw_possible, priority=3).items():
        merged[key] = row
    for key, row in dedupe_match_rows(raw_strong, priority=2).items():
        old = merged.get(key)
        if old is None or row["_priority"] < old["_priority"] or (row["_priority"] == old["_priority"] and float(row.get("Weighted Score (%)", 0) or 0) > float(old.get("Weighted Score (%)", 0) or 0)):
            merged[key] = row
    for key, row in dedupe_match_rows(raw_confirmed, priority=1).items():
        old = merged.get(key)
        if old is None or row["_priority"] < old["_priority"] or (row["_priority"] == old["_priority"] and float(row.get("Weighted Score (%)", 0) or 0) > float(old.get("Weighted Score (%)", 0) or 0)):
            merged[key] = row

    confirmed_pair_rows = one_to_one_global(list(merged.values()))
    for r in confirmed_pair_rows:
        r.pop("_priority", None)

    confirmed_store_ids: Set[Tuple[str, int]] = set()
    for r in confirmed_pair_rows:
        confirmed_store_ids.add((str(r["A_File"]), int(r["A_Idx"])))
        confirmed_store_ids.add((str(r["B_File"]), int(r["B_Idx"])))

    # Possible fallback (unmatched only): distance <=30m + new address similarity
    unmatched_after_confirmed = all_store_ids - confirmed_store_ids
    possible_pairs = build_possible_pairs_from_unmatched(files_data, unmatched_after_confirmed, new_addr_threshold=60.0)

    possible_pair_rows: List[Dict[str, object]] = []
    possible_store_ids: Set[Tuple[str, int]] = set()
    for a_id, b_id, naddr_score, dist_m in possible_pairs:
        file_a, idx_a = a_id
        file_b, idx_b = b_id
        row_a = files_data[file_a].iloc[idx_a]
        row_b = files_data[file_b].iloc[idx_b]

        name_score, generic_only, shared_generic, shared_distinctive = name_similarity(
            row_a.get("store_name", ""), row_b.get("store_name", "")
        )
        addr_score = address_similarity(row_a.get("store_address", ""), row_b.get("store_address", ""))
        new_addr_score = address_similarity(row_a.get("Address_English", ""), row_b.get("Address_English", ""))
        loc_score = location_score(dist_m)
        w_score = weighted_score(name_score, loc_score, addr_score, new_addr_score)

        rec: Dict[str, object] = {
            "A_File": file_a,
            "A_Idx": int(idx_a),
            "B_File": file_b,
            "B_Idx": int(idx_b),
        }
        rec.update({f"A_{col}": row_a[col] for col in row_a.index})
        rec.update({f"B_{col}": row_b[col] for col in row_b.index})
        rec.update(
            {
                "Comparison": f"{file_a} ↔ {file_b}",
                "Name Score (%)": round(name_score, 2),
                "Address Score (%)": addr_score,
                "New Address Score (%)": round(new_addr_score, 2) if new_addr_score is not None else None,
                "Location Score (%)": round(loc_score, 2),
                "Distance (m)": round(dist_m, 2),
                "Weighted Score (%)": round(w_score, 2),
                "Shared_Generic_Tokens": shared_generic,
                "Shared_Distinctive_Tokens": shared_distinctive,
                "Reason_Code": "POSSIBLE_DIST<=30_AND_NEWADDR",
                "Confidence": "⚠ Possible (Distance+NewAddress)",
                "Generic_Only_Overlap": "Yes" if generic_only else "No",
            }
        )
        possible_pair_rows.append(rec)
        possible_store_ids.add(a_id)
        possible_store_ids.add(b_id)

    # Unique means not matched in any iteration, and not in final possible fallback set
    unique_store_ids = all_store_ids - all_iteration_matched_ids - possible_store_ids
    unique_rows: List[Dict[str, object]] = []
    for file_name, idx in sorted(unique_store_ids):
        row = files_data[file_name].iloc[idx]
        rec = {col: row[col] for col in row.index}
        rec.update(
            {
                "Source_File": file_name,
                "Source_Idx": int(idx),
                "Status": "✗ Unique (No condition met)",
                "Reason_Code": "NO_RULE_MATCH",
            }
        )
        unique_rows.append(rec)

    matched_store_count = len(all_store_ids - unique_store_ids)
    coverage_status = "PASS" if (matched_store_count + len(unique_rows) == total_input_stores) else "FAIL"

    default_out = "/content/improved_bidirectional_matching.xlsx" if os.path.isdir("/content") else "improved_bidirectional_matching.xlsx"
    out_path = os.environ.get("MATCH_OUTPUT_PATH", default_out)
    out_path = str(Path(out_path).expanduser().resolve())

    # Build dashboard table (requested format) for non-base files vs totals
    base_file = "File A" if "File A" in file_names else file_names[0]
    compare_files = [f for f in file_names if f != base_file]

    unique_by_file: Dict[str, int] = {f: 0 for f in compare_files}
    for r in unique_rows:
        sf = str(r.get("Source_File", ""))
        if sf in unique_by_file:
            unique_by_file[sf] += 1

    confirmed_by_file: Dict[str, int] = {f: 0 for f in compare_files}
    for f in compare_files:
        confirmed_by_file[f] = sum(1 for ff, _ in confirmed_store_ids if ff == f)

    possible_by_file: Dict[str, int] = {f: 0 for f in compare_files}
    for f in compare_files:
        possible_by_file[f] = sum(1 for ff, _ in possible_store_ids if ff == f)

    total_by_file: Dict[str, int] = {f: int(len(files_data[f])) for f in compare_files}

    def pct(n: int, d: int) -> str:
        return f"{round((n / d) * 100):.0f}%" if d else "0%"

    dashboard_rows = [
        "Confirmed",
        "Possible",
        "Unique",
        "Total",
        "Confirmed %",
        "Confirmed & possible %",
    ]
    dashboard_data: Dict[str, List[object]] = {}
    for f in compare_files:
        c = confirmed_by_file.get(f, 0)
        p = possible_by_file.get(f, 0)
        u = unique_by_file.get(f, 0)
        t = total_by_file.get(f, 0)
        dashboard_data[f] = [c, p, u, t, pct(c, t), pct(c + p, t)]

    # Total column
    c_tot = sum(confirmed_by_file.values())
    p_tot = sum(possible_by_file.values())
    u_tot = sum(unique_by_file.values())
    t_tot = sum(total_by_file.values())
    dashboard_data["Total"] = [
        c_tot,
        p_tot,
        u_tot,
        t_tot,
        pct(c_tot, t_tot),
        pct(c_tot + p_tot, t_tot),
    ]

    dashboard_df = pd.DataFrame(dashboard_data, index=dashboard_rows).reset_index().rename(columns={"index": "Metric"})

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        dashboard_df.to_excel(writer, index=False, sheet_name="Summary", startrow=0)

        # Keep detailed run metadata below dashboard table
        pd.DataFrame(
            {
                "Metric": [
                    "Generated",
                    "Base File",
                    "Compared Files",
                    "Total Input Stores",
                    "Confirmed Pair Rows",
                    "Possible Pair Rows",
                    "Unique Stores",
                    "Matched Stores (counted once)",
                    "Coverage Check (matched+unique==input)",
                    "Matched Stores in Any Iteration",
                    "Name Weight",
                    "Location Weight",
                    "Address Weight",
                    "New Address Weight",
                    "Confirmed Distance Gate",
                    "Possible Extra Rule",
                ],
                "Value": [
                    datetime.now().isoformat(timespec="seconds"),
                    base_file,
                    ", ".join(compare_files),
                    total_input_stores,
                    len(confirmed_pair_rows),
                    len(possible_pair_rows),
                    len(unique_rows),
                    matched_store_count,
                    coverage_status,
                    len(all_iteration_matched_ids),
                    NAME_WEIGHT,
                    LOCATION_WEIGHT,
                    ADDRESS_WEIGHT,
                    NEW_ADDRESS_WEIGHT,
                    f"<= {CONFIRMED_DISTANCE_M}m",
                    "Distance<=30m and New Address similarity>=60 even without name match",
                ],
            }
        ).to_excel(writer, index=False, sheet_name="Summary", startrow=len(dashboard_rows) + 3)

        confirmed_output_rows = _present_pair_rows(confirmed_pair_rows)
        possible_output_rows = _present_pair_rows(possible_pair_rows)

        pd.DataFrame(confirmed_output_rows).to_excel(writer, index=False, sheet_name="Confirmed_Matches")
        pd.DataFrame(possible_output_rows).to_excel(writer, index=False, sheet_name="Possible_Matches")
        pd.DataFrame(unique_rows).to_excel(writer, index=False, sheet_name="Unique_Stores")

    apply_excel_colors(out_path)

    print(f"Saved: {out_path}")
    print(
        f"Coverage: {coverage_status} | Input={total_input_stores}, MatchedStores={matched_store_count}, Unique={len(unique_rows)}"
    )

    try:
        from google.colab import files as colab_files

        colab_files.download(out_path)
        print("Download started in Colab.")
    except Exception:
        print("If running locally, open the saved path above.")


if __name__ == "__main__":
    main()
