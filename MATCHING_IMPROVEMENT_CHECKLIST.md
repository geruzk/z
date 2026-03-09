# Technical Risk & Improvement Checklist for Store Matching (A/B/C)

## Quick audit findings from your current files

A lightweight random/sample audit on `File A.xlsx`, `File B.xlsx`, and `File C.xlsx` shows several high-risk false-positive patterns:

- Very frequent generic retail tokens are dominating name similarity, especially:
  - `ماركت` (2523 occurrences)
  - `بازار` (1902)
  - `سوبر`, `هايبر`, `بقالة/بقاله`, `اسواق`.
- Many records have missing native address (`store_address`) but have geocoded `Address_English`, which can bias scoring.
- A coordinate-only window can still produce wrong confirmed matches in dense areas.

### Sample suspicious pairs found within 30m

These examples illustrate why name rules need stronger anti-generic logic:

- `الفتح ماركت` (A) matched near `الرضا بقاله` (B), distance ~1.2m, no meaningful name overlap.
- `ماركت الفرسان` (A) near `ساندى ماركت` (B), overlap only on generic token `ماركت`.
- `بازار btc` (A) near `بازار القصر` (B), overlap only on generic token `بازار`.

---

## Priority risk checklist (what is likely causing bad outcomes)

## 1) Name false positives from generic words (HIGH)

- **Risk:** `token_set_ratio` rewards shared generic words too much.
- **Impact:** Different stores get Confirmed (e.g., `Market X` vs `Market Y`).
- **Fixes:**
  - Build an **Arabic retail stopword lexicon** and remove or strongly downweight those tokens.
  - Add **distinctive-token rule**: at least one non-generic token overlap (or high char similarity on non-generic core).
  - Penalize if only generic overlap exists.

Suggested Arabic generic list to start with:

`ماركت, سوبر, هايبر, ميني, بقالة, بقاله, بازار, اسواق, ستور, محمصه, محمصة, عطارة, عطاره, كشك, سنتر`

---

## 2) Location over-trust in dense areas (HIGH)

- **Risk:** Very close stores can still be different entities in malls/streets.
- **Impact:** Wrong confirmed pairs when distance is tiny.
- **Requested update implemented in logic design:** use **strict ±30m matching window** for confirmed proximity checks.
- **Fixes:**
  - Set confirmed distance gate to `<= 30m`.
  - Treat `30-60m` as review (not confirmed).
  - Reject `>100m` except exception workflow.
  - Keep 200m hard veto for safety.

---

## 3) Address signal quality issues (MEDIUM-HIGH)

- **Risk:** Empty/failed addresses distort scoring (`None -> neutral`) and can inflate match confidence.
- **Fixes:**
  - Use same stopword/normalization logic on Arabic address components.
  - Add address-quality flags (empty/error/short/only-digits).
  - If both addresses are low quality, cap address contribution rather than defaulting to high/neutral.

---

## 4) One-to-many duplicate matching (HIGH)

- **Risk:** Multiple candidate links survive and appear duplicated.
- **Fixes:**
  - Use **global bipartite assignment** (Hungarian / max-weight matching) for final unique pairing between each file pair.
  - Keep your existing “best one per store” as a fallback, but global assignment is cleaner and reduces collisions.

---

## 5) Threshold calibration not data-driven (MEDIUM)

- **Risk:** Fixed thresholds may not fit region/store-type variation.
- **Fixes:**
  - Build a reviewed truth set (200–500 pairs).
  - Sweep thresholds and optimize precision first (to reduce false positives).
  - Track precision/recall by segment (Arabic-only names, Latin names, missing address, dense urban clusters).

---

## 6) Explainability for reviewer trust (MEDIUM)

- **Risk:** Users cannot tell *why* two stores matched.
- **Fixes:**
  - Output explanation columns:
    - `Shared_Generic_Tokens`
    - `Shared_Distinctive_Tokens`
    - `Distinctive_Name_Score`
    - `Address_Quality_A/B`
    - `Reason_Code` (e.g., `LOC_STRONG_NAME_WEAK`, `GENERIC_ONLY_NAME_BLOCKED`).

---

## Recommended scoring changes (your requested weights)

Use this weighted average:

- **Name:** 15%
- **Location:** 60%
- **Address:** 15%
- **New Address:** 10%

And pair this with anti-false-positive guards:

1. Confirmed requires all:
   - distance `<= 30m`
   - distinctive name score above threshold
   - NOT generic-only name overlap
2. Strong Possible:
   - distance `<= 60m` with medium distinctive name score
3. Possible:
   - distance `<= 100m` and moderate total score

---

## Concrete implementation plan (recommended order)

1. **Token pipeline update (Name + Address):**
   - normalize Arabic letters/diacritics
   - split tokens
   - remove generic stopwords
   - compute two scores:
     - generic-inclusive (diagnostic)
     - distinctive-only (decision score)

2. **Rule gate before weighted score:**
   - if no distinctive tokens on both sides and only generic overlap -> block from Confirmed.

3. **Distance gating changes:**
   - confirmed `<= 30m`
   - review `30-60m`
   - possible `60-100m`

4. **Final dedup upgrade:**
   - run bipartite max-weight assignment per pairwise file comparison.

5. **QA dashboard sheet:**
   - add confusion buckets and top suspicious matches for manual review.

---

## Practical “random-check” review workflow you can run each batch

- Sample 50 rows each from Confirmed / Strong Possible / Possible.
- Manually label true/false quickly.
- Track:
  - Precision@Confirmed
  - % matches with generic-only overlap
  - % matches with empty address on one or both sides
  - Duplicates per source store
- Adjust thresholds weekly until Confirmed precision is acceptable.

---

## Expected improvements if applied

- Significant drop in “Market X vs Market Y” false confirmed matches.
- Fewer duplicate links due to stronger assignment constraints.
- Cleaner separation between matched and unique stores.
- Better auditability for business users.


---

## How to validate the output now (quick checklist)

1. **Run the script and check the console coverage line**
   - Expect: `Coverage: PASS` and `MatchedStores + Unique == Total Input`.
2. **Check the `Summary` sheet sanity rows**
   - `Coverage Check (matched+unique==input)` should be `PASS`.
   - `Per-File Split Sanity` should be `PASS`.
3. **Audit a small human sample from each segment**
   - Review 30 rows from `Confirmed_Matches`.
   - Review 30 rows from `Possible_Matches`.
   - Review 30 rows from `Unique_Stores`.
4. **Watch reason code drift**
   - If `POSSIBLE_DIST<=30_AND_NEWADDR` dominates too much, review those first.

Recommended run command:

```bash
MATCH_STAGE_MODE=all MATCH_OUTPUT_PATH=./improved_bidirectional_matching.xlsx python improved_store_matching.py
```

## How to increase segmentation counts **without changing thresholds**

You can increase candidate coverage while keeping thresholds unchanged:

1. **Increase candidate breadth per store** with `MAX_CANDIDATES` (default `60`).
   - Example:

```bash
MAX_CANDIDATES=120 MATCH_STAGE_MODE=all python improved_store_matching.py
```

2. **Keep full pipeline mode** (`MATCH_STAGE_MODE=all`) so BC-stage and direct matching both contribute.
3. **Rotate base file and compare unioned results** (A as base, then B as base, etc.) to recover asymmetric misses.
4. **Improve source text completeness** (store name/address quality) before matching; this boosts segmentation naturally without threshold edits.

## Output tabs control

By default, output stops at:
- `Summary`
- `Confirmed_Matches`
- `Possible_Matches`
- `Unique_Stores`

If you still need BC diagnostic tabs, enable:

```bash
MATCH_INCLUDE_EXTRA_TABS=1 python improved_store_matching.py
```
