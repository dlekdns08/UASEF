# Camera-Ready Path — Physician Audit + eICU LLM Completion

> **Purpose.** Two blocking items between ML4H Findings accept and
> camera-ready submission:
>   1. **Physician audit** (R10.5) — construct validity of
>      outcome-derived stratum labels via 3-physician Cohen's $\kappa$
>   2. **eICU LLM Pass A/B** (R11.3 LLM) — LLM cross-cohort
>      validation, currently tabular-only
>
> Both are *deferred* in `UASEF_ML4H_2026.md` §4.3 and can be
> executed independently in parallel.

**Timeline.** Assuming ML4H notification 2026-08-15, camera-ready
2026-09-15: 4 weeks. Both items fit within that window if started at
notification.

---

## 1. Physician Audit — R10.5

### 1.1 What is shipped

Location: `paper/irb_audit_package/`

| File | Purpose | Physician-facing |
|---|---|---|
| `instructions.md` (2,134 B) | Adjudication protocol | ✓ |
| `case_template.jsonl` (380 B) | Empty response template | ✓ |
| `example_filled.jsonl` (158 B) | Worked example | ✓ |
| `r10_5_audit_blinded.jsonl` (→ `data/raw/audit_r10_5/`) | 100 stratified cases (50 CRITICAL + 50 HIGH), decision-time features only | ✓ |
| `data/raw/audit_r10_5/r10_5_audit_ground_truth.csv` | Held out (NOT sent to physicians) | ✗ |

### 1.2 What each physician receives

**Physician-facing package** (email/secure-transfer):

```
paper/irb_audit_package/
├── instructions.md
├── case_template.jsonl              (blank)
├── example_filled.jsonl             (worked example)
└── r10_5_audit_blinded.jsonl        (100 cases — no outcome labels)
```

Each physician independently fills `case_template.jsonl` with:
```json
{"case_id": "<hadm_id>", "physician_id": "p1|p2|p3",
 "label": "escalate|no_escalate", "rationale": "1-sentence"}
```

### 1.3 Execution sequence

**Week 1 (notification → +1w)**: IRB submission

- File an "Exempt / Not Human Subjects Research" IRB application
  (retrospective de-identified MIMIC-IV data; outcome labels held
  out of physician view)
- Recruit 3 board-certified physicians (target: emergency medicine,
  internal medicine, family medicine — for inter-rater diversity)
- Budget: USD 80/hour × ~10 hours × 3 physicians = **$2,400**

**Week 2 (+1w → +2w)**: Package distribution + adjudication

- Send package to each physician via secure channel
  (encrypted email / institutional file-share; MIMIC-IV DUA
  disallows re-distribution but blinded-case adjudication under
  active credentialing is permitted per PhysioNet §5.2)
- Physicians independently adjudicate 100 cases each (~10 h each)
- Return blinded response JSONL

**Week 3 (+2w → +3w)**: Analysis

Merge 3 response files:

```bash
cat physician_p1_response.jsonl \
    physician_p2_response.jsonl \
    physician_p3_response.jsonl \
    > data/raw/physician_audit.jsonl

.venv/bin/python experiments/round10_physician_audit.py \
    --physician-labels data/raw/physician_audit.jsonl \
    --outcome-labels data/raw/audit_r10_5/r10_5_audit_ground_truth.csv \
    --out results/round10/r10_5_physician_audit
```

Produces:
- `results/round10/r10_5_physician_audit.{json,md}`
- Pairwise Cohen's $\kappa$ (p1–p2, p1–p3, p2–p3)
- Mean $\kappa$
- Confusion matrix vs outcome-derived label
- Disagreement bucket (which cases physicians labeled differently
  from our outcome-derived stratum)

**Week 4 (+3w → camera-ready)**: Paper §4.3 update

Replace the deferred paragraph with actual $\kappa$ values.
Pre-committed reporting: whatever the number, it appears. If mean
$\kappa < 0.5$ against outcome-derived labels, that becomes a
substantive discussion of construct-validity threats.

### 1.4 Contingency: if any physician drops out

- 2-of-3 fallback: pairwise $\kappa$ reported, mean $\kappa$ over
  the surviving pair.
- Full re-run with substitute physician if timeline permits.

### 1.5 Reporting into UASEF_ML4H_2026.md

New paragraph in §4.3 (Limitations → move to §4.5 Camera-ready
addendum):

```markdown
### 4.5 Camera-ready physician audit (R10.5)

Three board-certified physicians (emergency, internal medicine,
family medicine) independently adjudicated 100 stratified cases (50
CRITICAL + 50 HIGH from MIMIC-IV, decision-time features only, no
future outcomes) using the shipped IRB package
(`paper/irb_audit_package/`). Mean pairwise Cohen's $\kappa$ = [X];
confusion matrix vs outcome-derived label shows [Y]% agreement on
CRITICAL and [Z]% on HIGH. [Discussion.]
```

---

## 2. eICU LLM — R11.3 LLM Pass A/B

### 2.1 Current status

- **Tabular**: complete. Reported in §3.2 (RF/XGB/LogReg/GBDT,
  5-seed pooled).
- **LLM (gpt-oss-120b)**: shipped code, deferred execution.

### 2.2 Wallclock

At LMStudio's measured 0.10 calls/s:
- Pass A: 5 seeds × (3000 cal + 3000 test) = 30,000 calls → ~83 h
- Pass B: 5 seeds × 6,000 calls = 30,000 calls → ~83 h
- **Total: ~166 h ≈ 7 days** wall-clock (matches R11.3 tabular
  protocol parity, §3.2)

### 2.3 Prerequisites (already satisfied)

- eICU-CRD v2.0 downloaded to `data/raw/eicu-crd/` (5 CSV files)
- `data/raw/eicu_cases_v11_full.jsonl` generated (200,859 stays)
- `EHRCaseLite.source = "eicu_struct"` fix applied
  (`experiments/round11_eicu_replication.py`)
- LMStudio serving `openai/gpt-oss-120b` on `localhost:1234`

### 2.4 Launch command

Background execution with `caffeinate` (Mac sleep prevention) and
`nohup` (terminal-independent):

```bash
cd /Users/idaun/PoC/UASEF
export UASEF_BACKEND_NEVER_SEND_PHI=1
find experiments/__pycache__ -name "round11*.pyc" -delete 2>/dev/null

caffeinate -dimsu nohup .venv/bin/python \
    experiments/round11_eicu_replication.py \
    --jsonl data/raw/eicu_cases_v11_full.jsonl \
    --include-llm \
    --classifiers gpt_oss_120b \
    --n-cal 3000 --n-test 3000 \
    --seeds 42 43 44 45 46 \
    > results/round11/r11_3_full_llm.log 2>&1 &

echo "Background PID: $!"
echo "Progress: tail -f results/round11/r11_3_full_llm.log"
```

**Only** `gpt_oss_120b` in `--classifiers`: tabular results are
already cached (do not re-run RF/GBDT/etc). The script will merge
LLM results into existing per-classifier tables.

### 2.5 Progress monitoring

Every 12h (once/day) check:

```bash
# Which seed is currently running?
grep "seed=" results/round11/r11_3_full_llm.log | tail -3

# How many 3000/3000 blocks complete? (each seed = 2 blocks: cal + test)
grep -c "3000/3000" results/round11/r11_3_full_llm.log
# 0-10: Pass A in progress
# 10-20: Pass B in progress
# 20:    complete
```

Expected progress:
- Day 1: seeds 42-43 of Pass A
- Day 2: seeds 44-45 of Pass A
- Day 3: seed 46 Pass A + seed 42 Pass B
- ...
- Day 7: complete

### 2.6 Contingency: LMStudio crash / Mac reboot

- Result cache: `results/round11/r11_1_cache/gpt_oss_120b.json`
  (per-seed cache, persistence across restarts)
- Resume: re-run same command; script skips cached seeds

### 2.7 Reporting into UASEF_ML4H_2026.md

Add to §3.2 table (or footnote at time of camera-ready):

```markdown
LLM (gpt-oss-120b, 5-seed pooled, camera-ready addition):
- Pass A: miss [X]/274, over_esc [Y]%
- Pass B: miss [Z]/274, over_esc [W]%
Pre-registered expectation: LLM has lower over_esc than tree
ensembles (limited sharpness caps escalate-all) but higher CRITICAL
miss, matching the R10.1 MIMIC-IV pattern. Result confirms/refutes
[X]. 
```

---

## 3. Combined Timeline (parallel execution)

```
Week 0 (notification)
├── Week 1: IRB submission + physician recruitment  ─────┐
│           eICU LLM background start (PID recorded)    │
├── Week 2: Physician adjudication (10h each × 3)       │
├── Week 3: Physician $\kappa$ analysis                  │  parallel
│           eICU LLM completes (~day 7-8)                │  tracks
├── Week 4: Paper §4.3 → §4.5 update                    │
│           Camera-ready submission                       ─┘
```

**Total human hours** (author-side): ~10 h (IRB paperwork + physician
coordination + $\kappa$ interpretation). Physicians: 10 h × 3.
Wallclock: 4 weeks parallel.

**Total cost**: $2,400 (physician honoraria). $0 external API. $0
compute (local Mac Studio).

---

## 4. Files updated at camera-ready

1. `paper/UASEF_ML4H_2026.md` §4.3 → §4.5 physician audit results
2. `paper/UASEF_ML4H_2026.md` §3.2 LLM row of eICU table
3. `results/round10/r10_5_physician_audit.{json,md}` (generated)
4. `results/round11/r11_3_eicu_pass_{a,b}.json` LLM per_seed
5. `data/raw/physician_audit.jsonl` (blinded merged responses;
   commit only anonymized aggregate CSV, NOT raw responses)
6. `experiments/round11_paper_audit.py` — extend to verify new
   numerics (30 s runtime)

---

_Camera-ready path drafted 2026-07-02. Both items independent;
either can be dropped without affecting the other._
