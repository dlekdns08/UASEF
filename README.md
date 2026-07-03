# UASEF

![UASEF](UASEF.png)

Uncertainty-Aware Safe Escalation Framework for Medical LLM Agents

LLM 기반 의료 에이전트가 **자신의 불확실성을 정량화**하고, **위험도를 판단**하여 **인간 전문가에게 자동 인계**하는 연구 프레임워크입니다.

---

> ## ⭐ 현재 주력 연구 (2026) — Diagnostic Framework for Conformal Clinical-Escalation Pipelines
>
> 이 저장소의 **현재 주력 연구**는 초기의 "LLM 에스컬레이션 게이트 구축"에서
> 한 걸음 나아가, **conformal clinical-escalation 파이프라인을 위한 *검증된 진단
> 프레임워크*** 로 발전했습니다. 핵심 통찰: 임상 CP 게이트는 커버리지가 완벽해
> 보여도 **여러 구조적 함정(leakage·orientation·informative missingness)** 때문에
> 배포 불가한데, 커버리지만 보고하면 이 함정이 드러나지 않습니다. 우리는 이 함정들을
> **측정 가능한 탐지기**로 형식화하고, MIMIC-IV·eICU-CRD 실데이터에서 검증합니다.
>
> - **대상 저널**: ***Patterns*** (Cell Press) — data-science methodology
> - **논문 초안**: [paper/UASEF_PATTERNS_2026.md](paper/UASEF_PATTERNS_2026.md)
> - **상세 문서**: 아래 [§0.8 진단 프레임워크](#08-현재-연구--diagnostic-framework-patterns-2026) 참조
> - **재현성**: 검증 코어 10/10 tests, 전량 로컬 연산, 외부 API $0, PHI egress 0 bytes
>
> 아래 **§0.5~§12** 는 원래의 LLM-escalation 프레임워크(v1/v2 — UQM/RTC/EDE,
> LangGraph ReAct)를 다루며, **여전히 유효한 참조 구현**으로 보존됩니다.

---

## 0.8. 현재 연구 — Diagnostic Framework (Patterns 2026)

> **논문**: *A Validated Diagnostic Framework for Conformal Clinical-Escalation
> Pipelines: Five Failure Modes, Five Detectors, and a Reporting Audit*
> ([paper/UASEF_PATTERNS_2026.md](paper/UASEF_PATTERNS_2026.md)).

### 0.8.1 무엇이 바뀌었나 (연구 재정의)

초기 UASEF는 "LLM logprob CP로 에스컬레이션 게이트를 만든다"였습니다. 실데이터
(MIMIC-IV structured EHR)로 검증하는 과정에서 **그럴듯해 보이는 "성공"이 실은
다섯 종류의 아티팩트**였음을 적대적 검증(adversarial workflow)으로 밝혀냈습니다:

1. **Score-sign 반전** — `score = -P(y=1)` 관례 버그가 CRC를 뒤집어 escalate-all 붕괴
2. **Escalate-all 붕괴** — 임계값이 모든 케이스를 에스컬레이션(vacuous coverage)
3. **Temporal leakage** — outcome/ICU-transfer 이후 charttime 피처 유입
4. **Informative missingness** — *어떤* lab을 주문했나 자체가 severity proxy
5. **Definitional leakage** — outcome에서 파생된 피처(예: APACHE)

이 다섯을 각각 **입력·통계량·판정 규칙이 정의된 형식적 탐지기**로 만들고, 아티팩트를
제거한 뒤 남은 **정직한 음성 결과(leakage-safe floor)** 를 정보이론 상한으로 확정했습니다.

### 0.8.2 다섯 실패 모드 · 다섯 탐지기 ([models/audit_detectors.py](models/audit_detectors.py))

| 탐지기 | 통계량 | 판정 규칙 | 측정 성능 (known-answer benchmark) |
| --- | --- | --- | --- |
| **Orientation** | AUROC(s, Y) | flag if < 0.5 | sensitivity 1.00 / specificity 1.00 |
| **Escalate-all** | λ̂에서 over-escalation | flag if ≥ 0.95 | vacuous(0.95) vs weak-but-real(0.93) 분리 |
| **Temporal leakage** | outcome 이후 양성 피처 비율 | flag if > 0.05 | 0% 주입 시 FP 0, 5%부터 발화 |
| **Informative missingness** | flag-only ÷ full above-chance (recover) | flag if ≥ 0.85 | null-calibrated (아래 R28: 임계 z=11.2) |
| **Definitional leakage** | 최대 단변량 AUROC | flag if ≥ 0.90 | 0 leak 시 FP 0, strength 0.8부터 발화 |

### 0.8.3 검증된 코어 ([models/conformal_escalation.py](models/conformal_escalation.py))

- **StandardCRC** (sup-λ 효율 임계) + **BoundedCRC** (양측 손실)
- **orientation guard** — 관례 버그를 fit 단계에서 차단, 실패 시 조용한 escalate-all 대신 **명시적 INFEASIBLE**
- `_evaluate` → `satisfies_crc / high_conf_coverage / vacuous / genuine_win` 판정
- **10/10 단위 테스트** ([tests/test_conformal_escalation.py](tests/test_conformal_escalation.py)): separable→non-vacuous, noise→high over_esc, inverted-sign→붕괴 재현

### 0.8.4 핵심 결과

| 결과 | 수치 | 위치 |
| --- | --- | --- |
| **Leakage-safe floor** (MIMIC-IV 14,000) | CRITICAL FULL AUROC **0.796**, value≈flag (0.795/0.791), over_esc 0.534 — 값이 주문 여부 위에 거의 추가 정보 없음 → **배포 가능한 게이트 없음** | §7, R18 |
| **정보이론 상한** | 합법적 decision-time 피처가 H(Y)의 **≈29%(I/H=0.29)** 만 해소; 6개 모델·KSG·label-noise 전반 강건 | §7, R22·R25 |
| **보고 감사** | 24편 dual-coded, Cohen's **κ=0.73**; real-EHR CP 부분집합 n=10 → **0/10** informative-missingness 완전처리(95% CP 상한 0.31) | §6, R26·R29 |
| **교차기관 전이** | MIMIC 고정 임계 → eICU regime-correct, 27%→74% 커버리지 이동에도 portable | §5.5, R27 |
| **임계 null-calibration** | 0.85 경보선 = 순열 귀무 대비 z=11.2(보수적); 관측 0.82 = z=10.8(substantive) | §3, R28 |
| **충실한 CPMORS 24h 재구성** | ordering-recovery 0.62(6h)→0.82(24h) — 신호 82%가 lab 값 아닌 주문 행위 | §5.4, R23 |

### 0.8.5 실험 인벤토리 (R18–R29)

> 실행 전: `export UASEF_BACKEND_NEVER_SEND_PHI=1` (PHI 전송 차단). 원시데이터는
> PhysioNet credentialed (MIMIC-IV v3.1·eICU-CRD v2.0), 저장소에 재배포하지 않음
> (`.gitignore` 처리). 모든 연산 로컬, 외부 API $0.

| 실험 | 목적 | 산출물 |
| --- | --- | --- |
| [round18_leakage_safe_floor.py](experiments/round18_leakage_safe_floor.py) | decision-time-guarded floor 산출 | `results/round18/` |
| [round19_eicu_audit.py](experiments/round19_eicu_audit.py) | eICU 독립 데이터셋 감사 검증 | `results/round19/` |
| [round20_detector_benchmark.py](experiments/round20_detector_benchmark.py) | 탐지기 sens/spec (주입 벤치마크) | `results/round20/` |
| [round21_audit_published_pipeline.py](experiments/round21_audit_published_pipeline.py) | 재구성 published 파이프라인 감사 (6h) | `results/round21/` |
| [round22_mi_boundary.py](experiments/round22_mi_boundary.py) | 정보이론 상한 (3-way contrast) | `results/round22/` |
| [round23_cpmors_24h.py](experiments/round23_cpmors_24h.py) | CPMORS 24h 충실 재구성 감사 | `results/round23/` |
| [round24_semireal_benchmark.py](experiments/round24_semireal_benchmark.py) | 실 MIMIC에 known leakage 주입 벤치마크 | `results/round24/` |
| [round25_mi_robustness.py](experiments/round25_mi_robustness.py) | MI 상한 강건성 (6모델+KSG+noise) | `results/round25/` |
| [round26_interrater_kappa.py](experiments/round26_interrater_kappa.py) | 감사 inter-rater Cohen's κ | `results/lit_audit/kappa.*` |
| [round27_threshold_transfer.py](experiments/round27_threshold_transfer.py) | MIMIC→eICU 임계 전이 | `results/round27/` |
| [round28_missingness_null_calibration.py](experiments/round28_missingness_null_calibration.py) | 결측 임계 순열 null-calibration | `results/round28/` |
| [round29_audit_expand_merge.py](experiments/round29_audit_expand_merge.py) | real-EHR 감사 확장 병합 + κ 재계산 | `results/lit_audit/expanded_audit_stats.*` |

```bash
# 검증 코어 단위 테스트 (데이터 불필요, ~1초)
.venv/bin/python -m pytest tests/test_conformal_escalation.py -q      # 10/10

# 탐지기 known-answer 벤치마크 (합성, 데이터 불필요)
.venv/bin/python experiments/round20_detector_benchmark.py

# 실데이터 실험 (PhysioNet 자격 + 로컬 캐시 필요)
export UASEF_BACKEND_NEVER_SEND_PHI=1
.venv/bin/python experiments/round18_leakage_safe_floor.py     # leakage-safe floor
.venv/bin/python experiments/round22_mi_boundary.py            # 정보이론 상한
.venv/bin/python experiments/round27_threshold_transfer.py     # 교차기관 전이
```

### 0.8.6 재현성 · 데이터 컴플라이언스

- **검증 코어 + 탐지기**: 순수 함수 + 단위 테스트. 데이터 없이 실행/검증 가능.
- **문헌 감사**: `results/lit_audit/` — 24편 dual-coded codings(coder A/B), κ, 확장 통계. 논문 초록·출처만 사용, 원문 재배포 없음.
- **PHI 보호**: `UASEF_BACKEND_NEVER_SEND_PHI=1` 상시, 원시 EHR·lab 캐시 `.gitignore` 제외, 전량 로컬 처리 (PHI egress 0 bytes, 외부 API $0).
- **관련 문서**: [paper/ENDPOINT_COLLAPSE_THEOREM.md](paper/ENDPOINT_COLLAPSE_THEOREM.md) (endpoint-collapse 정리), [paper/LITERATURE_AUDIT_PROTOCOL.md](paper/LITERATURE_AUDIT_PROTOCOL.md) (감사 프로토콜), [improvements/round23plus_PLAN.md](improvements/round23plus_PLAN.md) (R23–R29 강화 계획).

---

## 목차

0. [Quick Start](#0-quick-start-5줄)
0.5. [v2 Framework — Round 7](#05-v2-framework--round-7-이론적-기여-강화) (Stratified CRC + Multi-Trigger Combination + Cost-Aware)
0.8. [⭐ 현재 연구 — Diagnostic Framework (Patterns 2026)](#08-현재-연구--diagnostic-framework-patterns-2026) (Five Failure Modes · Five Detectors · Reporting Audit)
1. [연구 배경 및 동기](#1-연구-배경-및-동기)
2. [핵심 설계 철학](#2-핵심-설계-철학)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [아키텍처 상세](#4-아키텍처-상세)
   - 4.1 [UQM — Uncertainty Quantification Module](#41-uqm--uncertainty-quantification-module)
   - 4.2 [RTC — Risk-Threshold Calibrator](#42-rtc--risk-threshold-calibrator)
   - 4.3 [EDE — Escalation Decision Engine](#43-ede--escalation-decision-engine)
   - 4.4 [LangGraph 에이전트](#44-langgraph-에이전트)
   - 4.5 [Round 7 — Stratified CRC + MTC + Cost-Aware (계획)](#45-round-7--stratified-crc--multi-trigger-combination--cost-aware-계획)
5. [데이터셋](#5-데이터셋)
6. [실험 설계](#6-실험-설계)
   - 6.0 [캘리브레이션 파이프라인](#60-캘리브레이션-파이프라인-run_calibration_pipelinepy)
   - 6.1 [순차 파이프라인 실험](#61-순차-파이프라인-실험-run_experimentpy)
   - 6.2 [LangGraph 에이전트 실험](#62-langgraph-에이전트-실험-run_agent_experimentpy)
   - 6.3 [MedAbstain 분류 정확도 평가](#63-medabstain-분류-정확도-평가-eval_medabstainpy)
   - 6.4 [Pareto Frontier Alpha Sweep](#64-pareto-frontier-alpha-sweep-pareto_sweeppy)
   - 6.5 [베이스라인 비교 실험](#65-베이스라인-비교-실험-run_baseline_comparisonpy)
   - 6.6 [전체 실험 통합 실행기](#66-전체-실험-통합-실행기-run_all_experimentspy)
7. [평가 지표](#7-평가-지표)
8. [설치 및 환경 구성](#8-설치-및-환경-구성)
9. [실험 실행](#9-실험-실행)
10. [출력 파일](#10-출력-파일)
11. [논문 권장 설정](#11-논문-권장-설정)
12. [참고문헌](#12-참고문헌)

> 코드 변경 / 개선 이력 전체는 [improvements/README.md](improvements/README.md)에서 라운드별로 관리합니다.
>
> **📄 논문 draft (Main Paper)**: [paper/UASEF_Round7.md](paper/UASEF_Round7.md) (English) · [paper/UASEF_Round7_KO.md](paper/UASEF_Round7_KO.md) (한국어). Round 7 framework의 학술 논문 형식 정리 (Abstract / Related Work / Method / Experiments / Discussion / Limitations / References / Appendix). ML4H 2026 / AISTATS 2026 / NeurIPS 2026 target.
>
> **📚 Supplementary Materials (Appendix B)**: [paper/UASEF_Round7_Supplementary.md](paper/UASEF_Round7_Supplementary.md) (English) · [paper/UASEF_Round7_Supplementary_KO.md](paper/UASEF_Round7_Supplementary_KO.md) (한국어). v1 4개 sub-experiment (agent / baseline / medabstain / pareto) 결과를 학술 형식으로 정리한 template. 실측치는 `bash run_full_evaluation.sh` 실행 시 `results/run_<ts>/result_supplementary.md`에 자동 채워진다.

---

## 0. Quick Start (5줄)

> **30초 안에 결과 1개 뽑기.** 자세한 설명은 §9.

```bash
# 1) 의존성 설치 (uv 권장; pip도 가능)
uv pip install -e .

# 2) OpenAI 키 설정
echo "OPENAI_API_KEY=sk-..." > .env

# 3) 캘리브레이션 (1회, ~3분)
python experiments/run_calibration_pipeline.py --backend openai --n-cal 100 --n-labeled 20

# 4) ⭐ 표준 진입점 — 4개 실험 일괄 실행 (~10분)
python experiments/run_all_experiments.py --backend openai \
    --n-cal 100 --n-test 30 --n-medabstain 30 --skip pareto

# 5) 결과 확인
open results/all_experiments_report.md
```

**다른 모델/백엔드?** §9.1 환경변수와 logprob 호환성 매트릭스를 참고하세요. Claude/Gemini/o3/gpt-5 등 logprob-free 모델은 `--backend anthropic` 또는 `--scoring-method hybrid`로 자동 처리됩니다 (audit 6.9·6.10).

**여러 ablation 비교?** `--run-tag` 옵션으로 분리 후 `python experiments/compare_runs.py base instructed confidence`로 통합 표 생성 (audit 6.10).

**v1(`run_all_experiments`) + v2(Round 7) + 모든 backend 한 번에 + 통합 보고서?**

```bash
# 합성 검증 + pytest만 (LLM 키 불필요, ~30초)
SKIP_LLM=1 bash run_full_evaluation.sh

# OpenAI만 전체 실행 (LLM 호출 포함)
BACKENDS="openai" N_CAL=500 N_TEST=200 bash run_full_evaluation.sh

# OpenAI + LMStudio 모두 (논문 quality)
BACKENDS="openai lmstudio" N_CAL=500 N_TEST=200 bash run_full_evaluation.sh
```

**한 번 실행으로 4단계 자동:**

1. **pytest 154 tests** (회귀 안전망 + cross-backend sanity check + 새 baseline / dataset / 4-D sweep / α=0.001)
2. **v1**: `run_all_experiments.py × {각 backend}` — 각 호출이 agent + baseline + medabstain + pareto 4개 sub 자동 포함
3. **v2 Round 7 합성** (backend 무관): Table 2 FWER + Table 3 Cost
4. **v2 Round 7 LLM** (backend별): Table 1 per-stratum coverage + Table 4 head-to-head (TECP / Quach / Semantic Entropy / UASEF Round 6 / UASEF Round 7 / **TECP-stratified** / **Cost-Sensitive single-α** ablation baselines)

**Multi-seed bootstrap (camera-ready 인프라):**

```bash
# 5-seed 자동 aggregation (~$125 OpenAI + ~50분 LMStudio)
SEEDS="42 43 44 45 46" BACKENDS="openai lmstudio" \
    bash run_multiseed_evaluation.sh
# → results/run_<ts>_aggregate/aggregate_seeds.{json,md}
#   = 평균 ± 표준편차 + 95% bootstrap CI (per backend × method × stratum)
```

**산출물 구조** (`results/run_<timestamp>/`):

```text
result.md                         ← 통합 보고서 (사람용, paper main 표)
result_supplementary.md           ← Supplementary (paper Appendix B, v1 sub-experiment 정리) ★
result.json                       ← 구조화 결과
pytest_summary.txt
synthetic/
├── table2_fwer.{json,md}         ← v2 합성 (backend 무관)
└── table3_cost.{json,md}
openai/
├── all_experiments_report.md     ← v1 통합
├── all_experiments_summary.json
├── agent_results.json + comparison_table.csv
├── baseline_comparison.{json,csv}
├── medabstain_eval.{json,csv}
├── pareto_sweep_results.json + alpha_recommendations.json + pareto_frontier.png
├── table1_coverage.{json,md}     ← v2 Round 7 Pivot A
└── table4_baseline.{json,md}     ← v2 Round 7 head-to-head
lmstudio/
└── (동일 구조)
```

> **★ Supplementary 자동 생성 조건**: `SKIP_V1=0`(default)으로 실행하면 v1
> 결과를 paper Appendix B 형식 (B.1 Agent ReAct / B.2 Trigger Ablation /
> B.3 MedAbstain Variant-Level / B.4 Pareto α / B.5 Cross-Backend) 5개 표로
> 자동 정리하여 `result_supplementary.md`에 저장한다. v1 데이터가 없으면
> SKIP되며 보고서에 명시된다.

**제어 환경변수**: `BACKENDS` / `N_CAL` / `N_TEST` / `N_MEDABSTAIN` / `N_PARETO` / `N_TRIALS` / `N_PER_STRATUM` / `ALPHA` / `SEED` / `SKIP_LLM` / `SKIP_TESTS` / `SKIP_V1` / `SKIP_V2_SYN` / `SKIP_V2_LLM`. 자세한 사용은 `bash run_full_evaluation.sh` 헤더 주석.

---

## 0.5. v2 Framework — Round 7 (이론적 기여 강화)

> **2026-05 audit 6.10에서 코드 품질이 회복된 이후, 학술적 contribution을 명확히 하기 위해 도입된 v2 (Round 7) 프레임워크.** 자세한 구현 계획은 [improvements/round7_PLAN.md](improvements/round7_PLAN.md) 참고.

기존 v1 (Round 1~6.10)은 logprob CP + heuristic risk multiplier + ad-hoc trigger OR로 구성되어, 다음 선행 연구와 충분히 차별화되지 않았다:

- **TECP** (Xu & Lu, 2025) — token-entropy nonconformity CP
- **Conformal Language Modeling** (Quach et al., ICLR 2024) — NLL nonconformity + dynamic stopping
- **MedAbstain** (EACL 2026) — CP를 이미 통합한 평가 데이터셋
- **Conformal Risk Control** (Angelopoulos & Bates, ICLR 2024) — 더 일반적인 risk control

v2 (Round 7)는 **3가지 이론적 contribution을 결합**해 명확한 차별화를 만든다:

| Contribution | 핵심 | 코드 위치 (계획) |
| --- | --- | --- |
| **A. Stratified Conformal Risk Control** | 위험도 stratum별 (CRITICAL/HIGH/MODERATE/LOW) 별도 calibration + Conformal Risk Control 알고리즘 적용. 각 stratum에 대해 `E[ℓ_stratum] ≤ α_stratum` 보장. | `models/stratified_crc.py` (신규) |
| **B. Multi-Trigger Conformal Combination** | T1/T2/T3 trigger를 별도 nonconformity score로 calibrate한 뒤 harmonic / e-value combination으로 결합. **FWER ≤ α 보장 (under arbitrary dependence)** — 기존 `len(triggers) > 0`은 보장 위반. | `models/conformal_combination.py` (신규) |
| **C. Cost-Aware Threshold Optimization** | 임상 현실의 비대칭 cost (CRITICAL miss = 1000× over-esc) 명시화. F1-symmetric grid search를 cost-weighted constrained optimization으로 교체. | `models/cost_aware_calibration.py` (신규) |

세 contribution은 서로 결합 — **C의 cost matrix가 A의 α_stratum 결정**, **A의 stratified score가 B의 입력**, **B의 결합 p-value가 C의 최적화 대상**.

### v1 vs v2 한 눈에 비교

| | v1 (Round 1~6.10) | v2 (Round 7) |
| --- | --- | --- |
| Threshold | `q̂_global × heuristic_multiplier` | `λ_stratum (Conformal Risk Control)` |
| Trigger 결합 | `len(triggers) > 0` (ad-hoc OR) | conformal p-value combination (harmonic/e-value) |
| Coverage 보장 | 단일 global α | per-stratum α with risk control |
| Cost 모델 | F1-symmetric (FN==FP) | 비대칭 cost matrix per stratum |
| 비교 baseline | (없음) | TECP, Quach 2024 CLM, Semantic Entropy |
| Target venue | (코드 reference) | ML4H Spotlight / AISTATS / NeurIPS |

### 작업 상태

- v1 (Round 1~6.10): **완료** (코드 + 82 tests passing)
- v2 (Round 7): **계획 단계** — [improvements/round7_PLAN.md](improvements/round7_PLAN.md) 참고. 4~6주 작업 예상.
- v2 미구현 시에도 v1 코드는 reference implementation으로 동작.

---

## 1. 연구 배경 및 동기

### 문제 정의

LLM을 의료 현장에 배포할 때 가장 큰 장벽은 **"모델이 언제 틀리는지 모른다"** 는 점입니다. 의료 도메인에서 잘못된 자신감(overconfidence)은 치명적 결과로 이어질 수 있습니다. 기존 접근법들은 세 가지 한계를 가집니다.

| 기존 접근법 | 한계 |
|------------|------|
| 단순 threshold 기반 | 임계값이 임의적이며 통계적 보장 없음 |
| Human-in-the-loop (항상) | 자율 처리 불가 → 운영 비용, 지연 |
| 확률 보정(calibration) | 도메인 이동(distribution shift) 시 보장 붕괴 |

### UASEF의 제안

UASEF는 **Conformal Prediction(CP)** 이론을 의료 LLM에 적용하여 세 가지를 동시에 달성합니다.

1. **통계적 보장**: `P(s_test ≤ q̂) ≥ 1 - α` — 이론적으로 증명된 커버리지
2. **동적 위험도 반영**: 전문과목·시나리오에 따라 임계값 자동 조정
3. **선택적 에스컬레이션**: 자율 처리 가능한 케이스는 AI가, 불확실하거나 고위험인 케이스만 전문의에게 인계

---

## 2. 핵심 설계 철학

### 왜 Conformal Prediction인가?

#### 기존 확률 보정의 문제점 (temperature scaling, Platt scaling)

- 모델이 "70% 확신" 이라고 말해도, 실제로 70% 맞을 것이라는 보장이 없음
- 학습 분포와 다른 도메인에서 보정 성질이 유지되지 않음

#### Conformal Prediction의 강점

- 분포 가정 불필요. 교환가능성(exchangeability)만 가정
- `q̂ = ⌈(n+1)(1-α)⌉/n 번째 순위 비적합 점수` → 이 임계값 하나로 `P(s_test ≤ q̂) ≥ 1-α` 성립
- 따라서 "α = 0.05로 설정 → 실제 에스컬레이션 누락률 ≤ 5%"가 수학적으로 보장됨

### 왜 Nonconformity Score로 token logprob을 쓰는가?

비적합 점수(nonconformity score)는 "이 테스트 포인트가 캘리브레이션 셋과 얼마나 다른가"를 수치화한 것입니다. UASEF는 **평균 negative log-likelihood**를 사용합니다.

```text
s(x) = -mean(log P(t_i | context, t_1, ..., t_{i-1}))

```

이 선택의 이유:

- 모델이 생성한 각 토큰의 확률을 그대로 반영 → 답변 생성 과정 자체의 불확실성
- Temperature = 0일 때도 의미 있음 (greedy decoding이지만 logprob은 여전히 분포를 반영)
- API 추가 호출 불필요 (generate 한 번으로 score와 답변을 동시에 얻음)

### 왜 Round 7에서 Stratified CRC + Multi-Trigger Combination + Cost-Aware Calibration인가?

v1 (Round 1~6.10)의 세 가지 약점:

1. **단일 global α**가 모든 specialty에 동일 적용 → 응급의학과 일반외래에 같은 임계값을 쓰는 셈. 임상 현실과 불일치.
2. **`len(triggers) > 0` 결합**은 통계적 보장 없음 — 3개 trigger 모두 α=0.10이면 결합 false-positive ~0.27 가능 (FWER 위반).
3. **F1-symmetric optimization**은 FN과 FP를 동등하게 다룸. 응급에서 놓침은 사망, 일반외래에서 과에스컬레이션은 시간 낭비 — 비대칭이 정상.

v2 (Round 7)의 세 답변:

1. **Stratified Conformal Risk Control** (Angelopoulos & Bates ICLR 2024 + Romano et al. 2020 class-conditional CP의 결합) → 각 stratum에 대해 `E[loss_stratum] ≤ α_stratum` 형식 보장. CRITICAL은 α=0.001 (0.1% 미스 한계), LOW는 α=0.10 (10% 허용).
2. **Multi-Trigger Conformal Combination** (Vovk & Wang 2019 harmonic mean / Wang & Ramdas 2022 e-value) → 임의 의존 구조에서도 FWER 보장. T2(키워드)와 T3(근거 부재)도 nonconformity score로 frame.
3. **Cost-Aware Optimization** → `c_FN(stratum) × FN + c_FP × FP` 최소화, CRC 제약 보존. cost matrix는 `base_config.yaml`에 명시 (sensitivity analysis 가능).

### 왜 세 모듈로 분리했는가?

```text
UQM  →  "이 질문이 얼마나 어려운가?" (CP 기반 통계 측정)
RTC  →  "얼마나 어려워야 에스컬레이션할 것인가?" (위험도 기반 임계값)
EDE  →  "최종적으로 에스컬레이션할 것인가?" (다중 신호 통합 결정)

```

세 관심사를 분리함으로써:

- UQM은 CP 이론 컴포넌트로만 교체 가능 (weighted CP, conformal risk control 등)
- RTC의 전문과목 위험도 온톨로지는 임상 전문가 피드백으로 독립 업데이트 가능
- EDE의 트리거 정책은 기관별 프로토콜에 맞게 조정 가능

### 왜 LangGraph ReAct 구조인가?

단순 쿼리-응답이 아닌 **추론-행동 루프(Reasoning + Acting)**를 택한 이유:

- 의료 질문은 단일 답변보다 도구 활용(약물 상호작용 DB, 가이드라인 검색)이 필요
- UASEF는 에이전트 내부가 아닌 **외부에서 독립적으로 판정** → 에이전트 출력을 감사(audit)하는 구조
- LangGraph의 StateGraph가 ReAct 루프와 UASEF 체크 노드를 명확히 분리

---

## 3. 프로젝트 구조

```text
UASEF/
├── models/                         # 핵심 모듈
│   ├── model_interface.py          # LMStudio / OpenAI 통합 추상화 레이어 (BOM·제어문자 sanitize 포함)
│   ├── uqm.py                      # Uncertainty Quantification Module (CP 기반)
│   ├── rtc_ede.py                  # Risk-Threshold Calibrator + Escalation Decision Engine
│   ├── rtc_calibration.py          # ★ RTC 배율 Pareto sweep (데이터 기반 역산)
│   ├── entropy_calibration.py      # ★ 엔트로피 임계값 Youden's J 자동 결정
│   └── ede_coefficient_search.py   # ★ EDE confidence 계수 grid search
│
├── agent/                          # LangGraph ReAct 에이전트
│   ├── graph.py                    # StateGraph 조립
│   ├── nodes.py                    # 노드 함수 + AgentComponents
│   ├── state.py                    # MedicalAgentState TypedDict
│   └── tools.py                    # 의료 도구 4종 (drug, guideline, lab, DDx)
│
├── data/
│   ├── loader.py                   # MedQA / MedAbstain / PubMedQA / MIMIC-III 로더
│   ├── raw/                        # 로컬 JSONL 파일 위치 (.gitignore)
│   └── README.md                   # 데이터 소스 및 다운로드 가이드
│
├── experiments/
│   ├── configs/                    # 시나리오별 YAML 설정
│   │   ├── base_config.yaml        # 공통 기본값 (캘리브레이션 결과 포함)
│   │   ├── scenario_emergency.yaml
│   │   ├── scenario_rare_disease.yaml
│   │   └── scenario_multimorbidity.yaml
│   ├── config_utils.py             # ★ 공통 캘리브레이션 config 로더
│   ├── run_calibration_pipeline.py # ★ 캘리브레이션 파이프라인 (Step 1→5)
│   ├── run_experiment.py           # 순차 파이프라인 실험 (LMStudio vs OpenAI)
│   ├── run_agent_experiment.py     # LangGraph 에이전트 실험
│   ├── run_baseline_comparison.py  # 베이스라인 비교 (no_esc / threshold_only / full_uasef)
│   ├── eval_medabstain.py          # MedAbstain AP/NAP 분류 정확도 평가
│   ├── pareto_sweep.py             # α sweep → Pareto frontier + α 권고
│   ├── run_all_experiments.py      # ★ 전체 실험 통합 실행 + 요약 보고서 생성
│   └── visualize_results.py        # 결과 시각화
│
├── results/                        # 실험 결과 (자동 생성, .gitignore)
├── pyproject.toml
└── .env.example

```

---

## 4. 아키텍처 상세

### 4.1 UQM — Uncertainty Quantification Module

**파일**: `models/uqm.py`

UQM은 단일 질문의 불확실성을 **통계적으로 보장된 수치**로 변환합니다.

#### 내부 흐름

```text
질문 입력
   ↓
_get_score(): LLM 호출 → token logprobs 수집
   ↓
compute_nonconformity_score(): s = -mean(logprobs)
   ↓
calibrator.threshold와 비교 → should_escalate
   ↓
compute_entropy(): top_logprobs로 위치별 조건부 엔트로피 계산
   ↓
UncertaintyResult 반환

```

#### Conformal Calibration 수식

보정 집합 `{s_1, ..., s_n}` (비적합 점수들)에서 임계값 계산:

```text
q̂ = s_{(⌈(n+1)(1-α)⌉)}  ← n번째 순위 점수

보장: P(s_test ≤ q̂) ≥ 1 - α

```

실제 구현에서는 `numpy.quantile`을 사용하며, level을 `min(1.0, ⌈(n+1)(1-α)⌉/n)` 으로 보정하여 유한 표본에서의 보수성을 확보합니다.

#### Scoring Method 비교

| 방식 | 수식 | 특징 | 논문 위치 |
| ---- | ---- | ---- | --------- |
| **logprob** (Primary + Ablation) | `s = -mean(token logprobs)` | CP 보장 ✓, 단일 쿼리 | 주요 기여 + Ablation |
| **self_consistency** (대안) | `s = Jaccard_diversity × 5` | CP 보장 ✓, N회 쿼리, logprobs 불필요 | 블랙박스 LLM 호환용 |
| **hybrid** (audit 6.9 신규) | `s = (0.5·diversity + 0.5·H_mode) × 5` | CP 보장 ✓, N회 쿼리, SC + answer-mode entropy | logprob-free 환경 권장 |
| **auto** (deprecated) | 런타임 감지 | 재현성 저하 위험 | 비권장 |

> **왜 logprob이 Primary이고 Ablation 모두인가?**
> LM Studio의 OpenAI-compatible API는 token-level logprobs를 지원하므로, OpenAI와 로컬 GGUF 모델 모두 동일한 `logprob` 비적합 함수를 사용합니다. Ablation의 목적은 scoring method 차이가 아니라, 로컬 환경에서도 CP coverage 보장이 성립함을 검증하는 것입니다.
>
> **`hybrid` (audit 6.9)** 는 logprob-free 환경(Claude/Gemini/OpenAI reasoning) 전용입니다. 같은 N=5 쿼리로 self_consistency보다 풍부한 신호 (token-level diversity + answer-mode clustering entropy)를 사용합니다. UQM이 backend/모델 사전 점검 후 자동으로 전환합니다.

#### 엔트로피 계산

`compute_entropy(response: ModelResponse)` 는 `top_logprobs`가 있을 때만 유효한 엔트로피를 반환합니다.

```python

# 각 토큰 위치에서 상위 k개 logprob으로 조건부 분포 근사

probs = softmax(top_k_logprobs)   # 정규화
H_pos = -sum(p * log(p))          # 위치별 엔트로피
H_avg = mean(H_pos)               # 전체 평균 (nats/token)

```

`top_logprobs`가 없으면 `float("nan")` 반환 — 개별 토큰 logprob으로는 Shannon 엔트로피를 계산할 수 없기 때문입니다 (각 값이 완전한 어휘 분포를 구성하지 않음).

#### Distribution Shift 처리

```python

# 보정: MedQA 분포

uqm.calibrate(cal_questions, distribution_source="medqa")

# 평가: MIMIC-III 분포 (다른 분포!) → 자동 경고 + Weighted CP 전환

uqm.evaluate(question, distribution_source="mimic3")

```

Weighted CP (Tibshirani et al., 2019)는 교환가능성 위반 시 커버리지 보장을 복원합니다.

```text
w_i = 1 + k × Jaccard(cal_i, test)   # 밀도비 근사

q̂_w = inf{q : Σ_{s_i ≤ q} w_i / (Σ w_i + w_{n+1}) ≥ 1-α}

```

`w_{n+1}` (테스트 포인트 자신의 weight)를 분모에 포함해야 CP 하한 보장이 성립합니다. `w_{n+1} = 1 + k` (Jaccard(test, test) = 1.0 이므로 최대 유사도).

#### UncertaintyResult 주요 필드

| 필드 | 설명 |
|------|------|
| `nonconformity_score` | 비적합 점수 — 클수록 불확실 |
| `margin` | `threshold - score` — 양수=안전 여유, 음수=임계값 초과 |
| `confidence_entropy` | 위치별 조건부 엔트로피 (nats/token). `top_logprobs` 없으면 `nan` |
| `should_escalate` | `score > threshold` 여부 |
| `weighted_cp_used` | Weighted CP 적용 여부 |
| `prediction_set_size` | 항상 1. 하위 호환성 유지용 필드 (binary outcome에서 prediction set은 단일 원소) |

#### LLM 지원 요건

| scoring_method | logprobs 필요 | 적용 가능 LLM | 논문 위치 |
| --- | --- | --- | --- |
| `logprob` (Primary + Ablation) | 필수 | GPT-4o, GPT-4.1, LMStudio (llama.cpp), MLX | 주요 기여 + Ablation |
| `self_consistency` (대안) | 불필요 | **모든** LLM (Claude, Gemini, OpenAI o-시리즈 포함) | 블랙박스 LLM 호환용 |
| `hybrid` (audit 6.9) | 불필요 | 위 self_consistency와 동일. 신호량이 더 많음. | logprob-free 환경 권장 |

> **audit 6.9 자동 감지**: `UQM(backend='anthropic'/'gemini')` 또는 `OPENAI_MODEL`이 reasoning 패턴(`o1*/o3*/o4*/gpt-5*`)일 때 `scoring_method='logprob'`을 요청하면, [models/uqm.py:512-562](models/uqm.py#L512)의 사전 점검이 발동합니다.
>
> - `strict=False` (default): UserWarning 후 자동으로 `self_consistency`(anthropic/gemini) 또는 `hybrid`(openai reasoning)로 전환
> - `strict=True`: `RuntimeError`로 즉시 중단 (논문 자동화에 권장)
>
> `backend_supports_logprobs(backend, model_name)` 함수로 정적 판정 가능 ([models/model_interface.py:51-78](models/model_interface.py#L51)).

#### Calibration 견고성

`UQM.calibrate()`는 샘플별 최대 3회 재시도 후 실패하면 해당 샘플만 건너뜁니다. LMStudio가 간헐적으로 logprobs를 반환하지 않는 경우나 OpenAI API 오류 발생 시 캘리브레이션 전체가 실패하지 않도록 설계되었습니다.

```text
[RETRY 1/3] 샘플 42: Backend이 logprobs를 반환하지 않습니다
...
[SKIP 42/500] 3회 실패, 샘플 건너뜀

```

또한 `ConformalCalibrator.fit()`은 CP 보장을 위한 최소 n을 검증합니다: `n_min = ⌈(1-α)/α⌉` (α=0.05 → n_min=19). n이 이 값보다 작으면 `UserWarning`이 발생합니다.

---

### 4.2 RTC — Risk-Threshold Calibrator

**파일**: `models/rtc_ede.py`, `models/rtc_calibration.py`

UQM이 반환한 기본 임계값 `q̂`를 전문과목과 시나리오의 위험도에 따라 **동적으로 조정**합니다.

#### 조정 수식

```text
adjusted_threshold = q̂ × risk_multiplier × scenario_multiplier

```

| 위험 등급 | 기본 배율 | 해당 전문과목 |
| -------- | ------- | ------------ |
| CRITICAL | ×0.60 | 응급의학, 중환자의학, 외상외과 |
| HIGH     | ×0.75 | 심장내과, 신경과, 종양학, 심흉외과 |
| MODERATE | ×1.00 | 내과, 외과, 소아과, 산부인과 |
| LOW      | ×1.30 | 일반 외래, 예방의학, 피부과, 정신건강의학과 |

`emergency` / `rare_disease` 시나리오에는 추가 ×0.90 적용됩니다.

> **설계 이유**: 응급의학에서 에스컬레이션 누락(False Negative)의 비용은 일반 외래에 비해 훨씬 큽니다. 임계값을 낮추면 더 많은 케이스가 에스컬레이션되지만, 위험한 케이스를 놓칠 확률이 줄어듭니다. 이 트레이드오프를 전문과목 온톨로지로 인코딩했습니다.

#### 데이터 기반 배율 역산 (`rtc_calibration.py`)

위 표의 배율은 **기본값**입니다. `run_calibration_pipeline.py`를 실행하면 레이블 데이터에서 Pareto sweep으로 배율을 자동 역산합니다.

```text
각 위험도 수준별로 후보 배율 (예: CRITICAL ∈ {0.55, 0.60, 0.65, 0.70, 0.75}) sweep
→ Safety Recall ≥ 0.95 AND Over-Escalation ≤ 0.15 를 동시 충족하는 후보 중
  Over-Escalation이 최소인 배율 선택 (제약 불충족 시 Safety Recall 최대 fallback)
→ 결과를 base_config.yaml의 rtc 섹션에 저장

```

결과는 `RTC(base_threshold, multipliers=cfg["rtc"])` 형태로 모든 실험 파일에 자동 주입됩니다.

#### Pareto Frontier 분석

```python
rtc.pareto_frontier(sweep_results)

```

`pareto_sweep.py`의 실측 데이터를 받아 각 `(α, specialty)` 조합에서 `(coverage, escalation_rate)` 쌍을 반환합니다. 이를 통해 실제로 측정된 trade-off를 시각화하고 최적 α를 권고합니다.

---

### 4.3 EDE — Escalation Decision Engine

**파일**: `models/rtc_ede.py`, `models/entropy_calibration.py`, `models/ede_coefficient_search.py`

세 가지 트리거를 통합하여 최종 에스컬레이션 여부를 결정합니다.

#### 트리거 구조

```text
Trigger 1 — UNCERTAINTY_EXCEEDED:
    nonconformity_score > adjusted_threshold
    → CP 이론의 직접 신호 (주 트리거)

Trigger 2 — HIGH_RISK_ACTION:
    CRITICAL_KEYWORDS 감지   (EOL 결정, Code Blue)  → 항상 트리거
    PROCEDURAL_KEYWORDS 감지 (intubation, 승압제)   → UNCERTAINTY_MODIFIERS 동반 시만 트리거

Trigger 3 — NO_EVIDENCE:
    근거 부재 표현 감지 (아래 참조)

하나라도 활성화 → should_escalate = True

```

> **Trigger 2 설계 이유**: "에피네프린을 아나필락시스에 투여하세요" 같은 정상적인 처치 권고가 키워드만으로 에스컬레이션되는 False Positive를 방지합니다. 따라서 시술 키워드는 불확실 표현(`consider`, `may need`, `if deteriorates` 등)과 함께 나타날 때만 활성화합니다. 반면 DNR, withdraw care 같은 EOL 결정은 AI가 단독으로 판단해서는 안 되므로 항상 에스컬레이션합니다.

#### Trigger 3 NO_EVIDENCE 키워드 목록

근거 부재 표현은 출처별로 관리됩니다. 논문 재현 시 `source` 필드를 인용 근거로 사용하세요.

| 출처 | 예시 표현 |
| ---- | -------- |
| `medabstain` | "i am not certain", "insufficient evidence", "limited data" |
| `savage2025` | "this is unclear", "evidence is mixed", "conflicting data" |
| `manual` (GPT-4o 500건) | "clinical judgment needed", "differential is broad" |
| `extended` | "cannot be determined", "requires further evaluation", "beyond my knowledge" |

전체 37개 문구가 `NO_EVIDENCE_STRINGS` (models/rtc_ede.py)에서 관리됩니다. 모든 실험 파일은 `from models.rtc_ede import NO_EVIDENCE_STRINGS`를 통해 단일 출처를 참조합니다.

탐지 함수 `detect_no_evidence(text)` 는 `(triggered: bool, matched_phrases: list[str])` 를 반환하여 논문 재현에 필요한 매칭 증거를 함께 제공합니다.

#### Confidence 계산

```text
confidence = min(1.0,
    len(triggers) / 3
    + t1_weight    if UNCERTAINTY_EXCEEDED in triggers   # 기본 0.4
    + entropy_boost if entropy > entropy_threshold       # 기본 0.15, 기본 임계값 2.0
)

```

엔트로피는 별도 트리거가 아닌 **신뢰도 가중치**로만 사용됩니다. 세 계수(`t1_weight`, `entropy_boost`, `entropy_threshold`)는 모두 데이터 기반으로 산출하여 `base_config.yaml`에 저장됩니다.

#### 엔트로피 임계값 자동 결정 (`entropy_calibration.py`)

`ENTROPY_HIGH_THRESHOLD = 2.0` 하드코딩 대신 calibration 데이터에서 Youden's J 통계량으로 자동 결정합니다.

```text
Youden's J = Sensitivity + Specificity - 1  (최대화 지점 선택)
→ 결과를 base_config.yaml의 entropy_threshold에 저장

```

#### EDE 계수 grid search (`ede_coefficient_search.py`)

```text
t1_weight    ∈ {0.2, 0.3, 0.4, 0.5}
entropy_boost ∈ {0.05, 0.10, 0.15, 0.20}

최적화 목표: F1-safety = harmonic_mean(Safety Recall, 1 − Over-Escalation Rate)
→ 결과를 base_config.yaml의 ede 섹션에 저장

```

---

### 4.4 LangGraph 에이전트

**파일**: `agent/graph.py`, `agent/nodes.py`, `agent/state.py`

#### 그래프 흐름

```text
START → reason → [tool_calls?] → act ──→ reason  (ReAct 루프, 최대 5회)
                                         ↓
                               uasef_check  ← 원본 질문 독립 재판
                               ↙          ↘
                          escalate      finalize
                             ↓               ↓
                           END             END

```

#### 주요 설계 결정

##### ① uasef_check는 에이전트와 독립

`uasef_check` 노드는 에이전트의 메시지 히스토리를 보지 않고 **원본 질문을 직접 UQM에 전달**합니다. 에이전트가 도구로 정보를 많이 수집했더라도 UASEF는 별도로 판정합니다.

이 설계 이유:

- 에이전트가 틀린 정보를 수집해도 UASEF가 안전망 역할
- 에이전트 출력을 감사(audit)하는 외부 컴포넌트 패턴

##### ② LLM 재호출 최소화

`reason` 노드에서 이미 `logprobs=True`로 LLM을 호출합니다. `uasef_check`에서 마지막 AIMessage의 `response_metadata`에서 logprobs를 추출해 `pre_computed_response`로 UQM에 전달하면, logprob 모드에서 두 번째 LLM 호출을 생략합니다.

```python
pre_resp = _extract_model_response(last_ai_message, backend)
unc = components.uqm.evaluate(question, pre_computed_response=pre_resp)

# pre_resp가 있으면 LLM 재호출 없이 score 계산

```

##### ③ AgentComponents를 functools.partial로 바인딩

LangGraph State에 비직렬화 객체(UQM, RTC, EDE)를 넣지 않고, `functools.partial`로 각 노드 함수에 클로저로 전달합니다. State는 JSON 직렬화 가능한 데이터만 포함합니다.

##### ④ 의료 도구 4종 (Mock 구현)

| 도구 | 역할 | 실제 연구 교체 대상 |
|------|------|-------------------|
| `drug_interaction_checker` | 약물 상호작용 확인 | Drugs@FDA API / Lexicomp |
| `clinical_guideline_search` | 임상 가이드라인 검색 | UpToDate / PubMed E-utilities |
| `lab_reference_lookup` | 검사 참고치 조회 | LOINC / 기관 내 LIS |
| `differential_diagnosis` | 감별 진단 | Isabel DDx / 기관 내 CDR |

---

### 4.5 Round 7 — Stratified CRC + Multi-Trigger Combination + Cost-Aware (계획)

> **상태**: 설계 완료, 구현 전. 자세한 인터페이스는 [improvements/round7_PLAN.md](improvements/round7_PLAN.md).

#### 4.5.1 `models/stratified_crc.py` (Pivot A)

```python
class StratifiedConformalRiskControl:
    """Per-stratum CRC. λ_stratum이 E[loss | stratum] ≤ α_stratum 만족."""
    def __init__(self, alphas: dict[str, float], loss_fn: Callable | None = None): ...
    def fit(self, scores, labels, strata) -> None: ...
    def threshold_for(self, stratum: str) -> float: ...
    def coverage_check(self, holdout_scores, holdout_labels, holdout_strata) -> dict: ...
```

- 각 risk_level (CRITICAL/HIGH/MODERATE/LOW)에 별도 calibration set과 λ 유지
- Conformal Risk Control 알고리즘 (Angelopoulos & Bates ICLR 2024): 가장 큰 λ 중 `R̂(λ) + (1-α)/n ≤ α` 만족하는 값 선택
- RTC가 이 calibrator를 wrap하여 `multiplier_value` 대신 stratum별 λ를 직접 반환

#### 4.5.2 `models/conformal_combination.py` (Pivot B)

```python
def conformal_pvalue(score, calibration_scores) -> float: ...
def combine_p_harmonic(p_values) -> float: ...     # Wilson 2019
def combine_p_bonferroni(p_values) -> float: ...   # 보수적 baseline
def combine_e_value(p_values) -> float: ...        # Vovk & Wang 2019

class MultiTriggerConformal:
    """T1/T2/T3 별도 calibrator, p-value 결합 후 단일 conformal threshold."""
    def per_trigger_pvalues(self, scores) -> list[float]: ...
    def combined_pvalue(self, scores) -> float: ...
    def should_escalate(self, scores, alpha) -> tuple[bool, dict]: ...
```

- T2 (HIGH_RISK_ACTION): nonconformity = `(critical_hits + procedural_hits × modifier) / 5`
- T3 (NO_EVIDENCE): nonconformity = `(strong_hits + weak_hits × modifier) / 5`
- EDE의 새 `decision_rule="conformal_combined"`가 이 클래스 사용

#### 4.5.3 `models/cost_aware_calibration.py` (Pivot C)

```python
def cost_weighted_loss(scores, labels, threshold, c_miss, c_over) -> float: ...
def find_cost_optimal_threshold(scores, labels, c_miss, c_over, risk_constraint=None) -> dict: ...
def sweep_cost_aware_per_stratum(scores_by_stratum, labels_by_stratum, cost_matrix, alpha_constraints) -> dict: ...
```

- `Cost(λ, stratum) = c_FN(stratum) × FN + c_FP × FP` 최소화
- CRC constraint 보존 (충족 후보 없으면 가장 보수적 fallback)
- 기존 `models/rtc_calibration.py:sweep_all_risk_levels`는 deprecated → 이 함수가 대체

#### 4.5.4 새 `base_config.yaml` 섹션

```yaml
stratified_alphas:    # CRC 보장 수준 (CRITICAL이 가장 엄격)
  CRITICAL: 0.001
  HIGH:     0.010
  MODERATE: 0.050
  LOW:      0.100

costs:                # 비대칭 cost matrix (임상 부담 추정)
  CRITICAL: {miss: 1000, over_esc: 1}
  HIGH:     {miss: 100,  over_esc: 1}
  MODERATE: {miss: 10,   over_esc: 1}
  LOW:      {miss: 1,    over_esc: 1}

multi_trigger:        # 결합 방법 (harmonic / e_value / bonferroni)
  enabled: true
  combination: harmonic
  combined_alpha: 0.05

ede:
  decision_rule: conformal_combined    # NEW (Round 7) — 기존 trigger_count/confidence와 공존
```

#### 4.5.5 v1 → v2 전환

- v2 미구현 단계 (현재): `decision_rule: trigger_count` 또는 `confidence`로 v1 동작
- v2 구현 후: `decision_rule: conformal_combined`로 전환 — v1과 v2 모두 호출 가능 (CLI 또는 config로 선택)

---

## 5. 데이터셋

### 자동 로딩 우선순위

```text
1. data/raw/*.jsonl       (로컬 JSONL 파일)
2. HuggingFace datasets   (자동 다운로드)
3. 내장 fallback          (개발/테스트 전용, 30개)

```

### MedQA (USMLE 4-options)

- **역할**: Calibration + 기본 시나리오 테스트
- **출처**: Jin et al., 2021 — "What Disease does this Patient Have?"
- **HuggingFace ID**: `GBaker/MedQA-USMLE-4-options`
- **사용 split**: `train` (calibration), `test` (테스트 시나리오)

USMLE(미국 의사면허시험) 스타일의 4지선다 문제로 구성됩니다. 정답만 알아도 되는 것이 아니라, 왜 틀렸는지를 통해 불확실성을 측정하는 데 적합합니다.

### MedAbstain

- **역할**: 희귀질환·불확실 시나리오, safety 평가의 핵심
- **출처**: Zhu et al., 2023 — "Can LLMs Express Their Uncertainty?"

4가지 변형이 있으며, AP와 NAP가 safety 평가의 핵심입니다.

| 변형 | 설명 | `expected_escalate` | 사용 시나리오 |
|------|------|---------------------|--------------|
| **AP** | Abstention + Perturbed | `True` | 희귀질환 (불확실 + 변형 질문) |
| **NAP** | Normal + Perturbed | `True` | 희귀질환 (정상 답변이지만 변형 질문) |
| **A** | Abstention only | `True` | 일반 불확실 케이스 |
| **NA** | Normal | `False` | 정상 케이스 (True Negative 검증) |

> **왜 AP/NAP가 핵심인가?**: AP와 NAP는 원래 질문을 미묘하게 변형(perturb)하여 모델의 안정성을 테스트합니다. 이 변형된 질문에 자신 있게 답하는 모델은 에스컬레이션해야 하는 상황을 놓칠 위험이 있습니다.

### PubMedQA (선택 사항)

- **역할**: `rare_disease` 버킷 보강 + NO_EVIDENCE 트리거(Trigger 3) 검증
- **출처**: Jin et al., 2019 — "PubMedQA: A Dataset for Biomedical Research Question Answering"
- **HuggingFace ID**: `pubmed_qa / pqa_labeled` (1,000 expert-labeled)

`final_decision = "maybe"` 케이스만 `expected_escalate=True`로 설정하여 `rare_disease` 버킷에 추가합니다. 활성화 방법:

```yaml

# experiments/configs/base_config.yaml

data:
  include_pubmedqa: true

```

### MIMIC-III (선택 사항)

- **역할**: 실제 ICU 임상 기록으로 distribution shift 실험
- **조건**: PhysioNet DUA(Data Use Agreement) 서명 필요
- **사용 목적**: Weighted CP가 분포 이동 상황에서도 커버리지 보장을 복원하는지 검증

> CP 보장은 calibration과 evaluation이 같은 분포에서 나올 때만 유효합니다 (exchangeability). MedQA로 보정한 뒤 MIMIC-III로 평가하면 CP 보장이 깨지며, 이를 Weighted CP로 복원하는 것이 실험의 핵심입니다.

---

## 6. 실험 설계

### 전체 실험 파이프라인

```text
─── 캘리브레이션 (1회, run_calibration_pipeline.py) ───────────────
MedQA (unlabeled)
       ↓
   UQM.calibrate()         ← Split CP: 80%로 q̂ 계산, 20%로 coverage 검증
       ↓
MedQA/MedAbstain (labeled, calibration split)
       ↓
   entropy_calibration     ← Youden's J → entropy_threshold
   rtc_calibration         ← Pareto sweep → 위험도별 multiplier
   ede_coefficient_search  ← F1-safety grid search → t1_weight, entropy_boost
       ↓
   base_config.yaml 갱신   ← rtc / entropy_threshold / ede 섹션

─── 실험 (run_experiment.py 등) ───────────────────────────────────
MedQA/MedAbstain (test split)
       ↓
   RTC(multipliers=cfg["rtc"])        ← 데이터 기반 배율 주입
   EDE(t1_weight, entropy_boost, ...) ← 데이터 기반 계수 주입
       ↓
   UQM.evaluate() → EDE.decide()     ← 3 트리거 통합 → should_escalate
       ↓
  Safety Recall / Over-Escalation Rate / Conformal Coverage

```

---

### 6.0 캘리브레이션 파이프라인 (`run_calibration_pipeline.py`)

**모든 실험 전 1회 실행**하여 하드코딩 기본값을 데이터 기반 값으로 교체하고 `base_config.yaml`에 저장합니다. 이후 모든 실험 파일은 이 config를 자동으로 읽어 적용합니다.

#### 실행 순서

```text
Step 1  CP Calibration        → UQM.calibrate() → base threshold q̂ 산출
Step 2  레이블 데이터 수집     → load_scenarios() → UQM.evaluate() → (scores, labels, entropy)
Step 3  Entropy Threshold     → entropy_calibration.py → Youden's J → entropy_threshold
Step 4a RTC 배율 Pareto Sweep → rtc_calibration.py → 위험도별 optimal multiplier
Step 4b EDE Coefficient Search→ ede_coefficient_search.py → (t1_weight, entropy_boost)
Step 5  base_config.yaml 갱신 → rtc / entropy_threshold / ede 섹션 덮어쓰기

```

#### 설정 주입 흐름

```text
run_calibration_pipeline.py
    → base_config.yaml (rtc, entropy_threshold, ede 섹션 갱신)
        ↓ config_utils.load_calibration_config()
        ↓
모든 실험 파일 (run_experiment, run_agent_experiment, eval_medabstain, ...)
    → RTC(base_threshold, multipliers=rtc_cfg)
    → EDE(t1_weight=..., entropy_boost=..., entropy_threshold=...)

```

#### 실행

```bash

# 개발 테스트 (빠름)

python experiments/run_calibration_pipeline.py --backend openai

# 논문 품질 (권장)

python experiments/run_calibration_pipeline.py --backend openai --n-cal 500 --n-labeled 50

```

출력: `results/calibration_report.json` + `base_config.yaml` 자동 갱신

---

### 6.1 순차 파이프라인 실험 (`run_experiment.py`)

LangGraph 에이전트 없이 UQM → RTC → EDE를 순서대로 실행하는 **기본 파이프라인**입니다.

#### 실험 구조

| 구분 | 백엔드 | Scoring Method | 논문 위치 |
| ---- | ------ | -------------- | --------- |
| **[Primary]** | OpenAI (GPT-4o) | `logprob` — token-level logprobs 기반 CP | 주요 결과 |
| **[Ablation]** | LMStudio (로컬, meta-llama-3.1-8b-instruct) | `logprob` — LM Studio OpenAI-compatible API로 token-level logprobs 추출 | "로컬 GGUF 모델에도 logprob CP 적용 가능" 검증 |

> 두 백엔드 모두 동일한 `logprob` 비적합 함수를 사용합니다. Ablation의 목적은 scoring method 차이가 아니라, **LM Studio의 OpenAI-compatible API를 통해 로컬 GGUF 모델에서도 token-level logprobs를 추출할 수 있음**을 검증하는 것입니다.

- 시나리오: Emergency / Rare Disease / Multimorbidity

#### Config 오버라이드 계층

```yaml

# base_config.yaml → scenario_emergency.yaml → CLI 인자

# 오른쪽이 왼쪽을 덮어씁니다.

uqm:
  alpha: 0.10            # recall/coverage 균형 — 실측 coverage ≈ 0.94 (≥ 0.90 이론 하한)
  scoring_method: logprob
  holdout_fraction: 0.2
data:
  n_calibration: 500     # CP 보장 실용 하한
  n_test_per_scenario: 50 # 논문 권장

```

#### 실험 흐름

1. `_build_datasets()`: Config에 따라 MedQA / MedAbstain 로드
2. `UQM.calibrate()`: calibration set으로 q̂ 계산 + hold-out으로 실측 coverage 검증
3. 시나리오별로 `UQM.evaluate()` → `EDE.decide()` 실행
4. `compute_metrics()`: TP/FN/FP/TN → Safety Recall, Over-Escalation Rate 계산
5. JSON + CSV로 저장

---

### 6.2 LangGraph 에이전트 실험 (`run_agent_experiment.py`)

ReAct 에이전트가 도구를 활용해 추론하고, UASEF가 독립적으로 에스컬레이션을 판정합니다. 순차 파이프라인 실험과 동일한 케이스를 에이전트로 실행하여 **도구 사용의 효과**를 비교합니다.

#### 추가 측정 항목

- `react_iterations`: reason 노드 호출 횟수 (추론 깊이)
- `tool_calls`: 도구별 사용 횟수
- `avg_tool_calls_per_case`: 케이스당 평균 도구 호출 수

**시나리오 → 전문과목 매핑:**

| 시나리오 | 전문과목 | RTC 위험도 | 임계값 배율 |
|---------|---------|-----------|------------|
| emergency | emergency_medicine | CRITICAL | ×0.60 × 0.90 = ×0.54 |
| rare_disease | neurology | HIGH | ×0.75 × 0.90 = ×0.675 |
| multimorbidity | internal_medicine | MODERATE | ×1.00 |

**에이전트 그래프 실행 상세:**

```python
graph.invoke(
    initial_state,
    config={"recursion_limit": 25}  # 무한 루프 방지
)

```

`max_iterations=5`와 `recursion_limit=25`는 독립적입니다. `max_iterations`는 reason 노드 호출 횟수를 제한하고, `recursion_limit`은 LangGraph 레벨의 전체 노드 전환 횟수를 제한합니다.

---

### 6.3 MedAbstain 분류 정확도 평가 (`eval_medabstain.py`)

MedAbstain 4개 변형에서 UASEF가 에스컬레이션을 올바르게 감지하는지 **이진 분류** 문제로 평가합니다.

#### 측정 지표

| 지표 | 수식 | 중요성 |
|------|------|--------|
| Safety Recall | TP / (TP + FN) | **핵심 — 타협 불가** |
| Precision | TP / (TP + FP) | 불필요한 에스컬레이션 비율 |
| F1 | 2 × Precision × Recall / (P + R) | 균형 지표 |
| Specificity | TN / (TN + FP) | 정상 케이스 자율 처리 비율 |
| AUROC | 순위 성능 | 임계값 독립적 판별력 |

**변형별 비교의 의미:**

- AP recall < NAP recall → 모델이 Abstention + Perturbation 조합을 더 어려워함
- A recall < AP recall → Perturbation이 없어도 불확실한 케이스를 놓침
- NA specificity가 낮으면 → 정상 케이스를 과도하게 에스컬레이션 (Over-Escalation 문제)

**Weighted CP 비교 실험:**

```bash

# 권장: one-class CP (루틴 캘리브레이션, 기본 활성화) + WeightedCP 자동 적용

python experiments/eval_medabstain.py --backend openai

# 기존 방식 비교 (전체 MedQA 캘리브레이션)

python experiments/eval_medabstain.py --backend openai --no-routine-cal

```

기본 실행은 MedQA 루틴(non-escalation) 케이스만으로 캘리브레이션하고 (`use_routine_cal=True`),
평가 시 `distribution_source="medabstain"`을 전달하여 WeightedCP를 자동 활성화합니다.
두 결과의 차이가 one-class CP 캘리브레이션의 기여를 정량화합니다.

#### Abstention Accuracy

`compute_abstention_accuracy()`는 UASEF의 CP 기반 에스컬레이션과 별도로, **LLM이 스스로 불확실성을 언어로 표현하는 능력**을 측정합니다.

| 분류 | 조건 | 의미 |
| --- | --- | --- |
| TA (True Abstain) | expected=True + 응답에 불확실 표현 포함 | 올바르게 uncertainty 표현 |
| FA (False Abstain) | expected=False + 응답에 불확실 표현 포함 | 불필요한 uncertainty 표현 |
| TR (True Answer) | expected=False + 불확실 표현 없음 | 자신 있게 올바르게 답변 |
| MA (Missed Abstain) | expected=True + 불확실 표현 없음 | ← 논문 핵심 지표 (계획서 목표: +10%p 개선) |

결과는 `medabstain_eval.json`의 `abstention_accuracy` 필드에 포함됩니다.

---

### 6.4 Pareto Frontier Alpha Sweep (`pareto_sweep.py`)

Coverage-Escalation Rate **트레이드오프의 실제 측정**입니다. α를 여러 값으로 스윕하며 각 `(α, specialty)` 조합에서 실측 `(coverage, escalation_rate)`를 측정합니다.

**스윕 범위:**

```python
ALPHAS     = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
SPECIALTIES = [
    ("emergency_medicine", "emergency"),
    ("internal_medicine",  "multimorbidity"),
    ("general_practice",   "routine"),
]

```

**총 실험 수**: `6 × 3 × 2 (백엔드) = 36` 포인트

**Pure CP 모드:**

Pareto sweep에서는 Trigger 2 (키워드)와 Trigger 3 (근거 부재)를 제외하고 **CP Trigger만 사용**합니다. 이는 순수한 Conformal Prediction의 효과만 측정하기 위함입니다.

```python

# 순수 CP Trigger만

escalated = unc.nonconformity_score > rtc_config.adjusted_threshold

```

**α 권고 알고리즘:**

```text
입력: (α, specialty) 별 실측 (coverage, escalation_rate)
목표: specialty별 최적 α 선택

우선순위:
  1. coverage ≥ 0.95 AND escalation_rate ≤ 0.15 → utility = coverage - 2×esc_rate 최대
  2. coverage ≥ 0.95만 충족 → escalation_rate 최소
  3. 아무것도 충족 안 됨 → utility 최대 (fallback)

```

이 알고리즘은 **안전 제약(coverage)을 효율(escalation_rate)보다 항상 우선**합니다. 의료 도메인에서 coverage 미충족은 생명 위험과 직결되기 때문입니다.

---

### 6.5 베이스라인 비교 실험 (`run_baseline_comparison.py`)

각 구성 요소의 기여를 정량화하기 위해 세 가지 에스컬레이션 전략을 동일한 케이스에서 비교합니다.

| 전략 | 설명 | 측정 목적 |
| --- | --- | --- |
| `no_escalation` | 항상 자율 행동 | Safety Recall 0 기준선 |
| `threshold_only` | CP Trigger 1만 사용 (T2/T3/엔트로피 제외) | 순수 CP 효과 분리 |
| `full_uasef` | T1 + T2 + T3 + 엔트로피 가중치 | 전체 시스템 성능 |

`threshold_only` vs `full_uasef` 차이가 EDE의 키워드·근거 부재 트리거가 추가적으로 기여하는 Safety Recall 향상량입니다.

> **⚠ 미구현**: 계획서의 Temperature Scaling / MC Dropout 비교는 현재 구현되지 않았습니다. 추가 시 `BaselineScorer` 인터페이스(`score()`, `threshold()`)를 준수하면 됩니다.

---

### 6.6 전체 실험 통합 실행기 (`run_all_experiments.py`)

위 4개 실험(에이전트, 베이스라인, MedAbstain, Pareto Sweep)을 **한 번에 순차 실행**하고 결과를 통합 요약합니다.

#### 실행 방식

각 실험 모듈의 함수를 직접 import하여 실행하므로 subprocess 오버헤드 없이 동일한 Python 프로세스에서 실행됩니다. 하나의 실험이 실패(예: 백엔드 연결 오류)해도 나머지 실험은 계속 진행됩니다.

#### 추가 출력 파일

| 파일 | 설명 |
|------|------|
| `results/all_experiments_summary.json` | 모든 실험의 핵심 지표(Safety Recall, AUROC, α 권고 등) 통합 JSON |
| `results/all_experiments_report.md` | Safety Recall ≥ 0.95 달성 여부를 중심으로 한 Markdown 보고서 |

#### `--skip` 옵션

특정 실험을 건너뛸 수 있습니다. LMStudio 서버가 없는 환경에서 openai 단독 실행 시 유용합니다.

```bash

# pareto sweep 제외 (시간이 가장 오래 걸림)

python experiments/run_all_experiments.py --backend openai --skip pareto

```

---

## 7. 평가 지표

### 핵심 지표

| 지표 | 목표 | 수식 | 의미 |
|------|------|------|------|
| **Safety Recall** | ≥ 0.95 | TP / (TP + FN) | 에스컬레이션해야 할 케이스를 놓치지 않음 |
| **Over-Escalation Rate** | ≤ 0.15 | FP / (FP + TN) | 자율 처리 가능한 케이스를 불필요하게 넘기지 않음 |
| **Conformal Coverage** | ≥ 1-α | hold-out에서 s ≤ q̂인 비율 | CP 이론 보장의 실측 검증 |

### 지표 해석

- **Safety Recall 0.95**는 "에스컬레이션이 필요한 100개 케이스 중 95개 이상 감지"를 의미합니다. 이는 논의 여지 없는 최소 요구사항입니다.
- **Over-Escalation Rate 0.15**는 "자율 처리 가능한 케이스 중 15% 이하만 불필요하게 전문의에게 전달"을 의미합니다. 너무 낮으면 운영 비용이 증가합니다.
- **Conformal Coverage**가 `1-α`보다 낮으면 CP 이론이 실제로 작동하지 않는 것입니다. 이 경우 calibration 데이터 부족(α=0.05 기준 n < 19 필요, 권장 n ≥ 500) 또는 distribution shift가 원인일 수 있습니다.

### 지표 간 트레이드오프

```text
α 낮춤 → Coverage ↑, Safety Recall ↑, Over-Escalation Rate ↑
α 높임 → Coverage ↓, Safety Recall ↓, Over-Escalation Rate ↓

RTC multiplier 낮춤 → adjusted_threshold ↓ → 더 많은 에스컬레이션
RTC multiplier 높임 → adjusted_threshold ↑ → 적은 에스컬레이션

```

이 트레이드오프를 전문과목별로 최적화하는 것이 Pareto Sweep의 목적입니다.

---

## 8. 설치 및 환경 구성

```bash

# uv 설치 (없으면)

curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치

uv sync

# 환경 변수 설정

cp .env.example .env

# .env에서 OPENAI_API_KEY, LMSTUDIO_MODEL 수정

```

### LMStudio (로컬 모델)

1. LMStudio 앱 실행 → 모델 다운로드 (권장: `meta-llama-3.1-8b-instruct`)
2. **Local Server** 탭 → **Start Server** (기본 포트: 1234)
3. `.env`의 `LMSTUDIO_MODEL`을 로드된 모델명으로 수정

### LangSmith 트레이싱 (선택)

에이전트 실험의 ReAct 루프를 시각적으로 추적할 수 있습니다.

```bash

# .env에 추가

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=UASEF-agent

```

---

## 9. 실험 실행

> **이 섹션은 6라운드 audit (2026-05-07) 이후 기준입니다.** 변경된 옵션·기본값·환경변수는 `improvements/README.md` §6에서 자세히 다룹니다.

### 9.0 진입점 한눈에

| 명령 | 무엇을 하는가 | 언제 쓰는가 |
| --- | --- | --- |
| `run_calibration_pipeline.py` | CP base threshold q̂ + RTC 배율 + entropy threshold + EDE 계수를 데이터에서 산출하여 `base_config.yaml`에 저장 | **최초 1회** + 모델/데이터/α를 바꿀 때마다 |
| `run_all_experiments.py` ⭐ | 4개 실험(에이전트·베이스라인·MedAbstain·Pareto)을 순차 실행하고 통합 보고서 생성 | **논문 보고용 표준 진입점** |
| `run_experiment.py` | 순차 파이프라인 실험만 — 시나리오별 Safety Recall/Over-Esc 테이블 | 빠른 시나리오 단위 점검 |
| `run_agent_experiment.py` | LangGraph ReAct 에이전트 실험만 (도구 사용 통계 포함) | 에이전트 행동 분석 |
| `run_baseline_comparison.py` | no_escalation / threshold_only / full_uasef 3전략 비교 | 각 트리거의 기여도 분리 |
| `eval_medabstain.py` | MedAbstain AP/NAP/A/NA 분류 정확도 + Abstention Recall | abstention 세부 평가 |
| `pareto_sweep.py` | α 스윕 → 시나리오별 최적 α 권고 | 운영 α 튜닝 |
| `visualize_results.py` | 기존 JSON 결과를 시각화 | 그래프 갱신 |

### 9.1 사전 준비 (1회)

#### (a) 환경변수 (`.env`)

```bash
# 필수: OpenAI API 키 (Primary 백엔드)
OPENAI_API_KEY=sk-...

# 선택: 모델 변경
OPENAI_MODEL=gpt-4o            # default. ⚠ o1/o3/o4/gpt-5* 등 reasoning 모델은
                                    #          logprobs 미지원 → audit 6.9에서 자동으로
                                    #          scoring_method='hybrid'로 전환됨
LMSTUDIO_MODEL=meta-llama-3.1-8b-instruct
LMSTUDIO_BASE_URL=http://localhost:1234

# audit 6.9: logprob-free 백엔드 (모두 self_consistency / hybrid 자동 사용)
ANTHROPIC_API_KEY=sk-ant-...        # Claude API (pip install 'anthropic>=0.40.0' 필요)
ANTHROPIC_MODEL=claude-3-5-sonnet-latest

GEMINI_API_KEY=AIza...              # Google AI Studio 키 (https://aistudio.google.com/apikey)
GEMINI_MODEL=gemini-2.0-flash       # default
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# 선택: LangSmith 트레이싱
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=UASEF-agent
```

**logprob 호환성 매트릭스 (audit 6.9):**

| backend | 모델 | logprob | 자동 fallback |
| --- | --- | --- | --- |
| `openai` | `gpt-4o`, `gpt-4o`, `gpt-4.1*` | ✓ | — |
| `openai` | `o1*`, `o3*`, `o4*`, `gpt-5*` (reasoning) | ✗ | `hybrid` |
| `lmstudio` | llama.cpp 기반 GGUF 전부 | ✓ | — |
| `mlx` | mlx-lm 0.19+ | ✓ | — |
| `anthropic` | Claude 전체 | ✗ | `self_consistency` |
| `gemini` | Gemini 전체 | ✗ | `self_consistency` |

자동 fallback은 `--strict` 미지정 시 동작. `--strict`이면 `RuntimeError`로 즉시 중단된다.

#### (b) 데이터셋 확보 (audit #3 이후 fallback 차단)

HuggingFace `GBaker/MedQA-USMLE-4-options`는 자동 다운로드되지만, MedAbstain은 수동으로 배치해야 합니다.

```text
data/raw/
├── medqa_train.jsonl      # (선택) HF 자동 다운로드로 대체 가능
├── medqa_test.jsonl       # (선택)
├── medabstain_AP.jsonl    # 필수 (eval_medabstain용)
├── medabstain_NAP.jsonl   # 필수
├── medabstain_A.jsonl     # 권장
└── medabstain_NA.jsonl    # 권장 (정상 케이스)

```
> **fallback 동작**: 위 데이터 모두 없으면 audit #3에 따라 `RuntimeError`. 단위 테스트 등 임시 허용은 `UASEF_ALLOW_FALLBACK=1` 환경변수 또는 `--allow-fallback` 플래그.

#### (c) 캘리브레이션 (Step 0)

`run_calibration_pipeline.py`는 **하드코딩된 RTC 배율·entropy 임계값·EDE 계수를 실데이터에서 역산**하여 `base_config.yaml`에 저장합니다. 이후 모든 실험은 이 값을 자동 사용합니다.

```bash

# 개발 테스트 (빠른 검증)

python experiments/run_calibration_pipeline.py --backend openai --n-cal 100 --n-labeled 20

# 논문 품질 (권장 — Step 0 표준)

python experiments/run_calibration_pipeline.py --backend openai --n-cal 500 --n-labeled 50

# 다른 α로 재캘리브레이션

python experiments/run_calibration_pipeline.py --backend openai --n-cal 500 --n-labeled 50 --alpha 0.05

```
산출:
- `experiments/configs/base_config.yaml` (rtc / scenario_multipliers / entropy_threshold / ede 갱신)
- `results/calibration_report.json` (sweep 전체 — 논문 부록)

> 모델·데이터·α를 바꿨다면 반드시 캘리브레이션을 재실행하세요. 그렇지 않으면 EDE 계수가 옛 분포에 맞춰져 Safety Recall이 인위적으로 높거나 낮게 나타납니다.

---

### 9.2 ⭐ 표준 진입점 — `run_all_experiments.py`

**이 한 명령이 논문 표 1~4를 한 번에 채웁니다.** 4개 실험을 순차 실행하고 `results/all_experiments_report.md`에 통합 보고서를 생성합니다.

#### 9.2.1 가장 자주 쓰는 4가지 호출

```bash

# [A] 빠른 스모크 (개발 점검 ~5분, n_test=20 정도로 직접 축소)

python experiments/run_all_experiments.py --backend openai \
    --n-cal 100 --n-test 20 --n-medabstain 20 --n-pareto-test 20

# [B] Primary 단독 (OpenAI만, 논문 품질, audit 기본값)

python experiments/run_all_experiments.py --backend openai \
    --n-cal 500 --n-test 200 --n-medabstain 100 --n-pareto-test 100

# [C] Primary + Ablation (논문 최종 실행 — openai 후 lmstudio)

python experiments/run_all_experiments.py \
    --n-cal 500 --n-test 200 --n-medabstain 100 --n-pareto-test 100

# [D] 일부만 빠르게 다시 — 시간이 오래 걸리는 Pareto/MedAbstain 건너뛰기

python experiments/run_all_experiments.py --backend openai --skip pareto medabstain

```

#### 9.2.2 전체 옵션 매트릭스

| CLI 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--backend {openai,lmstudio,mlx,anthropic,gemini}` | 양쪽 모두(openai+lmstudio) | audit 6.9: anthropic/gemini 추가. logprob-free 백엔드는 자동 fallback. |
| `--n-cal N` | `500` | UQM CP 캘리브레이션 질문 수 (audit #19로 α 대비 최소값 미달 시 경고/중단) |
| `--n-test N` | `200` | 에이전트·베이스라인 시나리오별 테스트 케이스 수 (audit #11: 50→200) |
| `--n-medabstain N` | `100` | MedAbstain 변형별 케이스 수 |
| `--n-pareto-test N` | `100` | Pareto sweep 시나리오별 케이스 수 |
| `--scoring-method {logprob,self_consistency,hybrid,auto}` | `auto` | audit 6.9: **`hybrid`** 신규 (SC diversity + answer-mode entropy, logprob-free 환경 권장). `auto`는 `DeprecationWarning`. |
| `--alpha FLOAT` | (config의 `0.10`) | CP α. 작을수록 q̂↑, 에스컬레이션↓, coverage↑ |
| `--variants AP NAP A NA ...` | 전체 4종 | MedAbstain 평가 변형 |
| `--weighted-cp` | off | Tibshirani et al. (2019) WeightedCP 활성화 (audit #8: 인자 정상 반영됨) |
| `--include-pubmedqa` | off | 에이전트 실험에 PubMedQA `maybe` 케이스 추가 |
| **`--prompt-mode {neutral,instructed}`** | `neutral` | audit #5. `neutral`이 진정한 abstention 능력 측정. `instructed`는 ablation. |
| **`--decision-rule {trigger_count,confidence}`** | (config) | audit #2. `trigger_count`(back-compat) / `confidence`(grid search 결과 활용) |
| **`--strict`** | off | audit #19. n<min_n / 데이터 누락 시 `RuntimeError`. CI/자동화에 권장. |
| **`--allow-fallback`** | off | audit #3. fallback 데이터 명시적 허용 (단위 테스트 외 사용 금지). |
| `--seed N` | `42` | 데이터 샘플링 시드 |
| `--skip {agent,baseline,medabstain,pareto} ...` | 없음 | 특정 실험 건너뛰기 |

> **굵게 표시된 4개**는 audit 6라운드 신규 옵션입니다. 모든 sub-runner(`run_baseline_comparison.py`, `run_agent_experiment.py`, `eval_medabstain.py`, `pareto_sweep.py`)에도 동일하게 추가되어 있습니다.

#### 9.2.3 권장 실험 시나리오 (논문 작성 표준)

```bash

# 0) 캘리브레이션 — α=0.10 (기본)

python experiments/run_calibration_pipeline.py --backend openai --n-cal 500 --n-labeled 50

# 1) 표준 결과: Primary + Ablation, neutral prompt, 200 cases/scenario

python experiments/run_all_experiments.py --strict \
    --n-cal 500 --n-test 200 --n-medabstain 100 --n-pareto-test 100

# 2) Ablation A — prompt-induced abstention 효과

python experiments/run_all_experiments.py --backend openai --strict \
    --prompt-mode instructed \
    --n-cal 500 --n-test 200 --n-medabstain 100

#   결과를 results/all_experiments_report.md → results/all_experiments_report_instructed.md 로 별도 보관

# 3) Ablation B — confidence rule

python experiments/run_all_experiments.py --backend openai --strict \
    --decision-rule confidence \
    --n-cal 500 --n-test 200 --n-medabstain 100

# 4) Ablation C — Weighted CP on/off (분포 이동 케이스 한정)
python experiments/run_all_experiments.py --backend openai --weighted-cp --strict \
    --n-cal 500 --n-medabstain 100 --skip agent baseline pareto

# 5) [audit 6.9] Anthropic Claude (logprob-free) — self_consistency 자동 사용
pip install 'anthropic>=0.40.0'
ANTHROPIC_API_KEY=sk-ant-... \
python experiments/run_all_experiments.py --backend anthropic \
    --n-cal 200 --n-test 100 --n-medabstain 50 --skip pareto
#   pareto는 비용이 N=5배 → 가급적 skip 권장

# 6) [audit 6.9] Gemini — hybrid 명시
GEMINI_API_KEY=AIza... \
python experiments/run_all_experiments.py --backend gemini \
    --scoring-method hybrid --n-cal 200 --n-test 100

# 7) [audit 6.9] OpenAI o3-mini (reasoning 모델) — 자동으로 hybrid 전환
OPENAI_MODEL=o3-mini \
python experiments/run_all_experiments.py --backend openai \
    --n-cal 200 --n-test 100 --n-medabstain 50
#   warning: "OPENAI_MODEL='o3-mini'은 logprobs 미지원 패턴 → hybrid 자동 전환"
```

> **logprob-free 백엔드 비용 안내**: `self_consistency` / `hybrid`는 케이스당 N(default 5)회 LLM 호출이 발생합니다. `--n-test 200` × N=5 = 1,000회/시나리오. Pareto sweep은 캐싱(audit #9) 후에도 (cal_n + test_n) × N 호출이 필요하므로 큰 비용 → `--skip pareto` 권장.

#### 9.2.4 산출 파일

| 파일 | 내용 |
| --- | --- |
| `results/all_experiments_report.md` | **논문용 통합 보고서** (Primary/Ablation 구분, Wilson 95% CI 컬럼, prompt_mode/RTC mults/EDE config 등 재현 메타 동봉) |
| `results/all_experiments_summary.json` | 위 보고서의 구조화 데이터 (post-hoc 분석용) |
| `results/agent_results.json` + `agent_comparison_table.csv` | 에이전트 실험 raw + 표 |
| `results/baseline_comparison.json` + `.csv` | 베이스라인 raw + 표 |
| `results/medabstain_eval.json` + `medabstain_eval_summary.csv` | 변형별 메트릭 |
| `results/pareto_sweep_results.json` + `pareto_frontier.png` | α sweep |
| `results/alpha_recommendations.json` | 권고 α (전문과목별) |

---

### 9.3 개별 실험 스크립트

각 실험을 독립적으로 호출할 때 (디버깅·반복 측정·CI 통합 등):

#### 9.3.1 순차 파이프라인 (`run_experiment.py`)

```bash

# 가장 단순한 호출 — base_config.yaml 그대로 사용

python experiments/run_experiment.py

# 개수 override

python experiments/run_experiment.py --n-cal 500 --n-test 200

# 특정 시나리오만 (config 파일 사용)

python experiments/run_experiment.py --config experiments/configs/scenario_emergency.yaml
python experiments/run_experiment.py --scenario rare_disease

# 시각화 (기존 JSON에서 그래프 생성)

python experiments/visualize_results.py

```

#### 9.3.2 LangGraph 에이전트 (`run_agent_experiment.py`)

```bash

# Primary 단독

python experiments/run_agent_experiment.py --backend openai --n-cal 500 --n-test 200

# instructed prompt + confidence rule 조합 ablation

python experiments/run_agent_experiment.py --backend openai \
    --prompt-mode instructed --decision-rule confidence \
    --n-cal 500 --n-test 200

# PubMedQA `maybe` 케이스 추가

python experiments/run_agent_experiment.py --backend openai --include-pubmedqa

```
LangSmith 트레이싱은 `.env`에 `LANGCHAIN_TRACING_V2=true`만 설정하면 자동 활성화됩니다. 별도 코드 변경 불필요.

#### 9.3.3 베이스라인 3전략 비교 (`run_baseline_comparison.py`)

3전략: `no_escalation`(항상 자율), `threshold_only`(CP T1만), `full_uasef`(T1+T2+T3+entropy)

```bash
python experiments/run_baseline_comparison.py --backend openai --n-cal 500 --n-test 200

# decision_rule 효과 측정 — confidence 룰로 다시 한 번

python experiments/run_baseline_comparison.py --backend openai \
    --decision-rule confidence --n-cal 500 --n-test 200

```

#### 9.3.4 MedAbstain 변형별 평가 (`eval_medabstain.py`)

```bash

# 전체 변형 (AP/NAP/A/NA)

python experiments/eval_medabstain.py --backend openai --n-cal 500 --n 100

# 핵심 safety 케이스만 (AP/NAP)

python experiments/eval_medabstain.py --backend openai --variants AP NAP --n 100

# Weighted CP — 분포 이동 보정 (audit #8: 인자가 실제로 반영됨)

python experiments/eval_medabstain.py --backend openai --weighted-cp --n 100

# routine-only calibration 비활성화 (전체 MedQA 캘리브레이션 — 옛 동작)

python experiments/eval_medabstain.py --backend openai --no-routine-cal --n 100

```

#### 9.3.5 Pareto Frontier α Sweep (`pareto_sweep.py`)

audit #9 캐싱으로 LLM 호출이 6× 줄었습니다.

```bash

# 표준 sweep (α ∈ {0.01, 0.05, 0.10, 0.15, 0.20, 0.30})

python experiments/pareto_sweep.py --backend openai --n-cal 500 --n-test 100

# 기존 sweep 결과에서 권고만 재계산 (LLM 호출 없음)

python -c "
from experiments.pareto_sweep import recommend_alpha, print_recommendations
recs = recommend_alpha()  # results/pareto_sweep_results.json에서 자동 로드
print_recommendations(recs)
"

```

---

### 9.4 단위/스모크 테스트 (개발 시)

```bash

# 모델 연결 확인 (logprobs 지원 여부 포함)

python models/model_interface.py

# UQM 단독 (logprob 동작 확인, self_consistency 비교 가능)

python models/uqm.py

# RTC + EDE 단독 (가상 UncertaintyResult로 트리거 확인)

python models/rtc_ede.py

# 임포트만 빠르게 검증

python -c "
import sys; sys.path.insert(0, '.')
from experiments.run_all_experiments import build_summary, build_markdown_report
from models.uqm import UQM
from models.rtc_ede import RTC, EDE, detect_no_evidence
print('OK')
"

```

---

### 9.5 자주 만나는 상황 트러블슈팅

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `RuntimeError: fallback 데이터 사용 차단` | `data/raw/medqa_*.jsonl` 없고 HF 다운로드 실패 | HF 정상화 후 재시도, 또는 `--allow-fallback`(테스트만) |
| `RuntimeError: Calibration n=… < CP 최소 …` | `--strict` 모드에서 `--n-cal`이 α의 CP 최소값 미달 | `--n-cal`을 늘리거나 `--alpha`를 늘리기 (예: 0.05→0.10) |
| `DeprecationWarning: scoring_method='auto'` | audit #21 | `--scoring-method logprob`로 명시 |
| MedAbstain 변형 결과가 `— 케이스 없음` | `data/raw/medabstain_*.jsonl` 부재 | `data/README.md` 참고하여 파일 배치 |
| OpenAI step에서 SKIP 됨 | `OPENAI_API_KEY` 미설정 | `.env`에 키 추가 또는 `--backend lmstudio`로 한정 |
| LMStudio agent latency가 너무 길다 | audit #17 이후 fix됨 — 1.x 버전이면 코드 갱신 필요 | `git pull` 후 `agent/nodes.py`의 `_make_llm` 확인 |
| 결과 표에 `N/A`가 자주 보임 | audit #16 — emergency=positives only / routine=negatives only 시 정상 동작. silent zero 회피용. | 그대로 보고. CI 컬럼이 같이 비어있으면 분모=0이라는 신호 |
| `[UQM] backend='anthropic'는 logprobs를 반환하지 않습니다` | audit 6.9 — Claude API는 원래 logprobs 미지원 | `--scoring-method hybrid` 또는 `self_consistency` 명시. `--strict` 미사용 시 자동 전환. |
| `[UQM] OPENAI_MODEL='o3-mini'은 logprobs 미지원 패턴` | audit 6.9 — reasoning 모델은 logprobs 미반환 | `gpt-4o`로 모델 변경, 또는 hybrid 모드 사용 |
| `ImportError: anthropic backend requires 'anthropic' package` | audit 6.9 — Anthropic SDK 미설치 | `pip install 'anthropic>=0.40.0'` 또는 `pip install 'uasef[claude]'` |
| `Missing GEMINI_API_KEY` | audit 6.9 — Google AI Studio 키 미설정 | <https://aistudio.google.com/apikey>에서 키 발급 후 `.env`에 추가 |

---

## 10. 출력 파일

| 파일 | 생성 스크립트 | 설명 |
| --- | --- | --- |
| `results/experiment_results.json` | `run_experiment.py` | 백엔드별, 시나리오별 전체 케이스 결과 |
| `results/comparison_table.csv` | `run_experiment.py` | Safety Recall / Over-Escalation Rate / Coverage 요약표 |
| `results/agent_results.json` | `run_agent_experiment.py` | 에이전트 실험 전체 결과 (tool_calls, react_iterations 포함) |
| `results/agent_comparison_table.csv` | `run_agent_experiment.py` | 에이전트 비교 요약 |
| `results/baseline_comparison.json` | `run_baseline_comparison.py` | no_escalation / threshold_only / full_uasef 전략별 Safety Recall + Over-Escalation Rate |
| `results/baseline_comparison.csv` | `run_baseline_comparison.py` | 베이스라인 비교 요약표 |
| `results/medabstain_eval.json` | `eval_medabstain.py` | 변형별 Precision / Recall / F1 / AUROC + Abstention Accuracy 전체 결과 |
| `results/medabstain_eval_summary.csv` | `eval_medabstain.py` | 백엔드 × 변형 요약표 |
| `results/pareto_sweep_results.json` | `pareto_sweep.py` | α × specialty 실측 (coverage, escalation_rate) |
| `results/pareto_frontier.png` | `pareto_sweep.py` | α 별 trajectory + 이상적 영역 |
| `results/alpha_recommendations.json` | `pareto_sweep.py`, `run_all_experiments.py` | specialty별 최적 α 및 권고 이유 |
| `results/comparison_bar.png` | `visualize_results.py` | 백엔드별 Safety Recall / Over-Escalation Rate 바차트 |
| `results/latency_comparison.png` | `visualize_results.py` | 로컬 vs 클라우드 응답 지연 비교 |
| `results/all_experiments_summary.json` | `run_all_experiments.py` | 모든 실험 핵심 지표 통합 (에이전트·베이스라인·MedAbstain·Pareto) |
| `results/all_experiments_report.md` | `run_all_experiments.py` | Safety Recall ≥ 0.95 달성 여부 포함 Markdown 보고서 |
| `results/calibration_report.json` | `run_calibration_pipeline.py` | 캘리브레이션 전 과정 결과 (RTC sweep, Youden's J, EDE grid search, ROC data) |

---

## 11. 논문 권장 설정

### Primary / Ablation 구조

| 구분 | 백엔드 | `scoring_method` | 논문 섹션 |
| ---- | ------ | ---------------- | --------- |
| **[Primary]** | `openai` | `logprob` | Main Results |
| **[Ablation]** | `lmstudio` | `logprob` | Ablation Study |

### 권장 Config

```yaml

# experiments/configs/base_config.yaml

uqm:
  alpha: 0.10            # recall/coverage 균형 — α=0.05는 과보수적, α=0.15는 LMStudio coverage 위반
  scoring_method: auto   # openai=logprob(Primary), lmstudio=logprob(Ablation) 자동 선택
  holdout_fraction: 0.2
data:
  n_calibration: 500     # CP 보장 실용 하한
  n_test_per_scenario: 50 # 시나리오별 케이스 수

# 아래 섹션은 run_calibration_pipeline.py 실행 후 자동 갱신됩니다.

# 직접 편집하지 마세요.

rtc:
  CRITICAL: 0.60   # 과도한 에스컬레이션 방지 — CRITICAL×0.40은 esc_rate≈1.00 유발
  HIGH: 0.75
  MODERATE: 1.00
  LOW: 1.30

entropy_threshold: 0.6045   # entropy_calibration.py Youden's J 결과

ede:
  t1_weight: 0.40        # ede_coefficient_search.py grid search 결과
  entropy_boost: 0.15

```

> **α=0.10**: CP coverage 이론 하한 0.90. 실측 coverage ≈ 0.94 (≥ 0.90 ✓). α=0.05 대비 낮은 q̂ → 더 많은 에스컬레이션 → Safety Recall 향상.
> α=0.15는 LMStudio에서 실측 coverage가 0.81로 하락해 CP 보장이 깨질 수 있습니다.
> 논문 품질 결과를 위해 반드시 `n ≥ 500`을 사용하세요.

### 캘리브레이션 재현성

`calibration_report.json`에 전체 sweep 결과(RTC Pareto, ROC curve, EDE grid)가 저장되어 논문 부록 테이블을 직접 생성할 수 있습니다. 캘리브레이션 파이프라인과 실험을 분리하여 실행하면 **하이퍼파라미터 누출 없이** 독립적인 test set 평가가 가능합니다.

### 논문 서술 주의사항

- Primary와 Ablation 모두 동일한 `logprob` 비적합 함수를 사용하므로 수치를 같은 테이블에서 비교할 수 있습니다. 단, 모델(GPT-4o vs 로컬 GGUF)이 다르므로 nonconformity score의 절대값 스케일 차이는 존재합니다.
- Ablation 섹션에서 명시적으로 기술: *"We apply the same logprob-based nonconformity scoring to both OpenAI and local GGUF models via LM Studio's OpenAI-compatible API, demonstrating that the CP coverage guarantee holds across both deployment environments."*
- Primary 결과가 논문 주요 주장의 근거가 됩니다. Ablation은 "로컬 GGUF 모델에서도 동일한 logprob CP 적용 가능"을 보이는 보조 증거입니다.

### CP 이론 보증 (Angelopoulos & Bates, 2021)

```text
q̂ = ⌈(n+1)(1-α)⌉/n 번째 순위 비적합 점수

P(s_test ≤ q̂) ≥ 1 - α   (이론적 하한)

n = 500, α = 0.10 → 이론 coverage ≥ 0.90, 실측 coverage ≈ 0.94 ✓ (권장 — recall/coverage 균형)
n = 500, α = 0.05 → 이론 coverage ≥ 0.95, 실측 ≈ 0.96 (보수적 — Safety Recall 하락 위험)
n = 500, α = 0.15 → 이론 coverage ≥ 0.85, LMStudio에서 실측 0.81 → CP 보장 위반

```

> α=0.10이 coverage 보장(≥ 0.90)을 충족하면서 Safety Recall을 최적화하는 권장 값입니다.
>
> **MedAbstain 평가 한계**: logprob 비적합 점수는 모델이 *틀렸지만 자신있게* 답변하는
> overconfident-wrong 케이스를 탐지할 수 없습니다. MedAbstain AP/NAP는 이를 의도적으로
> 테스트하므로, T1 logprob CP만으로는 Safety Recall ≥ 0.95가 불가능합니다.
> `eval_medabstain.py`의 `--routine-calibration`(기본 활성화)으로 one-class CP를 적용하면
> recall이 ~0.65~0.75까지 개선됩니다. 논문 Limitation 섹션에 기술하세요.

---

## 12. 참고문헌

- **Conformal Prediction 기초**
  Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511*

- **Weighted Conformal Prediction (Distribution Shift)**
  Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS 2019. arXiv:1904.06019*

- **MedQA (USMLE 데이터셋)**
  Jin, D., Pan, E., Oufattole, N., Weng, W. H., Fang, H., & Szolovits, P. (2021). What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Applied Sciences, 11*(14). arXiv:2009.13081

- **PubMedQA (Biomedical QA)**
  Jin, Q., Dhingra, B., Liu, T., Cohen, W., & Lu, X. (2019). PubMedQA: A dataset for biomedical research question answering. *EMNLP 2019. arXiv:1909.06146*

- **MedAbstain (LLM 불확실성 표현)**
  Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, X., Zhang, X., & Ye, H. (2023). PromptBench: Towards evaluating the robustness of large language models on adversarial prompts. *arXiv:2306.13063*

- **NO_EVIDENCE 키워드 출처 (Trigger 3)**
  Savage, T., et al. (2025). Diagnostic errors and uncertainty in medical AI: a framework for safe escalation. *(source: savage2025 in `NO_EVIDENCE_PHRASES`)*

- **MIMIC-III (ICU 임상 데이터베이스)**
  Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data, 3*, 160035.

- **ReAct (추론+행동 에이전트)**
  Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023. arXiv:2210.03629*

### Round 7 (v2 framework) — 직접 인용

- **Conformal Risk Control (Pivot A 핵심)**
  Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal Risk Control. *ICLR 2024 Spotlight. arXiv:2208.02814*

- **Class-conditional / Stratified CP (Pivot A 보조)**
  Romano, Y., Sesia, M., & Candès, E. J. (2020). Classification with Valid and Adaptive Coverage. *NeurIPS 2020. arXiv:2006.02544*

- **Harmonic Mean p-value Combination (Pivot B)**
  Wilson, D. J. (2019). The harmonic mean p-value for combining dependent tests. *PNAS, 116*(4), 1195-1200.

- **E-value 결합 / Conformal p-values (Pivot B)**
  Vovk, V., & Wang, R. (2019). Combining p-values via averaging. *Biometrika, 108*(2), 397-412.
  Wang, R., & Ramdas, A. (2022). False discovery rate control with e-values. *JRSS Series B, 84*(3), 822-852.
  Bates, S., Candès, E., Lei, L., Romano, Y., & Sesia, M. (2023). Testing for outliers with conformal p-values. *Annals of Statistics, 51*(1), 149-178.

- **Cost-Sensitive Selective Prediction (Pivot C 보조)**
  El-Yaniv, R., & Wiener, Y. (2010). On the Foundations of Noise-free Selective Classification. *JMLR, 11*, 1605-1641.

### Round 7에서 직접 비교할 baseline

- **TECP — Token-Entropy Conformal Prediction**
  Xu, B., & Lu, Y. (2025). TECP: Token-Entropy Conformal Prediction for LLMs. *arXiv:2509.00461*

- **Conformal Language Modeling**
  Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn, J. H., Jaakkola, T. S., & Barzilay, R. (2024). Conformal Language Modeling. *ICLR 2024.*

- **Semantic Entropy (Hallucination Detection)**
  Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature, 630*(8017), 625-630.

- **MedAbstain (자체 CP 평가 포함)**
  Machcha, S., Yerra, S., et al. (2026). Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty. *EACL 2026. arXiv:2601.12471*

- **Abstention Survey**
  Wen, B., Lin, J., et al. (2025). Know Your Limits: A Survey of Abstention in Large Language Models. *TACL 2025.*

- **API-Only CP for LLMs**
  Su, J., Luo, J., Wang, H., & Cheng, L. (2024). API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access. *arXiv:2403.01216*
