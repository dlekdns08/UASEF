# Supplementary Materials — UASEF v1 sub-experiments

**Run timestamp:** `20260507-182038`  ·  **Backends:** `openai`, `lmstudio`
**Source paper:** [UASEF_Round7.md](../../paper/UASEF_Round7.md) (English) · [UASEF_Round7_KO.md](../../paper/UASEF_Round7_KO.md) (한국어)
**Template:** [UASEF_Round7_Supplementary.md](../../paper/UASEF_Round7_Supplementary.md) · [UASEF_Round7_Supplementary_KO.md](../../paper/UASEF_Round7_Supplementary_KO.md)

## B.1 Agent ReAct Behavior

| Backend | Accuracy | Safety Recall | Over-Esc | Avg Tool Calls | Avg ReAct Iters | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| `openai` | 0.7588 | 0.7489 | 0.1842 | 0.84 | 1.59 | 0.925 |
| `lmstudio` | 0.463 | 0.3699 | 0.0 | 0.04 | 1.04 | 0.95 |

## B.2 Trigger Contribution Ablation (Pivot B 동기 강화)

| Backend | Strategy | Safety Recall | 95% CI | Over-Esc Rate | TP/FN/FP/TN | OK (≥0.95)? |
| --- | --- | --- | --- | --- | --- | --- |
| `openai` | no_escalation | 0.0 | [0.000,0.017] | 0.0 | 0/219/0/38 | ✗ |
| `openai` | threshold_only | 0.5434 | [0.477,0.608] | 0.0263 | 119/100/1/37 | ✗ |
| `openai` | full_uasef | 0.5479 | [0.482,0.612] | 0.0 | 120/99/0/38 | ✗ |
| `lmstudio` | no_escalation | 0.0 | [0.000,0.017] | 0.0 | 0/219/0/38 | ✗ |
| `lmstudio` | threshold_only | 0.5114 | [0.446,0.577] | 0.0 | 112/107/0/38 | ✗ |
| `lmstudio` | full_uasef | 0.4932 | [0.428,0.559] | 0.0 | 108/111/0/38 | ✗ |

**해석.** `threshold_only` (T1만, 순수 CP) → `full_uasef` (T1 ∨ T2 ∨ T3)
의 Safety Recall 격차가 keyword/no-evidence trigger의 한계 기여이며,
`full_uasef`의 over-escalation 증가가 main paper §6.2 Table 2의 FWER 위반
으로 이어진다 — 이것이 Pivot B (조화평균 결합)의 동기다.

## B.3 MedAbstain 변형별 분석

### B.3.1 Per-Variant Metrics

| Backend | Variant | n | Recall | Precision | F1 | AUROC | OK (≥0.95)? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `openai` | AP | 50 | 0.18 | 1.0 | 0.3051 | None | ✗ |
| `openai` | NAP | 50 | 0.14 | 1.0 | 0.2456 | None | ✗ |
| `openai` | A | 50 | 0.14 | 1.0 | 0.2456 | None | ✗ |
| `openai` | NA | 50 | None | 0.0 | None | None | ✗ |
| `lmstudio` | AP | 50 | 0.12 | 1.0 | 0.2143 | None | ✗ |
| `lmstudio` | NAP | 49 | 0.102 | 1.0 | 0.1851 | None | ✗ |
| `lmstudio` | A | 50 | 0.04 | 1.0 | 0.0769 | None | ✗ |
| `lmstudio` | NA | 50 | None | 0.0 | None | None | ✗ |

### B.3.2 Abstention Accuracy (LLM 자체 abstention 능력)

| Backend | TA | FA | TR | MA | Abstention P | Abstention R | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `openai` | 0 | 0 | 50 | 150 | 0.0 | 0.0 | 0.0 |
| `lmstudio` | 0 | 0 | 50 | 149 | 0.0 | 0.0 | 0.0 |

**논의.** Abstention Recall (LLM이 스스로 불확실성을 표현하는 능력)이
낮을수록 UASEF의 CP 기반 결정이 더 큰 가치를 가진다 — 모델이 과신할 때
외부 안전 게이트가 가장 필요하다.

## B.4 Pareto Frontier — Specialty별 권고 α

| Backend | Specialty | Recommended α | Coverage | Esc Rate | Utility |
| --- | --- | --- | --- | --- | --- |
| `openai` | emergency_medicine | 0.01 | 1.0 | 0.5 | 0.0 |
| `openai` | internal_medicine | 0.01 | 1.0 | 0.0 | 1.0 |
| `openai` | general_practice | 0.01 | 1.0 | 0.0 | 1.0 |
| `lmstudio` | emergency_medicine | 0.01 | 1.0 | 0.24 | 0.52 |
| `lmstudio` | internal_medicine | 0.01 | 1.0 | 0.0 | 1.0 |
| `lmstudio` | general_practice | 0.01 | 1.0 | 0.0 | 1.0 |

**main paper Pivot A와의 연결.** Pareto sweep은 단일 전역 α를 specialty
조건부로 측정. main paper Pivot A는 단일 CRC 절차 안에서 stratum별 $\alpha_s$를
부여하여 한 단계 더 진행. 여기서의 권고 α는 기관 배포 시 
$\alpha_{\text{CRITICAL}}, \ldots, \alpha_{\text{LOW}}$ 선택 정보가 된다.

## B.5 Cross-Backend MedAbstain 종합

| Backend | Recall | Precision | F1 | AUROC | Safety Recall ≥ 0.95? |
| --- | --- | --- | --- | --- | --- |
| `openai` | 0.1533 | 0.7667 | 0.2555 | None | ✗ |
| `lmstudio` | 0.0872 | 0.7222 | 0.1556 | None | ✗ |

---

_생성: `run_full_evaluation.sh` (2026-05-08T03:05:59)_
_Source: `results/run_20260507-182038/<backend>/all_experiments_summary.json`_
