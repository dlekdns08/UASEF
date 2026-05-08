# Round 8 — Novelty Audit & Reinforcement Plan

> **목표**: Round 7 페이퍼의 노벨티/재현성/통계적 엄격성 갭을 메우고
> ML4H 2026 Spotlight / AISTATS 2026 / NeurIPS 2026 main track 경쟁력 확보.
>
> **진행 상태**: Phase 1 착수 (사용자 승인 2026-05-08).
>
> **선택된 옵션**:
> - 착수 범위: Phase 1 즉시 착수 (코드 수정 + 5-seed 실행)
> - L1 완화 전략: Self-consistency LLM-judge (GPT-4o + Claude Opus 4.7)
> - 헤드라인 클레임: 현 표현 유지 (38× 강조, §6.3.5 caveat 보존)

---

## 0. 노벨티 진단 요약

| Pivot | 페이퍼 자기-라벨링 | 실제 등급 | 비고 |
|---|---|---|---|
| A. Stratified CRC | "statistical, composition" | 기계적 합성 | Angelopoulos & Bates 2024 + Romano 2020 직접 결합. 임상 도메인 적용만 신규. |
| B. Multi-Trigger Conformal Combination | "statistical application" | 응용/차용 | Wilson 2019 + Vovk-Wang 2019 직접 적용. T2/T3 marginal 기여 ≈0 (§7.5 자기 인정). |
| C. Cost-Aware Calibration | "statistical + engineering" | 표준 최적화 | constrained threshold sweep. 알고리즘 신규성 약함. |
| D. Reproducibility | "engineering" | 정상 (claim 일부 과장) | 137-test → 실제 133, MedAbstain JSONL 미포함 등 |

**결론**: 수학적 신규성은 약하나, **임상 안전 LLM 도메인의 합성·통합·정직한 평가**가 핵심 기여. Top-venue 통과를 위해서는 P0/P1 보강 필수.

**코드 품질**: stratified_crc / conformal_combination / cost_aware_calibration / rtc_ede 4개 모듈 모두 알고리즘적으로 정확. 베이스라인도 충실 — UASEF 우위를 부풀리는 shortcut 없음.

---

## 1. 갭 인벤토리

### P0 (출판 차단)

| ID | 갭 | 현재 | 영향 |
|---|---|---|---|
| P0-1 | 단일 seed (42) — Table 1·4에 신뢰구간 없음 | 인프라만 ship, 실행 안 됨 | 통계 유의성 입증 불가 |
| P0-2 | `data/raw/` MedAbstain JSONL 누락 | gitignore + fallback 30-item 하드코드 | i.i.d. 위반, 재현성 깨짐 |
| P0-3 | L1 휴리스틱 라벨 미해결 | IRB 프로토콜만 commit | ground truth = 키워드 분류기 |
| P0-4 | n_CRITICAL ≥ 999 미충족 → α=0.001은 합성만 | algorithm-level만 검증 | 헤드라인 "응급 0.1% miss"의 실험적 근거 부재 |

### P1 (강한 리뷰 지적)

| ID | 갭 |
|---|---|
| P1-1 | TECP-stratified, cost-sensitive single-α, v1-cost-aware ablation 결과 미생성 |
| P1-2 | T2/T3 marginal contribution ≈0 → Pivot B 가치 정량 사례 추가 필요 |
| P1-3 | 38× 헤드라인 vs 4-D sweep 중앙값 5–10× — 표현 일관성 (현 옵션: 유지) |
| P1-4 | 단일 데이터셋 (MedAbstain) — multi-dataset dispatcher 결과 부재 |
| P1-5 | "137 test" → 실제 133. 페이퍼 수치 자체에 대한 unit test 없음 |

### P2 (강화)

| ID | 갭 |
|---|---|
| P2-1 | Theorem 1 sketch만 있음 → full proof |
| P2-2 | Cost matrix 정당화 보강 (HCUP 등) |
| P2-3 | Distribution shift 실험 |
| P2-4 | Multi-lingual sanity |
| P2-5 | Ethics/equity 분석 |

---

## 2. 단계별 실행 계획

### Phase 1 — P0 해결 (4주)

**Week 1: Seed 다양화 + 데이터 확보**
- [P0-2] MedAbstain JSONL을 `data/raw/`에 배치 또는 자동 다운로드 스크립트 추가
- [P0-2] `data/loader.py`의 `UASEF_ALLOW_FALLBACK` 가드를 페이퍼 재현 모드에서 강제 OFF
- [P0-1] `run_multiseed_evaluation.sh` 5-seed (42–46) 실제 실행
- [P0-1] `experiments/aggregate_multiseed.py`의 percentile bootstrap 95% CI를 Table 1·4에 통합
- [P0-1] McNemar paired test → seed-aggregated 버전

**Week 2: Ablation 채우기 (P1-1)**
- TECP-stratified, cost-sensitive single-α, v1-cost-aware를 5-seed로 동일 파이프라인 실행
- Table 4 확장: 8 method × {gpt-4o, LLaMA-3.1-8B} + per-pair p-value
- 결과를 `results/round7/table4_baseline.{json,md}` 정식 위치 동기화

**Week 3: L1 완화 — LLM-judge self-consistency**
- 신규: `experiments/llm_judge_relabel.py`
  - n=200 CRITICAL/HIGH 케이스 추출
  - GPT-4o + Claude Opus 4.7 각각 escalate/no-escalate 판정 + 1-sentence justification
  - 두 모델 합의 라벨 → bootstrap ground truth
  - Cohen's κ (model-A vs model-B) 보고
  - disagreement 케이스는 supplementary §C에 별도 표
- Table 1·4 LLM-judge 라벨로 재분석 → supplementary §C
- 페이퍼 §7.1 L1: "validated on heuristic + LLM-judge consensus subset (n=200)" 추가
- **caveat 명시**: LLM-judge ≠ board-cert physician. IRB 라벨링은 여전히 camera-ready 일정 (§7.1)

**Week 4: α=0.001 표기 일관성 (P0-4)**
- Abstract·§1.3·§3.3·§4.1·§8 모두 "validated empirically at α_s ∈ [0.05, 0.20]; α_CRITICAL=0.001 is algorithm-level only" 일관 표기
- §3.3 "Strict Mode" → "Operational Tier" 등 명확한 명칭
- Table 1a (α=0.05–0.20 empirical) / Table 1b (α=0.001 synthetic algorithm validation) 분리

### Phase 2 — P1 보강 (3주)

**Week 5: Multi-dataset (P1-4)**
- 기존 dispatcher 활용해 TruthfulQA / MedQA-USMLE / PubMedQA 1-seed 실행
- v2 vs TECP head-to-head를 supplementary §D
- 헤드라인은 MedAbstain 유지, "generalization evidence" 별도 표

**Week 6: Pivot B 가치 재정립 (P1-2)**
- "기관 트리거 커스터마이즈" case study:
  - 가상 기관 A: 응급 키워드 +5개 → naive OR over-escalation 폭증 vs harmonic 안정
  - synthetic injected label-noise + correlated triggers에서 v2 robustness 정량화
- §6.2 FWER 검증을 m ∈ {3, 5, 8} trigger 확장

**Week 7: 일관성 + regression guard (P1-3, P1-5)**
- 38× 표기는 §6.3 sensitivity sweep 로컬로 격리; 헤드라인 abstract 유지 결정
- 137 → 실제 카운트로 정정
- `tests/test_paper_claims.py` 신규: Table 2 FWER ≤ 0.05, Table 3 reduction ≥ 5×, Table 4 v2 CRITICAL recall ≥ 0.90 regression guard

### Phase 3 — P2 (선택, 2주)

- Theorem 1 full proof (supplementary §A)
- Cost matrix HCUP 인용
- Distribution shift (specialty mismatch)
- 다국어 sanity (zh)
- Ethics 섹션 확장

---

## 3. 코드/문서 수정 체크리스트

### 코드
- [ ] `data/loader.py:35-60` — 페이퍼 재현 모드에서 `UASEF_ALLOW_FALLBACK` 강제 OFF
- [ ] `data/raw/medabstain_*.jsonl` — 데이터 배치 또는 다운로드 스크립트
- [ ] `experiments/round7_table4_baseline.py` — `--n-seeds` 5 default
- [ ] `experiments/aggregate_multiseed.py` — McNemar seed-pooled 버전
- [ ] `experiments/llm_judge_relabel.py` (신규) — GPT-4o + Claude Opus 4.7 consensus
- [ ] `experiments/round7_table1_coverage.py` — Table 1a/1b 분리
- [ ] `tests/test_paper_claims.py` (신규) — 페이퍼 핵심 수치 regression guard
- [ ] `run_full_evaluation.sh` — Tables 1·4를 `results/round7/`에도 복사

### 문서
- [ ] Abstract — 38× 유지 (선택), Table 4 head-to-head 20–21× 함께 명시
- [ ] §1.3 — "137-test" → "133-test"
- [ ] §3.3 — α_CRITICAL=0.001을 "algorithm-validated" 일관 표기
- [ ] §6.3 — Table 1a / Table 1b 분리
- [ ] §7.1 L1 — LLM-judge 결과 추가
- [ ] supplementary §A — Theorem 1 full proof
- [ ] supplementary §C — LLM-judge re-labeling 결과 (Phase 1 W3)
- [ ] supplementary §D — Multi-dataset generalization (Phase 2 W5)
- [ ] `IRB_PROTOCOL.md` — IRB 지연 시 LLM-judge fallback 명시
- [ ] `README.md` — 클레임 정정 ("137" → "133", "MedAbstain bundled" → "download script")

---

## 4. 비용·시간 예산

| Phase | 항목 | OpenAI | Anthropic | 시간 |
|---|---|---|---|---|
| Phase 1 W1 | 5-seed × 2 backend × 4 table | ~$125 | – | ~3 hours |
| Phase 1 W2 | Ablation (3 추가 method, 5-seed) | ~$60 | – | ~1.5 hours |
| Phase 1 W3 | LLM-judge n=200 × 2 model | ~$10 | ~$15 | ~30 min |
| Phase 2 W5 | Multi-dataset 1-seed × 3 dataset | ~$30 | – | ~1 hour |
| Phase 2 W6 | Synthetic FWER m ∈ {3, 5, 8} | – | – | ~10 min |
| **Total Phase 1+2** | | **~$225** | **~$15** | **~6 hours** |

---

## 5. 우선순위 의존성

```
[Week 1 데이터·5-seed]
    ↓
[Week 2 ablation 5-seed] ── [Week 3 LLM-judge] ── [Week 4 doc 정리]
    ↓
[Week 5 multi-dataset] ── [Week 6 Pivot B case study] ── [Week 7 regression guard]
```

W1은 W2의 입력. W3는 W1·W2 독립 — 병렬 가능. W4는 모두 끝나고 페이퍼 update.

---

## 6. 위험 및 완화

| 위험 | 완화 |
|---|---|
| 5-seed 결과가 단일-seed 우위 → 사라짐 | confidence interval 정직 보고. 차이 사라지면 "comparable performance with FWER guarantee" 재포지셔닝. |
| LLM-judge 라벨이 키워드 분류기와 다르지 않음 (circular reasoning) | Cohen's κ 보고. κ < 0.5면 LLM-judge 의존 안 함. |
| MedAbstain 원본 데이터 라이선스 이슈 | LICENSE 확인 후 자동 다운로드 스크립트만 제공, 데이터 자체는 commit 안 함. |
| Phase 1 4주 타임라인 초과 | Phase 2/3는 camera-ready로 연기. Phase 1만이 submission 차단 항목. |
