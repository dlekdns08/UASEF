r"""
Tabular (classical-ML) baselines — REVISION_PLAN P0-6.

리뷰 #5 의 핵심 방어: "입력이 구조화 피처라면 XGBoost/LogReg 로도 되는 것 아닌가?"
이 모듈은 **decision-time feature 만** 으로 학습한 tabular classifier 를 동일한
StratifiedConformalRiskControl 에 연결해, LLM nonconformity score + CRC 와 직접
비교할 수 있게 한다. (LLM 호출 없음.)

설계
----
- score(case) = P(escalate | decision-time features)  (높을수록 escalate)
  → CRC 의 missed_escalation_loss = 1{label ∧ score ≤ λ} 에 그대로 투입.
    즉 "tabular score + CRC" 가 "LLM score + CRC" 와 drop-in 비교 가능.
- feature 는 case.question (leakage-safe 템플릿) 에서 파싱 → 미래정보 0.

baselines 제공:
  TabularCRCBaseline("logreg")  — Logistic Regression + CRC
  TabularCRCBaseline("gbdt")    — XGBoost(가능 시) / sklearn GradientBoosting + CRC
  AdmissionTypeHeuristic        — emergency/urgent ⇒ escalate (clinical 규칙)
  HighRiskAllEscalate           — stratum∈{CRITICAL,HIGH} ⇒ escalate (trivial safety)
"""
from __future__ import annotations

import re
from typing import Optional


# ── decision-time featurizer ──────────────────────────────────────────────────

AGE_BUCKETS = ["<18", "18-34", "35-49", "50-64", "65-79", "80+", "unknown"]
ADMIT_TYPES = ["EMERGENCY", "URGENT", "DIRECT EMER.", "EW EMER.", "ELECTIVE",
               "OBSERVATION ADMIT", "SURGICAL SAME DAY ADMISSION", "unknown"]
EARLY_LAB_VOCAB = ["lactate_high", "creatinine_high", "troponin_high", "ldh_high",
                   "leukocytosis", "thrombocytopenia", "hyperkalemia", "hyponatremia",
                   "acidemia", "low_bicarb"]
EMERGENCY_TYPES = {"EMERGENCY", "URGENT", "DIRECT EMER.", "EW EMER."}

_RE = {
    "age": re.compile(r"Age bracket:\s*(.+)"),
    "admit": re.compile(r"Admission type:\s*(.+)"),
    "service": re.compile(r"Service(?: at admission)?:\s*(.+)"),
    "labs": re.compile(r"Early lab abnormalities[^:]*:\s*(.+)"),
}


def parse_features(case) -> dict:
    """case.question(표준 decision-time 템플릿) → 원시 feature dict."""
    q = case.question or ""
    out = {"age_bucket": "unknown", "admission_type": "unknown",
           "service": "unknown", "early_labs": []}
    m = _RE["age"].search(q)
    if m: out["age_bucket"] = m.group(1).strip()
    m = _RE["admit"].search(q)
    if m: out["admission_type"] = m.group(1).strip().upper()
    m = _RE["service"].search(q)
    if m: out["service"] = m.group(1).strip()
    m = _RE["labs"].search(q)
    if m:
        raw = m.group(1).strip()
        if raw and raw.lower() != "none":
            out["early_labs"] = [t.strip() for t in raw.split(",") if t.strip()]
    return out


class DecisionTimeFeaturizer:
    """raw feature dict → fixed-length numeric vector (one/multi-hot)."""

    def __init__(self):
        self.services: list[str] = []   # learned at fit

    def fit(self, feats: list[dict]) -> "DecisionTimeFeaturizer":
        svc = sorted({f["service"] for f in feats})
        self.services = svc
        return self

    def transform_one(self, f: dict) -> list[float]:
        v: list[float] = []
        v += [1.0 if f["age_bucket"] == a else 0.0 for a in AGE_BUCKETS]
        v += [1.0 if f["admission_type"] == a else 0.0 for a in ADMIT_TYPES]
        v += [1.0 if f["service"] == s else 0.0 for s in self.services]
        labset = set(f["early_labs"])
        v += [1.0 if lab in labset else 0.0 for lab in EARLY_LAB_VOCAB]
        v += [float(len(labset))]  # acuity count
        return v

    def transform(self, feats: list[dict]) -> list[list[float]]:
        return [self.transform_one(f) for f in feats]


# ── tabular classifier + CRC ──────────────────────────────────────────────────

def _make_model(kind: str):
    """sklearn/xgboost 지연 import. 미설치 시 명시적 오류."""
    if kind == "logreg":
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as e:
            raise SystemExit("sklearn 필요: .venv/bin/pip install scikit-learn") from e
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if kind == "gbdt":
        try:
            from xgboost import XGBClassifier  # type: ignore
            return XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, eval_metric="logloss", n_jobs=2,
            )
        except ImportError:
            try:
                from sklearn.ensemble import GradientBoostingClassifier
            except ImportError as e:
                raise SystemExit(
                    "xgboost 또는 sklearn 필요: .venv/bin/pip install xgboost scikit-learn"
                ) from e
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05)
    raise ValueError(f"unknown kind: {kind!r}")


class TabularCRCBaseline:
    """
    decision-time feature → P(escalate) 를 nonconformity-style score 로 노출.

    fit(cal_cases): featurizer + classifier 학습.
    score(case)   : P(escalate) ∈ [0,1].  CRC 에 그대로 투입.
    """
    def __init__(self, kind: str = "logreg"):
        self.kind = kind
        self.name = f"Tabular-{kind}+CRC"
        self.feat = DecisionTimeFeaturizer()
        self.model = None
        self._backend = None

    def fit(self, cal_cases: list) -> "TabularCRCBaseline":
        feats = [parse_features(c) for c in cal_cases]
        y = [1 if c.expected_escalate else 0 for c in cal_cases]
        self.feat.fit(feats)
        X = self.feat.transform(feats)
        if len(set(y)) < 2:
            # 단일 클래스 → 상수 예측기 (P = prior).
            self._backend = "constant"
            self._prior = (sum(y) / len(y)) if y else 0.0
            return self
        self.model = _make_model(self.kind)
        self.model.fit(X, y)
        self._backend = type(self.model).__name__
        return self

    def score(self, case) -> float:
        f = parse_features(case)
        x = self.feat.transform_one(f)
        if self._backend == "constant":
            return float(self._prior)
        proba = self.model.predict_proba([x])[0]
        # P(class==1) — escalate 확률
        classes = list(getattr(self.model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else len(proba) - 1
        return float(proba[idx])

    def scores(self, cases: list) -> list[float]:
        return [self.score(c) for c in cases]

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "backend": self._backend,
                "n_features": len(self.feat.services) + len(AGE_BUCKETS)
                + len(ADMIT_TYPES) + len(EARLY_LAB_VOCAB) + 1}


# ── trivial / heuristic baselines (CRC 불필요, 직접 escalate 결정) ──────────────

class AdmissionTypeHeuristic:
    """emergency/urgent admission ⇒ escalate (단순 임상 규칙)."""
    name = "AdmissionType-heuristic"

    def predict(self, case) -> bool:
        return parse_features(case)["admission_type"] in EMERGENCY_TYPES

    def info(self) -> dict:
        return {"name": self.name, "rule": "admission_type ∈ emergency/urgent"}


class HighRiskAllEscalate:
    """risk_group(stratum) ∈ {CRITICAL, HIGH} ⇒ escalate (trivial safety upper)."""
    name = "HighRisk-all-escalate"

    def predict(self, case) -> bool:
        return (case.scenario_type or "").upper() in ("CRITICAL", "HIGH")

    def info(self) -> dict:
        return {"name": self.name, "rule": "stratum ∈ {CRITICAL, HIGH}"}
