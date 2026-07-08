"""Document type classification with ML model fallback and rule-based logic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple


DOCUMENT_LABELS = ("invoice", "contract", "act", "waybill", "statement", "unknown")

KEYWORDS: Dict[str, tuple[str, ...]] = {
    "invoice": ("счет", "счёт", "invoice", "счет-фактура", "оплата", "поставщик", "покупатель"),
    "contract": ("договор", "contract", "стороны", "предмет договора", "обязательства", "исполнитель", "заказчик"),
    "act": ("акт", "act", "сдачи-приемки", "сдачи-приёмки", "выполненных работ", "оказанных услуг", "оказано услуг", "работ не имеет", "сдали", "приняли"),
    "waybill": ("накладная", "товарная накладная", "waybill", "торг-12", "грузоотправитель", "грузополучатель"),
    "statement": ("выписка", "statement", "банковская выписка", "движение средств", "остаток", "расчетный счет"),
}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def classify_by_rules(text: str) -> dict:
    """Classify document text using keyword matches."""

    normalized = _normalize(text)
    if not normalized:
        return {"label": "unknown", "confidence": 0.0, "method": "rule_based"}
    if re.search(r"\bакт\s+(?:сдачи|приемки|приёмки|выполненных|оказанных)", normalized):
        return {"label": "act", "confidence": 0.9, "method": "rule_based"}

    scores: Dict[str, int] = {}
    for label, keywords in KEYWORDS.items():
        scores[label] = sum(1 for keyword in keywords if keyword in normalized)

    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return {"label": "unknown", "confidence": 0.0, "method": "rule_based"}

    confidence = min(0.95, 0.35 + best_score * 0.15)
    return {"label": best_label, "confidence": round(confidence, 3), "method": "rule_based"}


def _classify_with_model(text: str, model_path: Path) -> dict | None:
    try:
        import joblib
    except Exception:
        return None
    if not model_path.exists():
        return None

    try:
        model = joblib.load(model_path)
        label = str(model.predict([text])[0])
        confidence = 0.75
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            confidence = float(max(probabilities))
        return {"label": label, "confidence": round(confidence, 3), "method": "ml_model"}
    except Exception:
        return None


def classify_document(text: str, model_path: str | Path | None = None) -> dict:
    """Classify text as invoice, contract, act, waybill, statement, or unknown."""

    path = Path(model_path) if model_path else Path("models") / "document_classifier.pkl"
    ml_result = _classify_with_model(text, path)
    rule_result = classify_by_rules(text)
    if ml_result:
        if ml_result.get("confidence", 0.0) < 0.45 and rule_result.get("label") != "unknown":
            rule_result["method"] = "rule_based_low_confidence_ml_fallback"
            return rule_result
        return ml_result
    return rule_result


def load_training_samples(sample_dir: str | Path) -> Tuple[list[str], list[str]]:
    """Load training texts from data/sample_texts/<label> directories."""

    root = Path(sample_dir)
    texts: list[str] = []
    labels: list[str] = []
    for label in DOCUMENT_LABELS:
        if label == "unknown":
            continue
        label_dir = root / label
        if not label_dir.exists():
            continue
        for path in label_dir.glob("*.txt"):
            texts.append(path.read_text(encoding="utf-8"))
            labels.append(label)
    return texts, labels


def train_baseline_model(sample_dir: str | Path, model_path: str | Path) -> dict:
    """Train and save a TF-IDF + LogisticRegression baseline classifier."""

    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except Exception as exc:
        return {"saved": False, "warning": f"scikit-learn/joblib dependencies are unavailable: {exc}"}

    texts, labels = load_training_samples(sample_dir)
    if len(set(labels)) < 2:
        return {"saved": False, "warning": "At least two document classes are required for training"}

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(texts, labels)
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return {"saved": True, "model_path": str(output_path), "classes": sorted(set(labels))}
