"""Metrics for evaluating labeled claim-verification runs.

This module does not create ground truth or run the verification pipeline.
It scores predictions against caller-supplied labels so benchmark results
remain auditable and are never confused with confidence scores.
"""

LABELS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")


def classification_metrics(expected, predicted):
    """Calculate classification metrics with ``None`` as an abstention.

    Accuracy, precision, recall, F1, and the confusion matrix are computed on
    covered examples. ``coverage`` reports the proportion of all labeled
    examples that received one of the three valid classifications.
    """
    if not isinstance(expected, (list, tuple)) or not isinstance(predicted, (list, tuple)):
        raise ValueError("Expected and predicted labels must be lists or tuples.")
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have equal lengths.")
    if not expected:
        raise ValueError("At least one labeled example is required.")

    normalized_expected = [str(label).upper() if label is not None else "" for label in expected]
    normalized_predicted = [str(label).upper() if label is not None else None for label in predicted]

    invalid_expected = sorted(set(normalized_expected) - set(LABELS))
    if invalid_expected:
        raise ValueError(f"Invalid expected labels: {', '.join(invalid_expected)}")

    invalid_predicted = sorted(
        {label for label in normalized_predicted if label is not None and label not in LABELS}
    )
    if invalid_predicted:
        raise ValueError(f"Invalid predicted labels: {', '.join(invalid_predicted)}")

    covered = [
        (truth, guess)
        for truth, guess in zip(normalized_expected, normalized_predicted)
        if guess is not None
    ]
    if not covered:
        raise ValueError("No predictions were available to evaluate.")

    confusion = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    for truth, guess in covered:
        confusion[truth][guess] += 1

    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in LABELS if actual != label)
        fn = sum(confusion[label][guess] for guess in LABELS if guess != label)
        tn = len(covered) - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "false_negative_rate": round(fn / (fn + tp), 4) if fn + tp else 0.0,
        }

    correct = sum(confusion[label][label] for label in LABELS)
    total = len(covered)
    return {
        "labels": list(LABELS),
        "total_examples": len(expected),
        "covered_examples": total,
        "coverage": round(total / len(expected), 4),
        "accuracy_on_covered": round(correct / total, 4),
        "macro_f1": round(sum(item["f1"] for item in per_class.values()) / len(LABELS), 4),
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def evaluate_records(records):
    """Evaluate records containing ``expected`` and ``predicted`` labels."""
    if not isinstance(records, list):
        raise ValueError("Records must be a list.")
    return classification_metrics(
        [record.get("expected") if isinstance(record, dict) else None for record in records],
        [record.get("predicted") if isinstance(record, dict) else None for record in records],
    )
