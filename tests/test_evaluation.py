import pytest

from evaluation import classification_metrics, evaluate_records


def test_classification_metrics_report_per_class_and_confusion():
    result = classification_metrics(
        ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"],
        ["SUPPORTED", "CONTRADICTED", "CONTRADICTED", None],
    )

    assert result["total_examples"] == 4
    assert result["covered_examples"] == 3
    assert result["coverage"] == 0.75
    assert result["accuracy_on_covered"] == pytest.approx(2 / 3, abs=0.0001)
    assert result["confusion_matrix"]["SUPPORTED"]["CONTRADICTED"] == 1
    assert result["per_class"]["SUPPORTED"]["false_negative_rate"] == 0.5


def test_evaluate_records_uses_labeled_rows():
    result = evaluate_records([
        {"expected": "SUPPORTED", "predicted": "supported"},
        {"expected": "CONTRADICTED", "predicted": "CONTRADICTED"},
    ])

    assert result["accuracy_on_covered"] == 1.0
    assert result["coverage"] == 1.0


@pytest.mark.parametrize(
    "expected,predicted",
    [([], []), (["SUPPORTED"], []), (["UNKNOWN"], ["SUPPORTED"]), (["SUPPORTED"], ["UNKNOWN"])],
)
def test_invalid_metric_inputs_are_rejected(expected, predicted):
    with pytest.raises(ValueError):
        classification_metrics(expected, predicted)


def test_abstention_on_every_record_is_rejected():
    with pytest.raises(ValueError, match="No predictions"):
        classification_metrics(["SUPPORTED"], [None])
