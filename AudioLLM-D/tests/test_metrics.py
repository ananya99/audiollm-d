from audiollm_d.metrics import (
    compute_order_aware_disfluency_token_metrics,
    compute_disfluency_token_metrics,
)


def test_order_aware_disfluency_token_metrics_exact_match() -> None:
    refs = ["I &um think (.) this is fine"]
    preds = ["I &um think (.) this is fine"]
    disfluency_tokens = ["&um", "(.)"]

    metrics = compute_order_aware_disfluency_token_metrics(refs, preds, disfluency_tokens)

    assert metrics["disfluency_token_order_tp"] == 2
    assert metrics["disfluency_token_order_pred_total"] == 2
    assert metrics["disfluency_token_order_ref_total"] == 2
    assert metrics["disfluency_token_order_precision"] == 1.0
    assert metrics["disfluency_token_order_recall"] == 1.0
    assert metrics["disfluency_token_order_f1"] == 1.0
    assert metrics["per_token_order_aware_stats"]["&um"]["f1"] == 1.0
    assert metrics["per_token_order_aware_stats"]["(.)"]["f1"] == 1.0


def test_order_aware_disfluency_token_metrics_penalizes_reordered_tokens() -> None:
    refs = ["I &um think &uh this is fine"]
    preds = ["I &uh think &um this is fine"]
    disfluency_tokens = ["&um", "&uh"]

    order_metrics = compute_order_aware_disfluency_token_metrics(refs, preds, disfluency_tokens)
    count_metrics = compute_disfluency_token_metrics(refs, preds, disfluency_tokens)

    assert count_metrics["disfluency_token_f1"] == 1.0
    assert order_metrics["disfluency_token_order_tp"] == 0
    assert order_metrics["disfluency_token_order_f1"] == 0.0


def test_order_aware_disfluency_token_metrics_uses_alignment_anchors() -> None:
    refs = ["the boy &um is taking cookies"]
    preds = ["the boy is taking &um cookies"]
    disfluency_tokens = ["&um"]

    metrics = compute_order_aware_disfluency_token_metrics(refs, preds, disfluency_tokens)

    assert metrics["disfluency_token_order_tp"] == 0
    assert metrics["disfluency_token_order_pred_total"] == 1
    assert metrics["disfluency_token_order_ref_total"] == 1
    assert metrics["disfluency_token_order_f1"] == 0.0


def test_order_aware_disfluency_token_metrics_tracks_wrong_type_per_token_counts() -> None:
    refs = ["I &um think"]
    preds = ["I &uh think"]
    disfluency_tokens = ["&um", "&uh"]

    metrics = compute_order_aware_disfluency_token_metrics(refs, preds, disfluency_tokens)
    per_token = metrics["per_token_order_aware_stats"]

    assert metrics["disfluency_token_order_tp"] == 0
    assert metrics["disfluency_token_order_pred_total"] == 1
    assert metrics["disfluency_token_order_ref_total"] == 1
    assert metrics["disfluency_token_order_f1"] == 0.0

    # Wrong type attribution: prediction token gets FP side, reference token gets FN side.
    assert per_token["&uh"]["pred_total"] == 1
    assert per_token["&uh"]["ref_total"] == 0
    assert per_token["&uh"]["tp"] == 0
    assert per_token["&um"]["pred_total"] == 0
    assert per_token["&um"]["ref_total"] == 1
    assert per_token["&um"]["tp"] == 0


def test_compute_disfluency_token_metrics_includes_order_aware_fields() -> None:
    refs = ["&um hi"]
    preds = ["&um hi"]
    disfluency_tokens = ["&um"]

    metrics = compute_disfluency_token_metrics(refs, preds, disfluency_tokens)

    assert "disfluency_token_order_precision" in metrics
    assert "disfluency_token_order_recall" in metrics
    assert "disfluency_token_order_f1" in metrics
    assert "per_token_order_aware_stats" in metrics
    assert metrics["disfluency_token_order_f1"] == 1.0
