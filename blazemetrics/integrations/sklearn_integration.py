from typing import List, Optional, Callable

try:
    from sklearn.metrics import make_scorer
except Exception:  # pragma: no cover
    def make_scorer(func, greater_is_better: bool = True):  # type: ignore
        return func

from ..client import BlazeMetricsClient


def _make_text_metric_scorer(metric_name: str, greater_is_better: bool = True) -> Callable:
    client = BlazeMetricsClient()

    def _score(y_true: List[str], y_pred: List[str]) -> float:
        refs = [[r] for r in y_true]
        metrics = client.compute_metrics(y_pred, refs, include=[metric_name])
        # Prefer *_f1 keys for rouge
        if metric_name.startswith("rouge"):
            key = f"{metric_name}_f1"
        else:
            key = metric_name
        vals = metrics.get(key) or metrics.get(metric_name) or []
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    return make_scorer(_score, greater_is_better=greater_is_better)


rouge1_scorer = _make_text_metric_scorer("rouge1", True)
rouge2_scorer = _make_text_metric_scorer("rouge2", True)
rougeL_scorer = _make_text_metric_scorer("rougeL", True)
bleu_scorer = _make_text_metric_scorer("bleu", True)
chrf_scorer = _make_text_metric_scorer("chrf", True)
meteor_scorer = _make_text_metric_scorer("meteor", True)
wer_scorer = _make_text_metric_scorer("wer", False)
token_f1_scorer = _make_text_metric_scorer("token_f1", True)
jaccard_scorer = _make_text_metric_scorer("jaccard", True)


