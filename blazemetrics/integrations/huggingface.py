from typing import Any, Dict, Optional, Callable, List

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
except Exception:  # pragma: no cover
    TrainerCallback = object  # type: ignore
    TrainerControl = Any  # type: ignore
    TrainerState = Any  # type: ignore

from ..client import BlazeMetricsClient


class BlazeHFTrainerCallback(TrainerCallback):
    """
    Native HF integration via TrainerCallback.
    - Computes text metrics using Rust core (via EnhancedBlazeMetricsClient)
    - Emits metrics to HF logs and returns to Trainer
    - Minimal overhead; zero-copy lists where possible
    """

    def __init__(
        self,
        client: Optional[BlazeMetricsClient] = None,
        metric_names: Optional[List[str]] = None,
        compute_preds_and_refs: Optional[Callable[[Dict[str, Any]], tuple[List[str], List[List[str]]]]] = None,
    ) -> None:
        self.client = client or BlazeMetricsClient()
        self.metric_names = metric_names
        self.compute_preds_and_refs = compute_preds_and_refs

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):  # type: ignore
        try:
            if self.compute_preds_and_refs is not None:
                preds, refs = self.compute_preds_and_refs(kwargs)
            else:
                # If user provided eval predictions in kwargs as (preds, labels)
                eval_preds = kwargs.get("eval_preds")
                if eval_preds is None:
                    return control
                preds_raw, labels = eval_preds
                preds = [str(p) for p in preds_raw]
                # labels can be list[str] or list[list[str]]; normalize
                if labels and isinstance(labels[0], list):
                    refs = [[str(r) for r in rr] for rr in labels]
                else:
                    refs = [[str(l)] for l in labels]

            result = self.client.compute_metrics(preds, refs, include=self.metric_names)
            agg = self.client.aggregate_metrics(result)
            for k, v in agg.items():
                metrics[f"blaze_{k}"] = float(v)
            return control
        except Exception:
            return control


def with_blazemetrics_pipeline(task: str, pipeline_factory: Callable[..., Any], client: Optional[BlazeMetricsClient] = None, **pipeline_kwargs):
    """
    Wrap a HF pipeline to compute and return BlazeMetrics alongside outputs.
    Usage:
        from transformers import pipeline
        pipe = with_blazemetrics_pipeline("summarization", pipeline, model="...")
        out = pipe(inputs=[...], references=[[...], ...])
        -> returns dict with outputs and metrics
    """
    client = client or BlazeMetricsClient()

    base = pipeline_factory(task, **pipeline_kwargs)

    def _wrapped(*args, references: Optional[List[List[str]]] = None, compute_metrics: bool = True, include: Optional[List[str]] = None, **kwargs):
        outputs = base(*args, **kwargs)
        if not compute_metrics or references is None:
            return {"outputs": outputs}
        # Normalize to list[str]
        if isinstance(outputs, list):
            if outputs and isinstance(outputs[0], dict) and "summary_text" in outputs[0]:
                preds = [o["summary_text"] for o in outputs]
            elif outputs and isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                preds = [o["generated_text"] for o in outputs]
            else:
                preds = [str(o) for o in outputs]
        else:
            preds = [str(outputs)]
        metrics = client.compute_metrics(preds, references, include=include)
        agg = client.aggregate_metrics(metrics)
        return {"outputs": outputs, "blaze_metrics": metrics, "blaze_aggregate": agg}

    return _wrapped


