from typing import Optional, Dict, Any, List
from contextlib import contextmanager

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

from ..client import BlazeMetricsClient


@contextmanager
def BlazeMLflowRun(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Context manager that opens an MLflow run if MLflow is available.
    """
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def blazemetrics_autolog(
    client: Optional[BlazeMetricsClient] = None,
    include: Optional[List[str]] = None,
):
    """
    Register lightweight autolog hooks for computing and logging BlazeMetrics.
    Users call this once at startup.
    """
    if mlflow is None:
        return

    _client = client or BlazeMetricsClient()

    def _log_metrics(candidates: List[str], references: List[List[str]], step: Optional[int] = None, prefix: str = "blaze"):
        try:
            scores = _client.compute_metrics(candidates, references, include=include)
            agg = _client.aggregate_metrics(scores)
            if mlflow is not None:
                mlflow.log_metrics({f"{prefix}_{k}": float(v) for k, v in agg.items()}, step=step)
        except Exception:
            pass

    # exposed function attribute so users can invoke in their loops
    blazemetrics_autolog.log_metrics = _log_metrics  # type: ignore[attr-defined]


