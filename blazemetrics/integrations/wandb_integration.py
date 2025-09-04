from typing import Optional, Dict, Any, List

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None  # type: ignore

from ..client import BlazeMetricsClient


class BlazeWandbCallback:
    """
    Lightweight W&B callback/logger that computes BlazeMetrics and logs them.
    Can be used standalone or inside HF Trainer via Callback.
    """

    def __init__(self, project: Optional[str] = None, run_name: Optional[str] = None, client: Optional[BlazeMetricsClient] = None, group: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.project = project
        self.run_name = run_name
        self.group = group
        self.client = client or BlazeMetricsClient()
        self.config = config or {}
        self._run = None

    def start(self):
        if wandb is None:
            return
        if self._run is None:
            self._run = wandb.init(project=self.project, name=self.run_name, group=self.group, config=self.config, reinit=True)

    def finish(self):
        if wandb is None:
            return
        try:
            if self._run is not None:
                self._run.finish()
        finally:
            self._run = None

    def log_text_metrics(self, candidates: List[str], references: List[List[str]], step: Optional[int] = None, include: Optional[List[str]] = None, prefix: str = "blaze"):
        if wandb is None:
            return
        self.start()
        try:
            scores = self.client.compute_metrics(candidates, references, include=include)
            agg = self.client.aggregate_metrics(scores)
            wandb.log({f"{prefix}/{k}": float(v) for k, v in agg.items()}, step=step)
        except Exception:
            pass


