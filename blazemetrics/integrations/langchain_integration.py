from typing import Any, Dict, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover
    BaseCallbackHandler = object  # type: ignore

from ..client import BlazeMetricsClient


class BlazeLangChainHandler(BaseCallbackHandler):
    """
    LangChain callback handler that computes BlazeMetrics on LLM outputs.
    """

    def __init__(self, client: Optional[EnhancedBlazeMetricsClient] = None, include: Optional[list[str]] = None):
        self.client = client or BlazeMetricsClient()
        self.include = include

    def on_llm_end(self, response, *, run_id: Optional[str] = None, parent_run_id: Optional[str] = None, **kwargs: Dict[str, Any]):  # type: ignore[override]
        try:
            generations = response.generations if hasattr(response, "generations") else []
            preds = []
            for gset in generations:
                for gen in gset:
                    if hasattr(gen, "text"):
                        preds.append(str(gen.text))
            if not preds:
                return
            refs = [[""] for _ in preds]
            _ = self.client.compute_metrics(preds, refs, include=self.include)
        except Exception:
            return


