from typing import Any, Dict, Optional

try:
    from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
except Exception:  # pragma: no cover
    CallbackManager = None  # type: ignore
    CBEventType = None  # type: ignore
    EventPayload = None  # type: ignore

from ..client import BlazeMetricsClient


class BlazeLlamaIndexHandler:
    """
    Minimal LlamaIndex instrumentation to observe LLM/Node outputs.
    Users can add this handler into CallbackManager.
    """

    def __init__(self, client: Optional[BlazeMetricsClient] = None):
        self.client = client or BlazeMetricsClient()

    def bind(self, manager: "CallbackManager") -> None:  # type: ignore
        if manager is None:
            return
        manager.add_trace_map({
            CBEventType.LLM: self._on_llm_event,
            CBEventType.NODE_PARSING: self._on_node_event,
        })

    def _on_llm_event(self, event_type, payload):  # type: ignore
        try:
            text = None
            if isinstance(payload, dict):
                text = payload.get("response") or payload.get("output")
            if not text:
                return
            preds = [str(text)]
            refs = [[""]]
            _ = self.client.compute_metrics(preds, refs)
        except Exception:
            return

    def _on_node_event(self, event_type, payload):  # type: ignore
        return


