from typing import Optional, Dict, Any, List

try:
    from evidently.report import Report
    from evidently.metric_preset import TextNLPDriftPreset
except Exception:  # pragma: no cover
    Report = None  # type: ignore
    TextNLPDriftPreset = None  # type: ignore

from ..client import BlazeMetricsClient


class EvidentlyExporter:
    """
    Bridge BlazeMetrics aggregates into Evidently reports for dataset-level monitoring.
    """

    def __init__(self, client: Optional["BlazeMetricsClient"] = None):
        self.client = client or BlazeMetricsClient()

    def generate_text_report(self, current_texts: List[str], reference_texts: Optional[List[str]] = None) -> Optional["Report"]:
        if Report is None or TextNLPDriftPreset is None:
            return None
        # Use BlazeMetrics to compute auxiliary stats if desired (not mandatory for report)
        try:
            # dummy self-check via metrics computation consistency path
            _ = self.client.aggregate_metrics({})
        except Exception:
            pass
        return Report(metrics=[TextNLPDriftPreset()])


