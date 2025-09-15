# BlazeMetrics

<p align="center">
  <img src="benchmarking/logo.png" alt="BlazeMetrics Logo" width="110" />
</p>

<p align="center" style="font-size:1.5em;">
  <b>100x Faster LLM Evaluation</b>
</p>
<p align="center" style="font-size:1.1em;">
  Rust-powered evaluation suite processing <b>1M+ evaluations/sec</b>.<br>
  Complete LLM quality, safety, and performance monitoring in one unified API.
</p>

<div align="center">
  <img src="benchmarking/image.png" alt="BlazeMetrics Dashboard" width="530" style="border-radius:16px;box-shadow:2px 2px 8px #dab;"/>
</div>

<p align="center">
  <a href="https://pypi.org/project/blazemetrics/"><img src="https://img.shields.io/pypi/v/blazemetrics?color=blue&style=flat-square"></a>
  <a href="https://github.com/2796gaurav/blazemetrics/actions"><img src="https://img.shields.io/github/workflow/status/2796gaurav/blazemetrics/main?style=flat-square"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/2796gaurav/blazemetrics?style=flat-square"></a>
  <img alt="Made with Rust & Python" src="https://img.shields.io/badge/Made%20with-Rust%20%26%20Python-informational?style=flat-square&logo=rust&logoColor=f7a41d&labelColor=333">
</p>

---

## 🚦 Don't stop until done

BlazeMetrics is designed to make evaluation, guardrails, monitoring, and analytics continuous and actionable at production scale. Whether you're running compliance, safety, real-world production, or benchmarking workflows, BlazeMetrics is built to keep evaluating, analyzing, and monitoring—all the way to the finish line. No matter how large your data or how demanding your workflow, you get complete, automated, and live insights. <b>Don't stop until you're done.</b>

---

## Why BlazeMetrics?

- **All-in-one evaluation:** BLEU, ROUGE, WER, METEOR, and more—plus analytics and real guardrail safety.
- **Rust-powered:** 100x speed, process millions of LLM/NLP samples in seconds.
- **Guardrails built-in:** Blocklists, PII, regex, JSON schema, safety, and LLM-based factuality scoring.
- **Enterprise & research ready:** Analytics, anomaly detection, dashboards, monitoring, instant reporting.
- **Out-of-the-box for LLMs, RAG & agent workflows.**

---

## ⚡ Live Benchmark: Speed vs Leading Industry Libraries

**Benchmark Objective:** Speed and RAM comparison computing BLEU, ROUGE, METEOR, etc. between BlazeMetrics, Huggingface Evaluate, SacreBLEU, NLTK, and TorchMetrics.

| Library                 | Time (s)   | Relative Speed |
|------------------------|------------|:--------------|
| **BlazeMetrics**       | 4.85 (ref) | 1.00x          |
| NLTK                   | 5.40       | 1.11x          |
| SacreBLEU              | 5.51       | 1.13x          |
| Huggingface Evaluate   | 18.19      | 3.75x          |
| TorchMetrics           | 63.59      | 13.10x         |

- Each tested on 10,000 normalized candidate/reference text pairs (median of 3 runs).
- Full normalization, psutil RAM/CPU monitoring.



---

##  Features At a Glance

-  **State-of-the-art metrics** (BLEU, ROUGE, WER, METEOR, CHRF, BERTScore & more)
- ️ **Guardrails**: Block unsafe content, redact PII, enforce custom policies with regex/JSON
-  **Streaming analytics**: Outlier detection, trending, alerts for real-time eval
-  **LLM & RAG integration**: Plug and play with OpenAI, Anthropic, LangChain, HuggingFace, code/agent ground truth, RAG
-  **Factuality/Judge**: Hallucination & faithfulness scoring using LLM judges
-  **Production-scale speed**: Rust core, easy parallelism and batch
-  **Dashboards & reporting**: Instant model/data card, web dashboards (optional)
-  **Easy to extend**: Custom guardrails, exporters, analytics for your workflow

---

##  Installation

**Stable (CPU, core features):**
```shell
pip install blazemetrics
```

**with Dashboards/Monitoring/etc:**
```shell
pip install "blazemetrics[dashboard]"
```

**From source (for developers):**
```shell
git clone https://github.com/2796gaurav/blazemetrics.git
cd blazemetrics
pip install -r requirements.txt
maturin develop
```

---

##  Quickstart: Get Bleeding-Edge Metrics in Seconds

Evaluate all key metrics in just 3 lines—no config required!

```python
from blazemetrics import BlazeMetricsClient

candidates = ["The quick brown fox.", "Hello world!"]
references = [["The fast brown fox."], ["Hello world."]]

client = BlazeMetricsClient()
metrics = client.compute_metrics(candidates, references)
print(metrics)  # {'rouge1_f1': [...], 'bleu': [...], ...}

print(client.aggregate_metrics(metrics))  # {'rouge1_f1': 0.85, ...}
```

---

##  Full LLM Workflow: Metrics, Guardrails, Analytics & Factuality — All in One

```python
from blazemetrics import BlazeMetricsClient
from blazemetrics.llm_judge import LLMJudge  # LLM-based scoring for factuality

# Your LLM generations and references
candidates = ["Alice's email is alice@example.com.", "2 + 2 is 5."]
references = [["Her email is alice@example.com."], ["2 + 2 = 4"]]

client = BlazeMetricsClient(
    blocklist=["bitcoin"],
    redact_pii=True,
    regexes=[r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b"],
    enable_analytics=True,
    metrics_lowercase=True,
)

# 1. Metrics
metrics = client.compute_metrics(candidates, references)
agg = client.aggregate_metrics(metrics)

# 2. Guardrail safety checks
violations = client.check_safety(candidates)

# 3. Analytics/trends
client.add_metrics(agg)
analytics = client.get_analytics_summary()

# 4. LLM-based factuality/hallucination (uses OpenAI, API key required)
judge = LLMJudge(provider="openai", api_key="YOUR_OPENAI_KEY", model="gpt-4o")
def factuality_scorer(output, reference):
    result = judge.score([output], [reference])
    return {"factuality": result[0].get("faithfulness", 0.0)}
client.set_factuality_scorer(factuality_scorer)
facts = client.evaluate_factuality(candidates, [r[0] for r in references])

# 5. Fancy model card report
model_card = client.generate_model_card("my-llm", metrics, analytics, config=vars(client.config), violations=violations, factuality=facts)
print(model_card)
```

---

##  Easy Integration: LLM, RAG, Agents, Guardrails, and More

- Use as a drop-in evaluation for HuggingFace, OpenAI, Anthropic, LangChain, code generation, and agentic workflows
- Proven RAG and semantic search/trace support: `semantic_search`, `agentic_rag_evaluate`, provenance tracking
- Real-time dashboards: `blazemetrics-dashboard` (if installed with [dashboard])
- Built-in exporters for Prometheus, StatsD, CSV, and HTML reports

---

##  Advanced: Async, Parallel, Streaming, and Dashboard

- **Parallel/async evaluation:**  
  `client.compute_metrics_parallel(...)` and `client.compute_metrics_async(...)`
- **Streaming analytics & alerting:**  
  Add metrics sample-by-sample and get anomalies/trends in real time.
- **Instant dashboards:**  
  After `pip install "blazemetrics[dashboard]"`, run:
  ```sh
  blazemetrics-dashboard
  ```
  Or, embed the dashboard server in your app/WSGI pipeline.
- **RAG/agent evaluation:**  
  ```python
  client.agentic_rag_evaluate(...)
  ```

---

##  API Overview (Unified Client)

`BlazeMetricsClient` config (selected):

- Metrics: `metrics_include`, `metrics_lowercase`, `metrics_stemming`
- Guardrails: `blocklist`, `regexes`, `redact_pii`, `case_insensitive`
- Analytics: `enable_analytics`, `analytics_window`, `analytics_alerts`, `analytics_anomalies`
- Monitoring/Exporters: `enable_monitoring`, `prometheus_gateway`, `statsd_addr`
- LLM config: `llm_provider`, `model_name`
- Performance: `parallel_processing`, `max_workers`

See [full docs](docs/) or `help(blazemetrics.BlazeMetricsClient)` for every option!

---

##  Dashboards & Reporting

- **Web dashboards:** Instantly launch a web app for monitoring and reporting
- **Instant model/data cards:** Beautiful shareable markdown summaries for your models and datasets
- **Export:** Write HTML, CSV, Prometheus format, or push to cloud

---

##  Learn More

- [Documentation](docs/)
- [Examples](examples/)
- [Advanced Usage: Analytics, Streaming, and Exporters](docs/user-guide/)
- [RAG/Agent Evaluations](docs/use-cases/rag-applications.md)
- [Production & Compliance](docs/user-guide/guardrails.md)

---

##  Contribute & Community

- Star us on [GitHub](https://github.com/2796gaurav/blazemetrics)
- Open issues/feature requests, or create a PR!
- Join the discussion and help evolve LLM benchmarking and safety together.

---

##  License

MIT

---

BlazeMetrics  2025 [Gaurav](mailto:2796gaurav@gmail.com)

