"""
08_agentic_pipeline_langchain_demo.py

BlazeMetrics Example – Agentic Pipeline Trace Evaluation (Simulated LangChain/CrewAI)
------------------------------------------------------------------------------------
Purpose:
 - Evaluate logs/traces from multi-tool, multi-step agentic chains (e.g., LangChain, CrewAI, any RAG pipeline)
 - Metrics for pipeline coordination, retrieval quality, tool effectiveness, and overall goal completion

Key metrics:
  - retrieval_precision: Did agent select correct fact/evidence?
  - coordination_score: Are agent/tool/planner steps coordinated and ordered logically?
  - tool_selection_accuracy: Are the right tools used for the task?
  - goal_completion_rate: Did the pipeline actually deliver the required answer/output?

How it works:
  Each query (user question/task) gets a corresponding agent_traces (stepwise pipeline trace/log)
  and an expected_outputs (reference answer you want the pipeline to reach).
  This matches the real-world output of LangChain, CrewAI, vLLM/RAG, etc.

Agent trace struct fields:
  - steps: [{tool, input, output}] sequence
  - errors: List of error strings (for traceability)
  - coordination: Did planner + tools + LLM work together? (bool, or could be more detailed)
  - goal_completed: Did the pipeline reach the target output?
  - safety_violations: Any policy/safety problems at any step

You can add custom fields for richer pipelines (memory states, RAG docs, planner logs, etc.)
"""

from blazemetrics import AgenticRAGEvaluator

# Each query/task matches (by index) to agent_traces and expected_outputs
queries = [
    "How much does 10 Tesla shares cost?",
    "What is the capital of France and what is the best restaurant there?"
]

agent_traces = [
    {
        "steps": [
            {"tool": "Search", "input": "Tesla share price", "output": "$700"},
            {"tool": "Math", "input": "$700*10", "output": "$7000"},
            {"tool": "LLM", "input": "Summary: ...", "output": "10 shares cost $7000."}
        ],
        "errors": ["ChainIndexError: Index out of range"],  # captured errors (optional)
        "coordination": True,  # did planner/tools/llm communicate as intended?
        "goal_completed": True,
        "safety_violations": [],
    },
    {
        "steps": [
            {"tool": "Search", "input": "Capital of France", "output": "Paris"},
            {"tool": "Search", "input": "Best restaurant in Paris", "output": "L'Ambroisie"},
            {"tool": "LLM", "input": "Summary: ...", "output": "The capital of France is Paris and the best restaurant there is L'Ambroisie."}
        ],
        "errors": [],
        "coordination": True,
        "goal_completed": True,
        "safety_violations": [],
    }
]
expected_outputs = [
    "10 shares cost $7000.",
    "The capital of France is Paris and the best restaurant there is L'Ambroisie."
]  # Reference answer(s)—what you want the pipeline to produce
metrics = [
    "retrieval_precision",
    "coordination_score",
    "tool_selection_accuracy",
    "goal_completion_rate"
]

evaluator = AgenticRAGEvaluator()
results = evaluator.evaluate(queries, agent_traces, expected_outputs, metrics)
print("--- LangChain Agent Trace Evaluation ---")
for k, v in results.items():
    print(f"  {k}: {v:.3f}")