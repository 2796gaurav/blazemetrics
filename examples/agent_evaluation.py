"""
Example: LLM Agent Evaluation Framework

This script demonstrates how to use the AgentEvaluator from blazemetrics
to evaluate LLM agent performance across tool usage, reasoning, safety, and goals.
"""

from blazemetrics import AgentEvaluator

# Dummy tasks and traces
complex_tasks = [
    "Find the capital of France and calculate its population density.",
    "Write Python code to reverse a string safely."
]

execution_traces = [
    {
        "steps": [
            {"tool": "web_search", "query": "capital of France", "result": "Paris"},
            {"tool": "calculator", "operation": "population / area", "result": 21400}
        ],
        "reasoning": "Identify capital, then calculate density.",
        "achieved_goal": True,
        "safe": True
    },
    {
        "steps": [
            {"tool": "code_executor", "code": "s[::-1]", "result": "gfedcba"}
        ],
        "reasoning": "Use Python slicing to reverse string.",
        "achieved_goal": True,
        "safe": True
    }
]

# Initialize evaluator
evaluator = AgentEvaluator(
    available_tools=['web_search', 'calculator', 'code_executor'],
    safety_policies=['no_harmful_actions', 'privacy_protection'],
    goal_tracking=True
)

results = evaluator.evaluate(
    tasks=complex_tasks,
    agent_traces=execution_traces,
    metrics=[
        'tool_selection_accuracy',
        'reasoning_coherence',
        'goal_completion_rate',
        'safety_compliance_score',
        'efficiency_ratio'
    ]
)

print("Agent Evaluation Results:", results)