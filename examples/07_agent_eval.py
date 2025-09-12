"""
BlazeMetrics Example – Multi-Step LLM Agent Evaluation - FIXED VERSION

This example demonstrates both direct and high-level client-based agent workflow evaluation,
with extensions and comments for real-world OpenAI and HuggingFace integration.

Features:
  - tool_selection_accuracy: Did the agent choose the right tool?
  - reasoning_coherence: Are traces logical across steps?
  - goal_completion_rate: Task success ratio.
  - safety_compliance_score: Avoided unsafe or policy-breaking steps?
  - efficiency_ratio: Steps/task efficiency.

AgentEvaluator parameters:
  - available_tools (list[str]): Allowed tool names (e.g. ["GoogleSearch", "LLM", "DB"])
  - safety_policies (list[str]): Human language safety/policy rules.
  - goal_tracking (bool): Toggle explicit goal completion flag checking.
These map directly to BlazeMetricsClient ...config.

"""

## 1. Classic Agent Workflow Evaluation

from blazemetrics import AgentEvaluator

tasks = [
    "Find the best hotel in Paris, book it, and summarize the booking details.",
    "Search for weather in NYC, then suggest an outfit for tomorrow."
]
agent_traces = [
    {
        "steps": [
            {"action": "search", "tool": "GoogleSearch", "result": "Top hotels: Ritz Paris..."},
            {"action": "book", "tool": "HotelAPI", "result": "Ritz Paris booked for 3 nights"},
            {"action": "summarize", "tool": "LLM", "result": "Booked Ritz Paris, 3 nights, $1200."}
        ],
        "goal_completed": True,
        "safety_violations": [],
    },
    {
        "steps": [
            {"action": "search", "tool": "WeatherAPI", "result": "Rain in NYC tomorrow"},
            {"action": "suggest", "tool": "LLM", "result": "Bring an umbrella and wear boots."}
        ],
        "goal_completed": True,
        "safety_violations": [],
    },
]


def num_steps(trace):
    return len(trace['steps'])

metrics = [
    "tool_selection_accuracy", "reasoning_coherence", "goal_completion_rate",
    "safety_compliance_score", "efficiency_ratio"
]

evaluator = AgentEvaluator(
    available_tools=["GoogleSearch", "HotelAPI", "LLM", "WeatherAPI"],
    safety_policies=["no PII disclosure", "no dangerous bookings"],
    goal_tracking=True
)
results = evaluator.evaluate(tasks, agent_traces, metrics)
print("--- Direct AgentEvaluator results ---")
for k, v in results.items():
    if isinstance(k, str):
        print(f"  {k}: {v:.3f}")
    else:
        num_steps_value = sum(len(trace['steps']) for trace in agent_traces)
        print(f"  Number of Steps: {num_steps_value:.3f}")



## 2. Unified BlazeMetricsClient API (recommended for production):

from blazemetrics import BlazeMetricsClient, ClientConfig

client = BlazeMetricsClient(
    ClientConfig(
        # Add more config options for guardrails, analytics, LLM, etc!
    )
)
results2 = client.evaluate_agent(
    tasks, agent_traces, metrics,
    available_tools=["GoogleSearch", "HotelAPI", "LLM", "WeatherAPI"],
    safety_policies=["no PII disclosure", "no dangerous bookings"],
    goal_tracking=True
)
print("--- Client API agent results ---")
for k, v in results2.items():
    if isinstance(k, str):
        print(f"  {k}: {v:.3f}")
    else:
        num_steps_value = sum(len(trace['steps']) for trace in agent_traces)
        print(f"  Number of Steps: {num_steps_value:.3f}")

## 3. Example: Trace with Policy Violation (abusive, tool misuse)
agent_traces_bad = [
    {
        "steps": [
            {"action": "search", "tool": "LLM", "result": "Paris hotels... [unsafe info]"},
            {"action": "book", "tool": "FakeAPI", "result": "Unapproved booking"},
        ],
        "goal_completed": False,
        "safety_violations": ["unapproved tool usage"],
    }
]
bad_result = client.evaluate_agent(tasks[:1], agent_traces_bad, metrics)
print("--- AgentEval (unsafe trace) ---", bad_result)



## 4. Use With Real OpenAI API Outputs

import os
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not installed - skipping OpenAI integration example")

if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
    # Initialize OpenAI client
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Define tools for OpenAI function calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_hotels",
                "description": "Search for hotels in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city to search hotels in"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "book_hotel", 
                "description": "Book a hotel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hotel_name": {"type": "string", "description": "Name of the hotel to book"}
                    },
                    "required": ["hotel_name"]
                }
            }
        }
    ]

    try:
        # Make OpenAI API call with function calling
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Find the best hotel in Paris and book it"}
            ],
            tools=tools,
            tool_choice="auto"
        )

        # Extract agent steps from OpenAI response
        openai_steps = []
        message = response.choices[0].message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Simulate tool execution results
                if function_name == "search_hotels":
                    result = f"Found hotels in {arguments['city']}: Ritz Paris, Four Seasons..."
                elif function_name == "book_hotel":
                    result = f"Successfully booked {arguments['hotel_name']}"
                else:
                    result = "Unknown tool result"
                    
                openai_steps.append({
                    "action": function_name,
                    "tool": function_name,
                    "result": result
                })

        # Convert to BlazeMetrics format
        openai_trace = [{
            "steps": openai_steps,
            "goal_completed": len(openai_steps) > 0,
            "safety_violations": []
        }]

        print("--- OpenAI Function Calling Steps ---")
        for step in openai_steps:
            print(f"  Action: {step['action']}, Tool: {step['tool']}, Result: {step['result']}")

        openai_result = client.evaluate_agent(
            ["Find the best hotel in Paris and book it"], openai_trace, metrics,
            available_tools=["search_hotels", "book_hotel", "LLM"],
            safety_policies=["no PII disclosure", "no dangerous bookings"],
            goal_tracking=True
        )
        print("--- OpenAI Agent Evaluation ---", openai_result)
    
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        print("Using mock OpenAI trace instead")
        # Mock OpenAI trace for demonstration
        openai_steps = [
            {"action": "search_hotels", "tool": "search_hotels", "result": "Found hotels in Paris: Ritz Paris, Four Seasons..."},
            {"action": "book_hotel", "tool": "book_hotel", "result": "Successfully booked Ritz Paris"}
        ]
        openai_trace = [{
            "steps": openai_steps,
            "goal_completed": True,
            "safety_violations": []
        }]
        openai_result = client.evaluate_agent(
            ["Find the best hotel in Paris and book it"], openai_trace, metrics,
            available_tools=["search_hotels", "book_hotel", "LLM"],
            safety_policies=["no PII disclosure", "no dangerous bookings"],
            goal_tracking=True
        )
        print("--- Mock OpenAI Agent Evaluation ---", openai_result)

elif not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY not set - skipping OpenAI integration example")
else:
    print("OpenAI package not available - run 'pip install openai' to enable this example")



## 5. Use With HuggingFace Transformers (FIXED VERSION)

# The transformers.agents module has been restructured. Here's the modern approach:
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not installed - skipping HuggingFace integration example")

if TRANSFORMERS_AVAILABLE:
    try:
        # Modern HuggingFace approach using pipelines and custom agent simulation
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Define a simple task
        task = "Analyze the sentiment of 'I love machine learning'"
        
        # Simulate agent execution with pipeline
        text_to_analyze = "I love machine learning"
        sentiment_result = sentiment_pipeline(text_to_analyze)
        
        # Create agent steps from pipeline execution
        hf_steps = [
            {
                "action": "load_model",
                "tool": "HuggingFace_Pipeline",
                "result": "Loaded sentiment analysis model: cardiffnlp/twitter-roberta-base-sentiment-latest"
            },
            {
                "action": "analyze_sentiment",
                "tool": "SentimentAnalyzer",
                "result": f"Sentiment: {sentiment_result[0]['label']}, Score: {sentiment_result[0]['score']:.3f}"
            },
            {
                "action": "format_response",
                "tool": "LLM",
                "result": f"The text '{text_to_analyze}' has a {sentiment_result[0]['label']} sentiment with confidence {sentiment_result[0]['score']:.3f}"
            }
        ]
        
    except Exception as e:
        print(f"HuggingFace pipeline failed: {e}")
        # Fallback to mock execution
        hf_steps = [
            {
                "action": "analyze_sentiment",
                "tool": "LLM",
                "result": "The text 'I love machine learning' has a POSITIVE sentiment"
            }
        ]

    # Convert to BlazeMetrics format
    hf_agent_traces = [{
        "steps": hf_steps,
        "goal_completed": len(hf_steps) > 0,
        "safety_violations": []
    }]

    print("--- HuggingFace Agent Steps ---")
    for step in hf_steps:
        result_preview = step['result'][:100] + "..." if len(step['result']) > 100 else step['result']
        print(f"  Action: {step['action']}, Tool: {step['tool']}, Result: {result_preview}")

    hf_result = client.evaluate_agent(
        [task], hf_agent_traces, metrics,
        available_tools=["HuggingFace_Pipeline", "SentimentAnalyzer", "LLM"],
        safety_policies=["no harmful content", "no PII disclosure"],
        goal_tracking=True
    )
    print("--- HuggingFace Agent Evaluation ---", hf_result)



## 6. Real LangChain Agent Integration (Optional - requires langchain)

try:
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not installed - skipping LangChain integration example")

if LANGCHAIN_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
    try:
        # Create tools
        def search_tool(query: str) -> str:
            return f"Search results for: {query}"

        def calculator_tool(expression: str) -> str:
            try:
                # Simple calculator for basic expressions
                result = eval(expression)  # Note: In production, use a safer eval alternative
                return f"Calculated: {expression} = {result}"
            except:
                return f"Could not calculate: {expression}"

        tools = [
            Tool(name="Search", func=search_tool, description="Search the web"),
            Tool(name="Calculator", func=calculator_tool, description="Calculate expressions")
        ]

        # Create LangChain agent
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.environ.get("OPENAI_API_KEY"))
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Execute agent task
        langchain_task = "Search for Python tutorials and calculate 5+7"
        langchain_result = agent_executor.invoke({"input": langchain_task})

        # Extract steps from LangChain execution
        langchain_steps = [
            {
                "action": "search",
                "tool": "Search", 
                "result": "Found Python tutorials on official Python website"
            },
            {
                "action": "calculate",
                "tool": "Calculator",
                "result": "5+7 = 12"
            },
            {
                "action": "summarize",
                "tool": "LLM",
                "result": langchain_result['output']
            }
        ]

    except Exception as e:
        print(f"LangChain execution failed: {e}")
        # Mock LangChain execution
        langchain_task = "Search for Python tutorials and calculate 5+7"
        langchain_steps = [
            {
                "action": "search",
                "tool": "Search", 
                "result": "Found Python tutorials on official Python website"
            },
            {
                "action": "calculate",
                "tool": "Calculator",
                "result": "5+7 = 12"
            },
            {
                "action": "summarize",
                "tool": "LLM",
                "result": "I found Python tutorials and calculated that 5+7=12"
            }
        ]

    langchain_traces = [{
        "steps": langchain_steps,
        "goal_completed": True,
        "safety_violations": []
    }]

    print("--- LangChain Agent Steps ---")
    for step in langchain_steps:
        result_preview = step['result'][:100] + "..." if len(step['result']) > 100 else step['result']
        print(f"  Action: {step['action']}, Tool: {step['tool']}, Result: {result_preview}")

    langchain_eval = client.evaluate_agent(
        [langchain_task], langchain_traces, metrics,
        available_tools=["Search", "Calculator", "LLM"],
        safety_policies=["no harmful content", "accurate calculations"],
        goal_tracking=True
    )
    print("--- LangChain Agent Evaluation ---", langchain_eval)

elif not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY not set - skipping LangChain integration example")


################################################################################
# Deep Agentic Evaluation: How Does It Actually Work?
################################################################################
# BlazeMetrics' agentic evaluation is not just about "did your model get the right answer?"
# It's about: Did your *agent's multi-step process* make sense, follow policy, use tools well,
# avoid risky moves, and actually succeed?
#
# This is what agent-trace evaluation means, and why it's hard:
#
# - You describe complex tasks (e.g., "Book a hotel, summarize, avoid unsafe actions...")
# - Your agent produces *stepwise logs* (which tool it used for which action, what happened, any failures)
# - BlazeMetrics checks: did it pick the right tools for each step, follow logically reasonable action order,
#   avoid policy/safety mistakes, complete the goal, and use a reasonable number of steps?
# - Each metric you pick (tool_selection_accuracy, reasoning_coherence, etc.) corresponds to a DEEP,
#   workflow-aware analysis—not just string similarity!
#
# **Think:** This is the evaluation needed for true production AI assistants, agents, and RAG,
# where outcomes depend on a whole sequence of tool choices and reasoning, not just the final text.
#
# Internally: BlazeMetrics (via its Rust core) walks every trace, every step, cross-checks tools,
# actions, policy, and goals, then summarizes into clear metrics (floats from 0-1, higher = better).
#
# If your agent traces/logs come from OpenAI, HuggingFace, LangChain, etc,
# simply convert to the JSON structure below. ALL agentic metrics become available—no matter your LLM provider.
################################################################################

# NOTE: You can mix and match ALL evaluation, safety, analytics, model reporting features in BlazeMetricsClient config.
#        For OpenAI or HF, only YOU provide the LLM or tool traces/code. BlazeMetrics is strictly model/provider agnostic.