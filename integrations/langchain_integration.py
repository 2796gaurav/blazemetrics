"""
Integration Example: LangChain with BlazeMetrics

Demonstrates how to use BlazeMetrics as a callback handler in a LangChain LLM pipeline.
"""

from blazemetrics.integrations.langchain_integration import BlazeLangChainHandler

try:
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
except ImportError:
    print("LangChain is not installed. Please install with 'pip install langchain' to run this example.")
    exit(1)

handler = BlazeLangChainHandler()

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])

result = chain.run(topic="cats")
print("LangChain + BlazeMetrics Result:", result)