"""
Integration Example: LlamaIndex with BlazeMetrics

Demonstrates how to use BlazeMetrics as a callback handler in a LlamaIndex pipeline.
"""

from blazemetrics.integrations.llamaindex_integration import BlazeLlamaIndexHandler

try:
    from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
except ImportError:
    print("LlamaIndex is not installed. Please install with 'pip install llama-index' to run this example.")
    exit(1)

handler = BlazeLlamaIndexHandler()

# Example: Load documents and build index
documents = [SimpleDirectoryReader("docs").load_data()]
service_context = ServiceContext(callbacks=[handler])
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# Query the index
query = "What is BlazeMetrics?"
response = index.query(query)
print("LlamaIndex + BlazeMetrics Response:", response)