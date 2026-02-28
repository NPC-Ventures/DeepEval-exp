from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="...",  # Your query
    actual_output="...",  # The answer from RAG
    retrieval_context="...",  # Your retrieved context
)
