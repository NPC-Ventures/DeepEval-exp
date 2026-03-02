import os
import sys
import json
import asyncio
from pathlib import Path
import pytest
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test

# Add parent directory to path so we can import project modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_agent.main import RAGAgent
from src.utils.metric_factory import MetricFactory

goldens_path = Path(__file__).parent / "goldens" / "ipcc_rag_goldens.json"
with open(goldens_path, "r", encoding="utf-8") as file:
    qa_goldens = json.load(file)

dataset = EvaluationDataset(
    goldens=[
        Golden(input=item["input"], expected_output=item["expected_output"])
        for item in qa_goldens
    ]
)

documents_dir = "src/rag_agent/documents"
document_paths = [
    os.path.join(documents_dir, file_name)
    for file_name in os.listdir(documents_dir)
    if file_name.lower().endswith(".pdf")
]

if not document_paths:
    raise FileNotFoundError(f"No PDF files found in: {documents_dir}")

rag_agent = RAGAgent(document_paths=document_paths)
metric_factory = MetricFactory()

correctness_metric = metric_factory.build_metric(
    name="RAG Correctness",
    criteria="Determine whether the actual output answer is factually consistent with the expected output answer.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)

grounding_metric = metric_factory.build_metric(
    name="RAG Grounding",
    criteria="Evaluate if the actual output is grounded in the retrieval context and avoids unsupported claims or invented details.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.7,
)


@pytest.mark.parametrize("golden", dataset.goldens)
def test_rag_with_ipcc_goldens(golden: Golden):
    retrieved_docs = rag_agent.retrieve(golden.input)
    result = asyncio.run(rag_agent.answer_async(golden.input))
    actual_output = (
        result.get("answer")
        if isinstance(result, dict)
        else "No relevant information available."
    )

    test_case = LLMTestCase(
        name=f"RAG QA: {golden.input[:60]}",
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        retrieval_context=retrieved_docs,
    )

    assert_test(test_case=test_case, metrics=[correctness_metric, grounding_metric])
