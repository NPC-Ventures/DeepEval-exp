import os
import sys
import pytest
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test

# Add parent directory to path so we can import project modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_agent.main import RAGAgent
from src.utils.evaluator import Evaluator


QA_GOLDENS = [
    {
        "input": "What proportion of Earth’s surface is covered by the ocean?",
        "expected_output": "The ocean covers about 71% of Earth’s surface and contains approximately 97% of Earth’s water.",
    },
    {
        "input": "How many people rely on ocean and inland waters for dietary animal protein?",
        "expected_output": "More than 3.3 billion people obtain over 20% of their dietary animal protein from ocean and inland water resources.",
    },
    {
        "input": "At what average rate have marine species shifted geographically due to warming?",
        "expected_output": "Marine taxa and communities have shifted poleward at an average rate of about 59.2 ± 15.5 km per decade since the 1950s.",
    },
    {
        "input": "How have seasonal biological events changed in marine organisms?",
        "expected_output": "Seasonal events now occur earlier, advancing by 4.3–7.5 days per decade for plankton and about 3 days per decade for fish.",
    },
    {
        "input": "What major physical and chemical ocean changes are linked to climate change?",
        "expected_output": "Key changes include ocean warming, acidification, and deoxygenation, which alter marine ecosystems and species distributions.",
    },
    {
        "input": "Why are marine heatwaves considered dangerous for ecosystems?",
        "expected_output": "Marine heatwaves expose species to conditions beyond tolerance limits, causing mass mortality, biodiversity loss, and long-lasting ecosystem shifts.",
    },
    {
        "input": "How does climate change interact with non-climate stressors in the ocean?",
        "expected_output": "Climate change amplifies impacts from pollution, overfishing, habitat degradation, and invasive species, increasing ecosystem vulnerability and ecological damage.",
    },
    {
        "input": "What impacts do marine heatwaves have on human communities?",
        "expected_output": "Marine heatwaves can cause fisheries collapse, reduced coastal protection, economic losses, and disruptions to livelihoods dependent on marine ecosystems.",
    },
    {
        "input": "Why are coastal ecosystems important for climate regulation?",
        "expected_output": "Ocean and coastal systems regulate global climate by redistributing heat, cycling water and elements, and absorbing atmospheric carbon dioxide.",
    },
    {
        "input": "What happens to biodiversity as ocean warming increases beyond 2°C?",
        "expected_output": "Risks of species extirpation, extinction, and ecosystem collapse increase rapidly when warming exceeds 2°C by 2100.",
    },
    {
        "input": "How does climate change affect marine food webs?",
        "expected_output": "Climate-driven changes alter species distributions, timing of ecological events, and biomass levels, disrupting food webs and reducing ecological connectivity.",
    },
    {
        "input": "Why are coral reefs especially vulnerable to climate change?",
        "expected_output": "Increasing marine heatwaves can cause irreversible phase shifts in coral reef ecosystems, degrading structure and reducing growth rates that may not keep pace with sea level rise.",
    },
    {
        "input": "How does sea level rise affect coastal communities?",
        "expected_output": "Sea level rise increases coastal erosion, flooding, habitat loss, and groundwater salinisation, reducing shoreline protection and increasing risks to people and infrastructure.",
    },
    {
        "input": "What role does adaptive governance play in reducing climate risks?",
        "expected_output": "Inclusive governance integrating scientific, Indigenous, and local knowledge supports sustainable management of shifting marine resources and reduces human vulnerability.",
    },
    {
        "input": "Why are adaptation measures alone insufficient to protect marine ecosystems?",
        "expected_output": "Adaptation can reduce impacts but cannot fully offset climate-change effects without strong mitigation, and higher warming reduces adaptation effectiveness over time.",
    },
    {
        "input": "How do nature-based solutions help ocean adaptation?",
        "expected_output": "Nature-based solutions like habitat restoration and ecosystem-based management provide biodiversity conservation, coastal protection, carbon storage, and social benefits, but effectiveness declines under higher warming.",
    },
    {
        "input": "What socioeconomic impacts arise from shifting fish distributions?",
        "expected_output": "Poleward shifts in fish stocks force fisheries to change harvesting locations and practices, affecting commercial, artisanal, and recreational fishing economies.",
    },
    {
        "input": "Why is transformative adaptation necessary under high-emission scenarios?",
        "expected_output": "Under high emissions, traditional adaptation is insufficient and major institutional and governance changes are needed to reduce rising inequality and conflict risks.",
    },
    {
        "input": "Does the chapter conclude that marine protected areas fully prevent climate change impacts?",
        "expected_output": "No. Marine protected areas alone do not fully prevent climate change impacts unless designed and governed to address warming and heatwaves.",
    },
]

dataset = EvaluationDataset(
    goldens=[
        Golden(input=item["input"], expected_output=item["expected_output"])
        for item in QA_GOLDENS
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
evaluator = Evaluator()

correctness_metric = evaluator.build_metric(
    name="RAG Correctness",
    criteria="Determine whether the actual output answer is factually consistent with the expected output answer.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)

grounding_metric = evaluator.build_metric(
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
    result = rag_agent.answer(golden.input)
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
