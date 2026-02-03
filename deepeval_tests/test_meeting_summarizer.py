from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import os
import sys
from deepeval.dataset import Golden, EvaluationDataset

# Add parent directory to path so we can import meeting_summarizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meeting_summarizer.meeting_summarizer import MeetingSummarizer
from deepeval import evaluate
from src.utils.evaluator import Evaluator
from deepeval import assert_test
import pytest

documents_path = "src/meeting_summarizer/transcripts"
transcripts = []

for document in os.listdir(documents_path):
    if document.endswith(".txt"):
        file_path = os.path.join(documents_path, document)
        with open(file_path, "r") as file:
            transcript = file.read().strip()
        transcripts.append(transcript)

goldens = []
for transcript in transcripts:
    golden = Golden(input=transcript)
    goldens.append(golden)

for i, golden in enumerate(goldens):
    print(f"Golden {i}: ", golden.input[:20])

dataset = EvaluationDataset(goldens=goldens)


summarizer = MeetingSummarizer()  # Initialize with your best config


@pytest.mark.parametrize("golden", dataset.goldens)
def test(golden: Golden):
    summary, action_items = summarizer.summarize(golden.input)
    summary_test_case = LLMTestCase(
        name=f"Summary Test for {golden.input[:20]}",
        input=golden.input,
        actual_output=summary,
    )
    action_item_test_case = LLMTestCase(
        name=f"Action Item Test for {golden.input[:20]}",
        input=golden.input,
        actual_output=str(action_items),
    )

    evaluator = Evaluator()

    summary_concision = evaluator.build_metric(
        name="Summary Concision",
        # Write your criteria here
        criteria="Assess whether the summary is concise and focused only on the essential points of the meeting? It should avoid repetition, irrelevant details, and unnecessary elaboration.",
        threshold=0.9,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    action_item_check = evaluator.build_metric(
        name="Action Item Accuracy",
        # Write your criteria here
        criteria="Are the action items accurate, complete, and clearly reflect the key tasks or follow-ups mentioned in the meeting?",
        threshold=0.9,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    assert_test(test_case=summary_test_case, metrics=[summary_concision])
    assert_test(test_case=action_item_test_case, metrics=[action_item_check])
