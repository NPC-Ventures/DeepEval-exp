# from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# import os
# import sys
# from deepeval.dataset import Golden, EvaluationDataset

# # Add parent directory to path so we can import meeting_summarizer
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.meeting_summarizer.meeting_summarizer import MeetingSummarizer
# from deepeval import evaluate
# import json
# from src.utils.evaluator import Evaluator

# documents_path = "src/meeting_summarizer/transcripts"
# transcripts = []

# for document in os.listdir(documents_path):
#     if document.endswith(".txt"):
#         file_path = os.path.join(documents_path, document)
#         with open(file_path, "r") as file:
#             transcript = file.read().strip()
#         transcripts.append(transcript)

# goldens = []
# for transcript in transcripts:
#     golden = Golden(input=transcript)
#     goldens.append(golden)

# for i, golden in enumerate(goldens):
#     print(f"Golden {i}: ", golden.input[:20])

# dataset = EvaluationDataset(goldens=goldens)


# summarizer = MeetingSummarizer()  # Initialize with your best config
# summary_test_cases = []
# action_item_test_cases = []
# for golden in dataset.goldens:
#     summary, action_items = summarizer.summarize(golden.input)
#     print("*" * 40)
#     print(summary)
#     print("*" * 40)
#     print("JSON:")
#     print(json.dumps(action_items, indent=2))
#     print("*" * 40)
#     summary_test_case = LLMTestCase(input=golden.input, actual_output=summary)
#     action_item_test_case = LLMTestCase(
#         input=golden.input, actual_output=str(action_items)
#     )
#     summary_test_cases.append(summary_test_case)
#     action_item_test_cases.append(action_item_test_case)

# evaluator = Evaluator()

# summary_concision = evaluator.build_metric(
#     name="Summary Concision",
#     # Write your criteria here
#     criteria="Assess whether the summary is concise and focused only on the essential points of the meeting? It should avoid repetition, irrelevant details, and unnecessary elaboration.",
#     threshold=0.9,
#     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
# )

# action_item_check = evaluator.build_metric(
#     name="Action Item Accuracy",
#     # Write your criteria here
#     criteria="Are the action items accurate, complete, and clearly reflect the key tasks or follow-ups mentioned in the meeting?",
#     threshold=0.9,
#     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
# )

# evaluate(test_cases=summary_test_cases, metrics=[summary_concision])

# evaluate(test_cases=action_item_test_cases, metrics=[action_item_check])
