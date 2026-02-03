from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from src.utils.evaluator import Evaluator

evaluator = Evaluator()
correctness_metric = evaluator.build_metric(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.5,
)

input = "I have a persistent cough and fever. Should I be worried?"
actual_output = "A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen, persist beyond a few days, or if you experience difficulty breathing, chest pain, or other concerning signs."
expected_output = "A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."

test_case = LLMTestCase(
    input=input,
    actual_output=actual_output,
    expected_output=expected_output,
)

evaluate([test_case], [correctness_metric])
