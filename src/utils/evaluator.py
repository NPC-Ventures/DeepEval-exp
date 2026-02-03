import os
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from dotenv import load_dotenv


class Evaluator:
    def __init__(self):
        load_dotenv(".env")
        self.exec_env = os.getenv("EXECUTION_ENV", "local").lower()
        self.model = os.getenv("LOCAL_MODEL_NAME") if self.exec_env == "local" else None
        self.base_url = (
            os.getenv("LOCAL_MODEL_BASE_URL") if self.exec_env == "local" else None
        )

    def build_metric(
        self, name, criteria, evaluation_params, threshold=0.5, model=None
    ):
        if self.exec_env == "local":
            return GEval(
                name=name,
                criteria=criteria,
                evaluation_params=evaluation_params,
                threshold=threshold,
                model=OllamaModel(
                    model=self.model,
                    base_url=self.base_url,
                ),
            )
        elif self.exec_env == "cloud":
            return GEval(
                name=name,
                criteria=criteria,
                evaluation_params=evaluation_params,
                threshold=threshold,
            )

        else:
            raise ValueError(f"Unsupported execution environment: {self.exec_env}")
