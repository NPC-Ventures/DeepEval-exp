import os
from pathlib import Path
from typing import Any, Optional

from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from dotenv import load_dotenv


class MetricFactory:
    def __init__(
        self,
        exec_env: Optional[str] = None,
        local_model: Optional[str] = None,
        local_base_url: Optional[str] = None,
    ):
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(env_path)

        self.exec_env = (exec_env or os.getenv("EXECUTION_ENV", "local")).lower()
        self.local_model = local_model or os.getenv("LOCAL_LLM_MODEL")
        self.local_base_url = local_base_url or os.getenv("LOCAL_MODEL_BASE_URL")

    def build_metric(
        self,
        name: str,
        criteria: str,
        evaluation_params: list,
        threshold: float = 0.5,
        model: Any = None,
    ) -> GEval:
        if self.exec_env == "local":
            metric_model = model or OllamaModel(
                model=self.local_model,
                base_url=self.local_base_url,
            )
            return GEval(
                name=name,
                criteria=criteria,
                evaluation_params=evaluation_params,
                threshold=threshold,
                model=metric_model,
            )
        if self.exec_env == "cloud":
            metric_kwargs = {}
            if model is not None:
                metric_kwargs["model"] = model
            return GEval(
                name=name,
                criteria=criteria,
                evaluation_params=evaluation_params,
                threshold=threshold,
                **metric_kwargs,
            )

        raise ValueError(f"Unsupported execution environment: {self.exec_env}")
