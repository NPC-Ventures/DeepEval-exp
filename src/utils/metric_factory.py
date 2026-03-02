import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from dotenv import load_dotenv


class MetricFactory:
    _deepeval_backend_synced_for: Optional[tuple] = None

    def __init__(
        self,
        exec_env: Optional[str] = None,
        local_model: Optional[str] = None,
        local_base_url: Optional[str] = None,
    ):
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(env_path)

        self.exec_env = (exec_env or os.getenv("EXECUTION_ENV", "local")).lower()
        self.local_model = local_model or os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        self.local_base_url = local_base_url or os.getenv("LOCAL_MODEL_BASE_URL")
        self._sync_deepeval_backend()

    def _sync_deepeval_backend(self) -> None:
        sync_state = (self.exec_env, self.local_model, self.local_base_url)
        if MetricFactory._deepeval_backend_synced_for == sync_state:
            return

        deepeval_cli = shutil.which("deepeval")
        if deepeval_cli is None:
            return

        if self.exec_env == "local":
            command = [deepeval_cli, "set-ollama", "-q"]
            if self.local_model:
                command.extend(["-m", self.local_model])
            if self.local_base_url:
                command.extend(["-u", self.local_base_url])
        elif self.exec_env == "cloud":
            command = [deepeval_cli, "unset-ollama", "-q"]
        else:
            return

        subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        MetricFactory._deepeval_backend_synced_for = sync_state

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
