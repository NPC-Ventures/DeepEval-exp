import os
from typing import Optional

import ollama
from dotenv import load_dotenv
from openai import OpenAI

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class ModelProvider:
    def __init__(
        self,
        exec_env: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        load_dotenv(".env")
        self.exec_env = (exec_env or os.getenv("EXECUTION_ENV", "local")).lower()
        self.base_url = base_url or os.getenv(
            "LOCAL_MODEL_BASE_URL", "http://localhost:11434"
        )
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        local_llm_default = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        local_embedding_default = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
        cloud_llm_default = os.getenv("CLOUD_LLM_MODEL", "gpt-4o-mini")
        cloud_embedding_default = os.getenv(
            "CLOUD_EMBEDDING_MODEL", "text-embedding-3-small"
        )

        if self.exec_env == "local":
            self.llm_model = llm_model or local_llm_default
            self.embedding_model = embedding_model or local_embedding_default
        elif self.exec_env == "cloud":
            self.llm_model = llm_model or cloud_llm_default
            self.embedding_model = embedding_model or cloud_embedding_default
        else:
            raise ValueError(f"Unsupported execution environment: {self.exec_env}")

        self.openai_client = (
            OpenAI(api_key=self.openai_api_key) if self.exec_env == "cloud" else None
        )

    def get_langchain_embeddings(self, model_name: Optional[str] = None):
        selected_model = model_name or self.embedding_model
        if self.exec_env == "local":
            return OllamaEmbeddings(model=selected_model, base_url=self.base_url)
        return OpenAIEmbeddings(model=selected_model, api_key=self.openai_api_key)

    def get_langchain_llm(self, model_name: Optional[str] = None):
        selected_model = model_name or self.llm_model
        if self.exec_env == "local":
            return OllamaLLM(model=selected_model, base_url=self.base_url)
        return ChatOpenAI(model=selected_model, api_key=self.openai_api_key)

    def chat_completion(
        self, system_prompt: str, user_prompt: str, model_name: Optional[str] = None
    ) -> str:
        selected_model = model_name or self.llm_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.exec_env == "local":
            response = ollama.chat(model=selected_model, messages=messages)
            return response["message"]["content"].strip()

        response = self.openai_client.chat.completions.create(
            model=selected_model, messages=messages
        )
        return response.choices[0].message.content.strip()
