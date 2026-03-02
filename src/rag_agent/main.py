from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_agent.prompt import rag_prompt
from src.utils.model_provider import ModelProvider


class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        llm_model=None,
        exec_env: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 5,
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_provider = ModelProvider(
            exec_env=exec_env,
            llm_model=llm_model if isinstance(llm_model, str) else None,
            embedding_model=(
                embedding_model if isinstance(embedding_model, str) else None
            ),
        )
        self.embedding_model = (
            embedding_model
            if embedding_model is not None and not isinstance(embedding_model, str)
            else self.model_provider.get_langchain_embeddings(
                model_name=embedding_model if isinstance(embedding_model, str) else None
            )
        )
        self.default_llm_model = (
            llm_model
            if llm_model is not None and not isinstance(llm_model, str)
            else self.model_provider.get_langchain_llm(
                model_name=llm_model if isinstance(llm_model, str) else None
            )
        )
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        raw_documents = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        for document_path in self.document_paths:
            file_path = Path(document_path)
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                raw_documents.extend(loader.load())
            else:
                raise ValueError(f"Unsupported document type: {file_path.suffix}")

        documents = splitter.split_documents(raw_documents)

        return self.vector_store_class.from_documents(documents, self.embedding_model)

    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k=self.k)
        context = [doc.page_content for doc in docs]
        return context

    def generate(
        self,
        query: str,
        retrieved_docs: list,
        llm_model=None,
        prompt_template: str = None,
    ):
        context = "\n".join(retrieved_docs)
        if llm_model is None:
            model = self.default_llm_model
        elif isinstance(llm_model, str):
            model = self.model_provider.get_langchain_llm(model_name=llm_model)
        else:
            model = llm_model
        prompt = prompt_template or rag_prompt
        prompt = prompt.format(context=context, query=query)
        response = model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def answer(self, query: str):
        retrieved_docs = self.retrieve(query)
        generated_answer = self.generate(query, retrieved_docs)
        return self._parse_json_response(generated_answer)

    async def generate_async(
        self,
        query: str,
        retrieved_docs: list,
        llm_model=None,
        prompt_template: str = None,
    ):
        context = "\n".join(retrieved_docs)
        if llm_model is None:
            model = self.default_llm_model
        elif isinstance(llm_model, str):
            model = self.model_provider.get_langchain_llm(model_name=llm_model)
        else:
            model = llm_model
        prompt = prompt_template or rag_prompt
        prompt = prompt.format(context=context, query=query)
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(prompt)
        else:
            response = await asyncio.to_thread(model.invoke, prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def answer_async(self, query: str):
        retrieved_docs = self.retrieve(query)
        generated_answer = await self.generate_async(query, retrieved_docs)
        return self._parse_json_response(generated_answer)

    def _parse_json_response(self, generated_answer: str):
        try:
            res = json.loads(generated_answer)
            return res
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON returned from model",
                "raw_output": generated_answer,
            }


if __name__ == "__main__":
    documents_dir = Path(__file__).parent / "documents"
    doc_path = [str(path) for path in documents_dir.glob("*.pdf")]

    if not doc_path:
        raise FileNotFoundError(f"No PDF files found in: {documents_dir}")

    query = "What socioeconomic impacts arise from shifting fish distributions?"
    retriever = RAGAgent(doc_path)
    answer = retriever.answer(query)
    print(answer)
