from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from prompt import rag_prompt
import json


class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 2,
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model="nomic-embed-text"
        )
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        documents = []
        for document_path in self.document_paths:
            with open(document_path, "r", encoding="utf-8") as file:
                raw_text = file.read()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            documents.extend(splitter.create_documents([raw_text]))

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
        model = llm_model or OllamaLLM(model="llama3.2")
        prompt = prompt_template or rag_prompt
        prompt = prompt.format(context=context, query=query)
        print(prompt)
        return model.invoke(prompt)

    def answer(self, query: str):
        retrieved_docs = self.retrieve(query)
        generated_answer = self.generate(query, retrieved_docs)

        try:
            res = json.loads(generated_answer)
            return res
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON returned from model",
                "raw_output": generated_answer,
            }


doc_path = ["sample.txt"]
query = "which research fields are mentioned?"

retriever = RAGAgent(doc_path)
answer = retriever.answer(query)
print(answer)
