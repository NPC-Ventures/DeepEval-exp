from string import Template

rag_prompt = """
You are a helpful assistant. Use the context below to answer the user's query. 
Format your response strictly as a JSON object with the following structure:

{{
  "answer": "<a concise, complete answer to the user's query>",
  "citations": [
    "<relevant quoted snippet or summary from source 1>",
    "<relevant quoted snippet or summary from source 2>",
    ...
  ]
}}

Only include information that appears in the provided context. Do not make anything up.
Only respond in JSON — No explanations needed. Only use information from the context. If 
nothing relevant is found, respond with: 

{{
  "answer": "No relevant information available.",
  "citations": []
}}


Context:
{context}

Query:
{query}
"""
