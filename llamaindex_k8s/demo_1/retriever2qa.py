import os
import logging
import sys
import chromadb
from llama_index.vector_stores import ChromaVectorStore
import openai
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext
)
from llama_index.llms import OpenAI
from llama_index.indices.loading import load_index_from_storage
from langchain.prompts import PromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

# rebuild storage context (load summary)
storage_context = StorageContext.from_defaults(persist_dir="../index_store/summary2vector/posts_summary_2")
doc_summary_index = load_index_from_storage(storage_context)
summary_ids = doc_summary_index.index_struct.summary_id_to_node_ids

# load vectorindex from chroma database
db2 = chromadb.PersistentClient(path="../index_store/summary2vector/posts_vector_2")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
service_context = ServiceContext.from_defaults(
    # embed_model="local:BAAI/bge-small-en"
    chunk_size=5000,
    llm=OpenAI()
)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)

# retriever
query = "如何提高pod的启动速度"
retriever = index.as_retriever(similarity_top_k=2)
results = retriever.retrieve(query)

# context
texts = ""
s_id = []
for node in results:
    doc_nodes = doc_summary_index.docstore.get_nodes(summary_ids[node.metadata["summary_id"]])
    for node in doc_nodes:
        texts = texts + "\n" + node.text
    texts = texts + "\n\n----------------\n\n"

# prompt and chat
template = """
你是一个kubernetes助手，你的回答基于事实，详细且准确。
目前已知信息如下：
{context}
如果所给信息与问题无关，请忽略。
"""
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context"],
)
prompt = PROMPT.format(context=texts)
rsp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]
)
answer = rsp.get("choices")[0]["message"]["content"]

rsp_origin = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {"role": "user", "content": query}
    ]
)
answer_origin = rsp_origin.get("choices")[0]["message"]["content"]

print("prompt:\n", prompt)
print("query:\n", query)
print("original answer:\n", answer_origin)
print("answer:\n", answer)
