import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from data_cleanup import get_metadata_from_md
import chromadb
from llama_index.vector_stores import ChromaVectorStore
import nest_asyncio
nest_asyncio.apply()
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    VectorStoreIndex,
    get_response_synthesizer,
    set_global_service_context
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext
from llama_index import Document

current_path = os.path.dirname(os.path.dirname(__file__))
# paths = [current_path + "/contents/posts/2023-04-17-topology-spread-features.md",
#         current_path + "/contents/posts/2023-05-15-kubernetes-1-27-updates-on-speeding-up-pod-startup.md",
#         current_path + "/contents/posts/2023-05-11-nodeport-dynamic-and-static-allocation.md",
#         current_path + "/contents/posts/2023-05-02-hpa-container-resource-metric.md",
#         current_path + "/contents/posts/2023-04-21-node-log-query-alpha.md"]

def recursive_listdir(path, paths):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            paths.append(file_path)
        elif os.path.isdir(file_path):
            recursive_listdir(file_path, paths)

def get_documents(paths):
    documents = []
    for path in paths:
        title = get_metadata_from_md(path)["title"]
        one_markdown = SimpleDirectoryReader(
            input_files=[path],
        ).load_data()
        doc = "title: " + title
        for i in one_markdown:
            doc = doc + "\n" + "header: " + i.text
        documents.append(Document(text=doc))
    return documents

path = current_path + "/contents/posts"
paths = []
recursive_listdir(path, paths)
documents = get_documents(paths)

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')
service_context = ServiceContext.from_defaults(
    # embed_model="local:BAAI/bge-small-en"
    chunk_size=800,
    llm=OpenAI(),
)

summary_query = (
    "以下是有关kubernetes信息的文本，请以其中的title为主，header为辅，尽可能详细地总结文本内容，并提出一些与内容相关的问题。\n请用中文回答。"
)

# generate summary
# doc_summary_index = DocumentSummaryIndex.from_documents(
#     documents,
#     # response_synthesizer=response_synthesizer,
#     summary_query=summary_query,
#     show_progress=True,
#     service_context=service_context
# )
# doc_summary_index.storage_context.persist("./index_store/summary2vector/posts_summary_2")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./index_store/summary2vector/posts_summary_2")
doc_summary_index = load_index_from_storage(storage_context)
# print(doc_summary_index.index_struct.summary_id_to_node_ids)
# print("-----------")
# print(doc_summary_index.index_struct.node_id_to_summary_id)
# print("------------")
# print(doc_summary_index.index_struct.doc_id_to_summary_id)
# print("------------------")

documents_vec = []
summary_ids = doc_summary_index.index_struct.summary_id_to_node_ids
for summary_id in summary_ids:
    print(doc_summary_index.docstore.get_node(summary_id).text)
    summary_text = doc_summary_index.docstore.get_node(summary_id).text
    doc = Document(
        text = summary_text,
        metadata = {
            "summary_id" : summary_id
        }
    )
    documents_vec.append(doc)
    # print(doc_summary_index.docstore.get_nodes(summary_ids[summary_id]))
    print("#####################")

# db = chromadb.PersistentClient(path="./index_store/summary2vector/posts_vector_2")
# chroma_collection = db.get_or_create_collection("quickstart")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# service_context = ServiceContext.from_defaults(
#     # embed_model="local:BAAI/bge-small-en"
#     chunk_size=3000,
#     llm=OpenAI()
# )
# vector_index = VectorStoreIndex.from_documents(
#     documents_vec,
#     show_progress=True,
#     storage_context=storage_context,
#     service_context=service_context
# )
# # print(vector_index.docstore.docs)

db2 = chromadb.PersistentClient(path="./index_store/summary2vector/posts_vector_2")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)

query = "如何提高pod的启动速度"
retriever = index.as_retriever(similarity_top_k=2)
results = retriever.retrieve(query)
print("----------")
print(results)

texts = ""
s_id = []
for node in results:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%")
    doc_nodes = doc_summary_index.docstore.get_nodes(summary_ids[node.metadata["summary_id"]])
    print(doc_nodes)
    for node in doc_nodes:
        texts = texts + "\n" + node.text


from langchain.prompts import PromptTemplate

template = """
你是一个kubernetes助手, 你的回答基于事实，详细且准确。
目前已知：
{context}
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context"],
    )
prompt = PROMPT.format(
            context="\n\n".join(texts),
            )

rsp = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-16k",
  messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]
)

answer = rsp.get("choices")[0]["message"]["content"]
print(answer)

