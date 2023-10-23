import os
import logging
import sys
from data_cleanup import get_metadata_from_md

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import nest_asyncio
nest_asyncio.apply()
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
    set_global_service_context
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext
from llama_index import Document

current_path = os.path.dirname(os.path.dirname(__file__))
paths = [current_path + "/contents/posts/2023-04-17-topology-spread-features.md",
        current_path + "/contents/posts/2023-05-15-kubernetes-1-27-updates-on-speeding-up-pod-startup.md",
        current_path + "/contents/posts/2023-05-11-nodeport-dynamic-and-static-allocation.md",
        current_path + "/contents/posts/2023-05-02-hpa-container-resource-metric.md",
        current_path + "/contents/posts/2023-04-21-node-log-query-alpha.md"]
documents = []
for path in paths:
    title = get_metadata_from_md(path)["title"]
    one_markdown = SimpleDirectoryReader(
        input_files=[path],
        # recursive=True,
    ).load_data()
    doc = "title: " + title
    for i in one_markdown:
        doc = doc + "\n" + "header: " + i.text
    documents.append(Document(text=doc))
print(documents)

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')
service_context = ServiceContext.from_defaults(
    # embed_model="local:BAAI/bge-small-en"
    chunk_size=800,
    llm=OpenAI()
)
set_global_service_context(service_context)

summary_query = (
    "以下是有关kubernetes的文本，请用中文，尽可能详细地总结其内容，并提出一些与内容相关的问题。"
)

# doc_summary_index = DocumentSummaryIndex.from_documents(
#     documents,
#     # response_synthesizer=response_synthesizer,
#     summary_query=summary_query,
#     show_progress=True,
# )
# # print(doc_summary_index.docstore)
# # print(doc_summary_index.get_document_summary('de168450-bc88-463a-83b0-3b88be5dd279'))
# doc_summary_index.storage_context.persist("./index_store/index_multi_doc_relation")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./index_store/index_multi_doc_relation")
doc_summary_index = load_index_from_storage(storage_context)
print(doc_summary_index.index_struct.summary_id_to_node_ids)
print("-----------")
print(doc_summary_index.index_struct.node_id_to_summary_id)
print("------------")
print(doc_summary_index.index_struct.doc_id_to_summary_id)
print("-------------")
print(doc_summary_index.docstore.docs)
print("------------------")
print(doc_summary_index.docstore.get_node("32fd9825-4422-4606-b587-46c1d2fe9f20"))
print("------------------")
print(doc_summary_index.docstore.get_nodes(['9f660d3d-c4fc-46b7-a164-9e706c2ecf5b', 'f29ea54b-3508-4efd-b2e5-25896fae2d7d', '7ae71f8e-b2d9-48bd-a987-99b6e006c6fa', '6be68af7-b364-4a1c-a68c-f2230810e994', 'a6f6cbf2-fed3-4e97-9701-f1caae72c84d']))
print("-----------------")
print(doc_summary_index.docstore.get_nodes(doc_summary_index.index_struct.summary_id_to_node_ids["32fd9825-4422-4606-b587-46c1d2fe9f20"]))
# print(doc_summary_index.get_document_summary('c1e8c5b7-3294-41bb-8f3f-ae390b5f5387'))
# print(doc_summary_index.get_document_summary('35acfbaf-a1c5-41ef-81e9-a55058c41873'))
# print(doc_summary_index.get_document_summary('e3d5792f-fb41-4927-a27e-90bec4c037ee'))
# print(doc_summary_index.get_document_summary('ded38f6c-b012-4226-aae5-16c3e088bf41'))
# print(doc_summary_index.get_document_summary('a0eb0e38-5794-4b8b-8874-444c12731121'))
retriever = doc_summary_index.as_retriever(
    retriever_mode="default",
    # similarity_top_k=4
)
print("------------")
results = retriever.retrieve("如何加快pod的启动")
print(results)
for i in results:
    print(i)
    print("-------------")