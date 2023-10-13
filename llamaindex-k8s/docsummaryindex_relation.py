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
        doc = doc + i.text
    documents.append(Document(text=doc))
print(documents)

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')
service_context = ServiceContext.from_defaults(
    # embed_model="local:BAAI/bge-small-en"
    chunk_size=500,
    llm=OpenAI()
)
set_global_service_context(service_context)
summary_query = (
    "总结以下文本的内容，并且提出一些与内容相关的问题。"
)
DEFAULT_SUMMARY_QUERY = (
    "Describe what the provided text is about. "
    "Also describe some of the questions that this text can answer. "
)
# doc_summary_index = DocumentSummaryIndex.from_documents(
#     documents,
#     # response_synthesizer=response_synthesizer,
#     summary_query=summary_query,
#     show_progress=True,
# )
# print(doc_summary_index.docstore)
# # print(doc_summary_index.get_document_summary('de168450-bc88-463a-83b0-3b88be5dd279'))
# doc_summary_index.storage_context.persist("./index_store/index_multi_doc")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./index_store/index_multi_doc")
doc_summary_index = load_index_from_storage(storage_context)
print(doc_summary_index.get_document_summary('b7b37416-641e-4246-a281-2643b516ad41'))
print(doc_summary_index.get_document_summary('02ff86b7-2f01-445c-a168-9b5c5d407e16'))
print(doc_summary_index.get_document_summary('81ea082a-2225-4eae-a769-241197f869c8'))
print(doc_summary_index.get_document_summary('9ecfd9df-0cb5-4c1a-8dc9-729e365475d5'))
print(doc_summary_index.get_document_summary('9b522367-1c43-447f-8242-750fcdf9aeda'))
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
