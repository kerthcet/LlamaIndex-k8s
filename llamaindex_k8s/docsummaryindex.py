import os
import logging
import sys

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

current_path = os.path.dirname(os.path.dirname(__file__))
filename_1 = "/contents/posts/2023-04-17-topology-spread-features.md"
filename_2 = "/contents/posts/2022-10-18-kubernetes-1.26-deprecations-and-removals.md"
file_dir_1 = current_path + filename_1
file_dir_2 = current_path + filename_2
documents = SimpleDirectoryReader(
    input_dir=current_path+"/contents/posts",
    input_files=[file_dir_1, file_dir_2],
    # recursive=True,
    ).load_data()

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

service_context = ServiceContext.from_defaults(
    # embed_model="local:BAAI/bge-small-en"
    chunk_size=3000,
    llm=OpenAI(
        api_key= os.getenv("OPENAI_API_KEY"),
        api_base= os.getenv('OPENAI_ENDPOINT')
))
set_global_service_context(service_context)

# default mode of building the index
# response_synthesizer = get_response_synthesizer(
#     # response_mode="tree_summarize",
#     use_async=True
# )

doc = []
doc.extend(documents)
# print(len(documents))
# for i in documents:
#     print(i)
#     print("--------------")
# doc_summary_index = DocumentSummaryIndex.from_documents(
#     doc,
#     # response_synthesizer=response_synthesizer,
#     show_progress=True,
# )
# # retriever = doc_summary_index.as_retriever()
# # results = retriever.retrieve("什么是最小域数")
# # print(results)
#
# doc_summary_index.storage_context.persist("index")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./index")
doc_summary_index = load_index_from_storage(storage_context)
retriever = doc_summary_index.as_retriever(
    retriever_mode="default",
    similarity_top_k=4)
print("------------")
results = retriever.retrieve("有关pod拓扑分布")
print(results)
for i in results:
    print(i)
    print("-------------")
