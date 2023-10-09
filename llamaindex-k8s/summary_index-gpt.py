import os
from llama_index import SimpleDirectoryReader
from typing import List, Set
from data_cleanup import get_metadata_from_md
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context, KeywordTableIndex, \
    SimpleKeywordTableIndex, SummaryIndex
from llama_index.llms import OpenAI

current_path = os.path.dirname(os.path.dirname(__file__))
filename = "/contents/posts/2023-04-17-topology-spread-features.md"
file_dir = current_path + filename
title = get_metadata_from_md(file_dir)["title"]


def metadata(filename):
    return {
        "file_name": filename.split("/")[-1] + "+" + title,
    }


reader = SimpleDirectoryReader(
    # input_files=[
    #     file_dir
    # ],
    input_dir=current_path + "\contents",
    required_exts=[".md"],
    recursive=True,
    file_metadata=metadata,
)
docs = reader.load_data()

service_context = ServiceContext.from_defaults(
    # embed_model="local:BAAI/bge-small-en"
    chunk_size=2000,
    llm=OpenAI(api_key="<KEY>",
               api_base="<URL>"),
)
set_global_service_context(service_context)

index = SummaryIndex.from_documents(docs)
retriever = index.as_retriever(retriever_mode="llm", choice_batch_size=3)
results = retriever.retrieve("拓扑域数")
print(results)
print(len(results))
