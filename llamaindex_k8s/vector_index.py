# linux
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    SimpleKeywordTableIndex
)
from IPython.display import Markdown, display
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
import os
current_path = os.path.dirname(os.path.dirname(__file__))
documents= SimpleDirectoryReader(
    input_files=[
        os.path.join(
            current_path, "contents/posts/2023-04-17-topology-spread-features.md"
             # current_path,"contents/posts/2022-05-25-contextual-logging/index.md"
            # current_path,"llamaindex-k8s/three_pigs"
        )
    ],
    # recursive=True,
).load_data()
# documents = SimpleDirectoryReader(input_files=["./data/markdown.md", "./data/markdown_1.md", "./data/markdown_2.md"],
#                                   required_exts=[".md"]).load_data()
print(documents[0].doc_id)

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

# embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-zh")
# embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
service_context = ServiceContext.from_defaults(
    # embed_model=embed_model,
    # embed_model="local:BAAI/bge-small-en",
    chunk_size=3000
)
set_global_service_context(service_context)
index = VectorStoreIndex.from_documents(documents)
print(index.index_struct)
print(index.docstore.docs)
# retriever = index.as_retriever(similarity_top_k=4)
# # retrieved_nodes = retriever.retrieve("how many little pigs are there in the story")
# retrieved_nodes = retriever.retrieve("pigs")
# print(retrieved_nodes)
