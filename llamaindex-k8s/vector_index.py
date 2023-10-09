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

documents = SimpleDirectoryReader(input_files=["./data/markdown.md", "./data/markdown_1.md", "./data/markdown_2.md"],
                                  required_exts=[".md"]).load_data()

from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

# embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-zh")
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    # embed_model="local:BAAI/bge-small-en",
    # chunk_size=64
)
set_global_service_context(service_context)
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=4)
# retrieved_nodes = retriever.retrieve("how many little pigs are there in the story")
retrieved_nodes = retriever.retrieve("在拓扑分布中如何定义最小域数")
print(retrieved_nodes)
