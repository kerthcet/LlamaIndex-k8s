import logging
import sys
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')
from llama_index.embeddings import OpenAIEmbedding
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
from data_cleanup import get_metadata_from_md

total_doc = []
def all_listdir(path):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            title = get_metadata_from_md(file_path)["title"]
            documents = SimpleDirectoryReader(
                input_files=[file_path],
            ).load_data()
            for doc in documents:
                # doc.metadata['text'] = doc.text
                doc.text = title + "\n\n" + doc.text.split("\n\n")[0]
                total_doc.append(doc)
        elif os.path.isdir(file_path):
            all_listdir(file_path)

path = os.path.dirname(os.path.dirname(__file__)) + "/contents"

all_listdir(path)

service_context = ServiceContext.from_defaults(
    embed_model = OpenAIEmbedding(),
    # embed_model="local:BAAI/bge-small-en",
    # chunk_size=64
)
set_global_service_context(service_context)
index = VectorStoreIndex.from_documents(total_doc)
retriever = index.as_retriever(similarity_top_k=4)
retrieved_nodes = retriever.retrieve("在拓扑分布中如何定义最小域数")
print(retrieved_nodes)