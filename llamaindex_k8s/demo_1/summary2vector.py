import os
import logging
import sys
import chromadb
from llama_index.vector_stores import ChromaVectorStore
import openai
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    Document,
    StorageContext
)
from llama_index.llms import OpenAI
from llama_index.indices.loading import load_index_from_storage

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

# rebuild storage context (load summary)
storage_context = StorageContext.from_defaults(persist_dir="../index_store/summary2vector/posts_summary_2")
doc_summary_index = load_index_from_storage(storage_context)

# process Document
documents_vec = []
summary_ids = doc_summary_index.index_struct.summary_id_to_node_ids
for summary_id in summary_ids:
    print(doc_summary_index.docstore.get_node(summary_id).text)
    summary_text = doc_summary_index.docstore.get_node(summary_id).text
    doc = Document(
        text=summary_text,
        metadata={
            "summary_id": summary_id
        }
    )
    documents_vec.append(doc)

# init chroma
db = chromadb.PersistentClient(path="../index_store/summary2vector/posts_vector_2")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# vector and persist
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    # embed_model="local:BAAI/bge-small-en"
    chunk_size=5000,
    llm=OpenAI()
)
vector_index = VectorStoreIndex.from_documents(
    documents_vec,
    show_progress=True,
    storage_context=storage_context,
    service_context=service_context
)
