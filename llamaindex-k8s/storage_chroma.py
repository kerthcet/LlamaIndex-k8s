from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext,load_index_from_storage
from llama_index.vector_stores import ChromaVectorStore, FaissVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb

documents = SimpleDirectoryReader(input_files=["./data/three_pigs"]).load_data()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

#save
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context
)

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)