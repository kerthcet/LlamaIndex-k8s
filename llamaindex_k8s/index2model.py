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
)
from IPython.display import Markdown, display
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

documents = SimpleDirectoryReader(input_files=["./data/markdown.md"], required_exts=[".md"]).load_data()
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    # embed_model="local:BAAI/bge-small-en",
    # chunk_size=64
)
set_global_service_context(service_context)
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=3)
query = "如何定义 Pod 拓扑分布中的最小域数"
retrieved_nodes = retriever.retrieve(query)
# print(retrieved_nodes)

template = """
你是一个基于事实的问答机器人, 你的回答详细且准确。
目前已知：
{context}

请回答：{question}
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

retrieved_result = []
for r in retrieved_nodes:
    retrieved_result.append(r.text)

prompt = PROMPT.format(
    context="\n-----\n".join(retrieved_result),
    question=query,
)
print(prompt)

tokenizer = AutoTokenizer.from_pretrained("/data/models/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/data/models/Baichuan2-7B-Chat", device_map="auto",
                                             torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/data/models/Baichuan2-7B-Chat")
messages = []
messages.append({"role": "user", "content": prompt})
response = model.chat(tokenizer, messages)
print(response)
