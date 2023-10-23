import os
import logging
import sys
from langchain.document_loaders import UnstructuredMarkdownLoader
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')
from langchain.prompts import PromptTemplate
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

#
# template = """
# 以下是有关kubernetes信息的文本，围绕title，从多个方面，尽可能详细地总结文本内容，并提出一些与内容相关的问题。
# {context}
# """
#
# PROMPT = PromptTemplate(
#     template=template,
#     input_variables=["context"],
#     )
#
# def summary(text):
#     prompt = PROMPT.format(
#                 context="\n\n".join(text),
#                 )
#     rsp = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo-16k",
#       messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     answer = rsp.get("choices")[0]["message"]["content"]
#     return answer
#
# def recursive_listdir(path, paths):
#     files = os.listdir(path)
#     for file in files:
#         file_path = os.path.join(path, file)
#         if os.path.isfile(file_path):
#             paths.append(file_path)
#         elif os.path.isdir(file_path):
#             recursive_listdir(file_path, paths)
#
# current_path = os.path.dirname(os.path.dirname(__file__))
# path_root = current_path + "/contents/posts"
# # paths = []
# # recursive_listdir(path_root, paths)
# paths = [current_path + "/contents/posts/2023-04-17-topology-spread-features.md",
#         current_path + "/contents/posts/2023-05-15-kubernetes-1-27-updates-on-speeding-up-pod-startup.md",
#         current_path + "/contents/posts/2023-05-11-nodeport-dynamic-and-static-allocation.md",
#         current_path + "/contents/posts/2023-05-02-hpa-container-resource-metric.md",
#         current_path + "/contents/posts/2023-04-21-node-log-query-alpha.md"]
#
#
# documents = []
# for path in paths:
#     loader = UnstructuredMarkdownLoader(path)
#     data = loader.load()
#     doc = data[0].page_content
#     print(doc)
#     doc_summary = summary(doc)
#     print("-----------")
#     print(doc_summary)
#     documents.append(Document(
#         page_content = doc+"\n--------------\nsummary:\n"+doc_summary,
#         metadata={
#             "filename": "/contents" + path.split("contents")[1],
#         },
#     ))
# print(documents)


import time
time_start = time.time()  # 记录开始时间


# db = Chroma.from_documents(documents, OpenAIEmbeddings(),persist_directory="./langchain_store")
# db.persist()
db = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./langchain_store")

query = "如何加速pod启动"
embedding_query = OpenAIEmbeddings().embed_query(query)
# response = db.similarity_search_by_vector(embedding_query)
response = db.similarity_search(query)
print(response[0].page_content)

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)

