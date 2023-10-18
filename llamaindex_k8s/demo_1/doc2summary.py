import os
import logging
import sys
from llamaindex_k8s.data_cleanup import get_metadata_from_md
import openai
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    Document
)

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')


def recursive_listdir(path, paths):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            paths.append(file_path)
        elif os.path.isdir(file_path):
            recursive_listdir(file_path, paths)


def get_documents(paths):
    documents = []
    for path in paths:
        title = get_metadata_from_md(path)["title"]
        one_markdown = SimpleDirectoryReader(
            input_files=[path],
        ).load_data()
        doc = "title: " + title
        for i in one_markdown:
            doc = doc + "\n" + "header: " + i.text
        documents.append(Document(text=doc))
    return documents


def summary_persist(documents):
    service_context = ServiceContext.from_defaults(
        # embed_model="local:BAAI/bge-small-en"
        chunk_size=800,
        llm=OpenAI(),
    )
    summary_query = (
        "以下是有关kubernetes信息的文本，请以其中的title为主，header为辅，尽可能详细地总结文本内容，并提出一些与文本内容相关的问题，该问题可在文本中找到答案。\n请用中文回答。"
    )
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents,
        # response_synthesizer=response_synthesizer,
        summary_query=summary_query,
        show_progress=True,
        service_context=service_context
    )
    doc_summary_index.storage_context.persist("../index_store/summary2vector/posts_summary_2")


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_root = current_path + "/contents/posts"
    paths = []
    recursive_listdir(path_root, paths)
    documents = get_documents(paths)
    summary_persist(documents)
