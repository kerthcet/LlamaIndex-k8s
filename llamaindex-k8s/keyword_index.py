import os
from llama_index import SimpleDirectoryReader
from typing import List, Set
from data_cleanup import get_metadata_from_md

current_path = os.path.dirname(os.path.dirname(__file__))
filename = "/contents/posts/2023-04-17-topology-spread-features.md"
file_dir = current_path + filename
title = get_metadata_from_md(file_dir)["title"]


def metadata(filename):
    return {
        "file_name": filename.split("/")[-1] + "+" + title,
    }


reader = SimpleDirectoryReader(
    input_files=[
        file_dir
    ],
    required_exts=[".md"],
    recursive=True,
    file_metadata=metadata,
)

docs = reader.load_data()

from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context, KeywordTableIndex, \
    SimpleKeywordTableIndex, SummaryIndex

service_context = ServiceContext.from_defaults(
    # embed_model="local:BAAI/bge-small-en"
    chunk_size=2000,
)
set_global_service_context(service_context)

from llama_index.retrievers import KeywordTableSimpleRetriever
from llama_index.indices.keyword_table.utils import simple_extract_keywords, rake_extract_keywords


class MyKeywordIndex(SimpleKeywordTableIndex):
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""

        keyword_1 = text.split("\n\n")[0].split("+")[1]
        keyword_2 = text.split("\n\n")[1]
        _keyword_1 = simple_extract_keywords(keyword_1, self.max_keywords_per_chunk)
        _keyword_2 = simple_extract_keywords(keyword_2, self.max_keywords_per_chunk)
        # rake = rake_extract_keywords(keyword_1, self.max_keywords_per_chunk)
        # print(rake)
        keyword = _keyword_1.union(_keyword_2)
        print(keyword_1)
        print(keyword_2)
        print(keyword)
        print("--------------")
        return keyword


index = MyKeywordIndex.from_documents(docs)

# class Printkeyword(KeywordTableSimpleRetriever):
#     def _get_keywords(self, query_str: str) -> List[str]:
#         """Extract keywords."""
#         l = list(
#             simple_extract_keywords(query_str, max_keywords=self.max_keywords_per_query)
#         )
#         print("#############",l)
#         return l
# retriever = Printkeyword(index)
retriever = KeywordTableSimpleRetriever(index)
# retriever = index.as_retriever(retriever_mode='simple',max_keywords_per_query=2,num_chunks_per_query=3)
results = retriever.retrieve("滚动升级后关注 Pod 拓扑分布 参与其中")

print(results)
print(len(results))
