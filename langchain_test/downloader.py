from langchain.document_loaders import UnstructuredMarkdownLoader
import os

current_path = os.path.dirname(os.path.dirname(__file__))
path = current_path + "/contents/posts/2023-04-17-topology-spread-features.md"
# path = current_path + "/contents/posts/2023-05-15-kubernetes-1-27-updates-on-speeding-up-pod-startup.md"
loader = UnstructuredMarkdownLoader(path)
data = loader.load()
print(data)
print(data[0].page_content)

from langchain.prompts import PromptTemplate
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

# template = """
# 以下是有关kubernetes信息的文本，使用500字左右，尽可能详细地总结文本内容，并提出一些与内容相关的问题。
# {context}
# """
system_message = """
你是一个kubernetes助手，你的回答基于事实，客观且准确。
请使用kubernetes相关的先验知识，总结user提供的文档内容。
并提出一些与内容相关的问题，该问题可从文档中找到答案。
从多个角度，尽可能详细，不要忽略一些细节。
"""

# system_message = '''
# 你是一个kubernetes助手，你的回答基于事实，客观且准确。
# 请使用kubernetes相关的先验知识，并根据user提供的文档，生成十组高质量的相关问题和对应的答案。
# 生成的内容尽可能详细，便于教学使用，帮助初学者理解。
# 可以生成带有shell、yaml或其他格式的答案。
#
# 例：
# 问题1：*******
# 回答1：*******
#
# 问题2：*******
# 回答2：*******
# ……
# '''

rsp = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system","content": system_message},
    {"role": "user", "content": data[0].page_content}
    ]
)
answer = rsp.get("choices")[0]["message"]["content"]
print(answer)