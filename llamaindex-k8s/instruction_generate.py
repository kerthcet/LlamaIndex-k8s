import openai
import os
from data_cleanup import get_metadata_from_md
from llama_index import SimpleDirectoryReader
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

messages = []
system_message = '''
通过使用kubernetes相关的先验知识，并根据user提供的文档，帮助生成三组有关kubernetes的相关问题和对应的答案
'''
messages.append({"role":"system","content":system_message})


current_path = os.path.dirname(os.path.dirname(__file__))
path = current_path + "/contents/posts/2023-04-17-topology-spread-features.md"
title = get_metadata_from_md(path)["title"]
documents= SimpleDirectoryReader(
    input_files=[os.path.join(path)],
    # recursive=True,
).load_data()
text = "title: " + title
for i in documents:
    text = text + i.text
messages.append({"role":"user","content": text})

response=openai.ChatCompletion.create(
  # model="gpt-3.5-turbo",
    model="gpt-4",
  messages=messages
)
reply = response["choices"][0]["message"]["content"]


print("system_message: ", system_message)
# print("content: ", text)
print("generate: ", reply)