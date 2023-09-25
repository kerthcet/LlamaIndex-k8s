import codecs
from typing import Dict
import os
import logging


# TODO: @junyang
def get_metadata_from_md(path: str) -> Dict:
    """
    return:
        {"title": <title", "filename": "filename"}
    """
    title = ""
    with codecs.open(path, "rb", 'utf-8', errors='ignore') as txtfile:
        for line in txtfile:
            if line.startswith("title:"):
                print(line)
                title = line.split("\r\n")[0]
                title = title.replace("title: ", "")
                # 提取''中的内容
                if len(title.split("\'")) > 1:
                    title = title.strip("\'").strip()
                # 提取""中的内容
                elif len(title.split("\"")) > 1:
                    title = title.strip("\"").strip()
                # 提取中文双引号“”中的内容
                elif len(title.split("”")) > 1:
                    title = title.replace("“","").replace("”","").strip()
                break
    print(title)
    print(path)
    print("\n")
    # fileName = os.path.basename(path)
    return {"title": title, "filename": path}


current_path = os.path.dirname(os.path.dirname(__file__))
path = current_path + "/contents/website/concepts/architecture/_index.md"

print(get_metadata_from_md(path))

