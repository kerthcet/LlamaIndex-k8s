import os

from llama_index import SimpleDirectoryReader


required_exts = [".md"]


def metadata(filename):
    return {
        "file_name": filename.split("/")[-1],
    }


current_path = os.path.dirname(os.path.dirname(__file__))

reader = SimpleDirectoryReader(
    input_files=[
        os.path.join(
            # current_path", "contents/posts/2023-04-17-topology-spread-features.md"
             current_path,"contents/posts/2022-05-25-contextual-logging/index.md"
        )
    ],
    required_exts=required_exts,
    recursive=True,
    file_metadata=metadata,
)

docs = reader.load_data()
print(f"Loaded {len(docs)} docs")
print(docs)
print(docs[0].metadata)

for i in docs:
    print(i.text)
    print("-----------------")
