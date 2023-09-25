from data_cleanup import get_metadata_from_md
import os


class TestCleanup:
    def test_demo(self):
        current_path = os.path.dirname(os.path.dirname(__file__))
        path = current_path + "/contents"
        recursive_listdir(path)


def recursive_listdir(path):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            get_metadata_from_md(file_path)
        elif os.path.isdir(file_path):
            recursive_listdir(file_path)
