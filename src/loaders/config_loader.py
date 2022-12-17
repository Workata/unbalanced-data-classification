"""
    TODO if there will be enough time then rewrite this
        for pydantic BaseSettings (class) with config validation
"""
from utils import YamlReader


class ConfigLoader:

    @classmethod
    def load(cls, config_file_path: str) -> dict:
        return YamlReader.read(file_path=config_file_path)
