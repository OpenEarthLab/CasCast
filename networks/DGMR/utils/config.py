import yaml
from collections import namedtuple

class ConfigException(Exception):
    def __init__(
        self,
        message
        ):
        super().__init__(message)

def convert(
    file_dir: str=None,
    dictionary: dict=None
    ):
    if file_dir is not None:
        assert file_dir.endswith("yaml"), "the file should be .yaml format"
        with open(file_dir, "r") as f:
            dictionary = yaml.full_load(f)
    if dictionary is not None:
        for k in dictionary.keys():
            if isinstance(dictionary[k], dict):
                dictionary[k] = convert(dictionary=dictionary[k])
        return namedtuple("GenericDict", dictionary.keys())(**dictionary)

if __name__ == "__main__":
    cfg = convert(file_dir="config.yaml")
    print(cfg)