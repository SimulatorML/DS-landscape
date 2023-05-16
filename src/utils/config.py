""" Configuration files helper """
import os
import jsonpickle

def dump(obj: object, config_path : str) -> None:
    """Save file with default config"""
    jsonpickle.set_preferred_backend('json')
    jsonpickle.set_encoder_options('json', ensure_ascii=False)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(obj, indent=4))

def load(config_path : str) -> object:
    """load config from file
    exsabple of using: 
        self.config : ParserConfig = config.load(config_path) or ParserConfig()
        """

    jsonpickle.set_preferred_backend('json')
    jsonpickle.set_encoder_options('json', ensure_ascii=False)

    if config_path is not None and os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return jsonpickle.decode(f.read())
    else:
        return None
