import llama_swiftkv
from lm_eval.__main__ import cli_evaluate

llama_swiftkv.register_auto()


if __name__ == "__main__":
    cli_evaluate()
