from lm_eval.__main__ import cli_evaluate

import llama_swiftkv
import qwen2_swiftkv

llama_swiftkv.register_auto()
qwen2_swiftkv.register_auto()


if __name__ == "__main__":
    cli_evaluate()
