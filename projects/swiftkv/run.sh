export HF_TOKEN=$1

HF_HOME=/data-fast/hf-hub python -c 'import tempfile; import os; \
		tempfile.tempdir = "/data-fast/tmp"; \
		import datasets; \
		datasets.load_dataset("HuggingFaceH4/ultrachat_200k"); \
		datasets.load_dataset("meta-math/MetaMathQA"); \
		datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K"); \
		datasets.load_dataset("lmsys/lmsys-chat-1m", token=os.environ.get("HF_TOKEN")); \
		datasets.load_dataset("Open-Orca/SlimOrca");'

HF_HOME=/data-fast/hf-hub deepspeed train.py

sleep 5
deepspeed /tmp/run.py
