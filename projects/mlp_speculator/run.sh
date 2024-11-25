HF_HOME=/data-fast/hf-hub python -c 'import datasets; \
		datasets.load_dataset("HuggingFaceH4/ultrachat_200k"); \
		datasets.load_dataset("teknium/OpenHermes-2.5"); \
		datasets.load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")'

HF_HOME=/data-fast/hf-hub deepspeed train_speculator.py
