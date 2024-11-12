HF_HOME=/data-fast/hf-hub python -c 'import datasets; \
		datasets.load_dataset("HuggingFaceH4/ultrachat_200k")'

HF_HOME=/data-fast/hf-hub deepspeed train.py
