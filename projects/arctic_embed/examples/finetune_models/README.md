# Fine-Tuning Embedding Models Using High-Quality Hard-Negative Mining

One of the key ingredients behind high-quality text embedding models is high-quality finetuning data. To create this data, it is important not just to cleanly annotate meaningful query-document relevance judgements, but also to select so-called *hard negative* examples to contrast against these positive examples. In this example, we use a teacher embedding model to mine for hard negative examples across a handful of common retrieval fine-tuning datasets, then feed this data into Arctic Embed training to fine-tune an improved version of the classic E5 model.

## Quickstart

To dive in faster, you can grab an already-preprocessed copy of the training data from Huggingface. There are several ways to [download datasets from Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/download), but given the large number of files in the dataset, it may be helpful to use the `git`-style approach.

``` shell
# First, ensure you have installed git-lfs (see `https://git-lfs.com/` for documentation).
git lfs install

# Clone the full precomputed data repository without large files as our `data/` directory.
mv ./data/.gitignore ./data.gitignore
rmdir ./data
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf.co/datasets/Snowflake/arctic-embed-ft-v1.git ./data
mv ./data.gitignore ./data/.gitignore

# Ensure we have all the files you need for training downloaded from LFS.
cd arctic-embed-ft-v1/
git lfs pull --include="combined/pretokenized/example_dot95/,eval/"

# Optional: Download more large files (e.g. everything but the very large precomputed embeddings).
git lfs pull --exclude="*embeddings*"
```

Next, adjust the `finetune_e5_base_unsupervised.py` script to meet your needs:
- Adjust the data path if you have put your data in a different place, adjust the checkpoint path if you'd like to save checkpoints to a different location, etc.
- Disable Weights & Biases logging if you do not wish to use it (simply set `wandb_conf=None`)
- Adjust other config and hyperparameters like logging level, loss temperature, learning rate schedule, etc. as you see fit

Then simply launch the training script using the `deepspeed` CLI: `deepspeed finetune_e5_base_unsupervised.py`!

## Data Processing For This Example

Although you can grab the preprocessed training data from Huggingface, this example also contains all the code needed to reproduce (and optionally customize) a high-quality fine-tuning dataset for retrieval from scratch on your own machine. Indeed, in addition to helpful tools for pretokenizing and batching data into the correct format, Arctic Embed also includes helpful tooling for data processing applications like hard-negative mining and consistency filtering in `arctic_embed.data_processing`.

### Prerequisites

First, we need to install Arctic Training, Arctic Embed, and the `fire` utility (which is used to implement the CLI interface in the data preparation scripts).

```shell
pip install fire -e path/to/ArcticTraining -e path/to/ArcticTraining/projects/arctic_embed
```

NOTE: You should also have at least one GPU and 150GB of free space on disk, since we will be computing and writing millions of uncompressed embeddings (4KiB each) to parquet files as part of the hard negative mining process. Multiple GPUs are recommended to run embedding faster.

### Step 0: Download Raw Data

We first download the raw data. The included scripts download the data from various sources (both original datasets and post-processed datasets hosted on Huggingface) into a standardized format (`queries.parquet`, `documents.parquet`, and `labels.parquet`).

The table schemas as as follows:
- `queries.parquet` contains all unique queries paired with 64bit ids
  - QUERY_ID: uint64
  - QUERY_TEXT: string
- `documents.parquet` contains all unique documents paired with 64bit ids
  - DOCUMENT_ID: uint64
  - DOCUMENT_TEXT: string
- `labels.parquet` identifies all pairs of query-document relevance labels via the ids of the queries and documents
  - QUERY_ID: uint64
  - DOCUMENT_ID: uint64

```shell
# Change the example path below to wherever you have the code on your machine.
cd path/to/arctic_embed/examples/finetune_models

# Run download scripts in sequence.
python data_download_scripts/download_fever.py
python data_download_scripts/download_hotpotqa_from_bge_data.py
python data_download_scripts/download_msmarco.py
python data_download_scripts/download_nq.py
python data_download_scripts/download_stackexchange.py
```

### Step 1: Embed All Queries And Documents

In order to find hard negatives, we will first embed all of the queries and documents in our datasets to leverage a teacher embedding model's sense of which negative examples appear somewhat semantically similar to each query.

Heads up! Even on a 8xH100 machine, this step may take an hour or two.

```shell
# Change the example path below to wherever you have the code on your machine.
cd path/to/arctic_embed/examples/finetune_models

for dataset in fever hotpotqa msmarco nq stackexchange; do
  python data_processing_scripts/stage_1_embed.py --query_pq_path="data/$dataset/text/queries.parquet" --doc_pq_path="data/$dataset/text/documents.parquet" --out_dir="data/$dataset/embeddings/arcticembed2l/" --batch_size=256
done
```

The output of this task is a pair of directories of parquet files containing the embedding vectors for the text content of `queries.parquet` and `documents.parquet`:

- `query_embeddings/part_*.parquet`
  - ID: uint64
  - VECTOR: fixed_size_list<element: float>[*fixed vector size, e.g. 1024*]
    - child 0, element: float
- `document_embeddings/part_*.parquet`
  - ID: uint64
  - VECTOR: fixed_size_list<element: float>[*fixed vector size, e.g. 1024*]
    - child 0, element: float

### Step 2: Dense Retrieval

In order to actually identify somewhat-semantically-similar negative documents for each query, we need to use our embedding vectors to perform retrieval of the most similar documents for each query via the embedding vectors we computed in step 1. To ensure we pull a fairly large number of candidate negatives that aren't too hard that we worry they should also be classified as positive examples, we'll use a deep top-k retrieval depth of 1000.

This code path will leverage a GPU if present to run batch dense retrieval much faster. Even with large datasets containing millions of vectors (which we swap into RAM and GPU memory in chunks to avoid overflowing hardware capacity in an unoptimized manner), this step should take well under an hour (perhaps 20 minutes or so) as long as you have a GPU for it to leverage.


```shell
# Change the example path below to wherever you have the code on your machine.
cd path/to/arctic_embed/examples/finetune_models

for dataset in fever hotpotqa msmarco nq stackexchange; do
  python data_processing_scripts/stage_2_score_and_retrieve.py --query_pq_dir="data/$dataset/embeddings/arcticembed2l/query_embeddings" --doc_pq_dir="data/$dataset/embeddings/arcticembed2l/document_embeddings" --out_path="data/$dataset/embeddings/arcticembed2l/retrieval_result_depth_1000.parquet" --retrieval_depth=1000
done
```

This script will creates a new file containing the scores for the top-k most relevant documents for each query. The file has the following schema:
- `retrieval_result_depth_1000.parquet`
  - QUERY_ID: uint64
  - DOCUMENT_ID: uint64
  - SCORE: float

### Step 3: Select Negatives From Retrieval Results

Because many of our datasets may contain documents which are also semantically highly relevant to a query despite not being annotated as "positives", we must exercise some care in selecting which of our rated-as-relevant examples we should use as negatives. In this example, we mine a fixed number of the highest-scoring examples whose relevance scores are below a fixed fraction of the relevance score assigned to the labeled-as-positive document. For example, let us consider  mining a fixed 10 negatives per query with a 90% false-positive threshold. In this case, if a positive document d_pos labeled for query q_i has a score of 0.8000, then the 10 highest scoring documents whose relevance score to query q_i is less than 0.7200 will be selected as hard negatives.

We will mine 10 negatives at threshold 0.95, but feel free to adjust these parameters and re-run the code yourself! Since this step only performs lightweight comparisons and filters on the retrieval results, it runs quickly and requires no GPU compute (1-2min).

```shell
# Change the example path below to wherever you have the code on your machine.
cd path/to/arctic_embed/examples/finetune_models

for dataset in fever hotpotqa msmarco nq stackexchange; do
  python data_processing_scripts/stage_3_mine.py --relevance_score_pq_path="data/$dataset/embeddings/arcticembed2l/retrieval_result_depth_1000.parquet" --labels_pq_path="data/$dataset/text/labels.parquet" --out_path="data/$dataset/mined/arcticembed2l_example_dot95.parquet" --negative_samples_per_query=10 --max_positives_per_query=3 --max_negative_to_positive_relevance_threshold="0.95"
done
```
This script will creates a new file containing relevance judgements for both the originally-labeled positive example and K mined hard-negative examples per query. Relevance will be -1 for a negative example and 1 for a positive example. The file has the following schema:
- `arcticembed2l_example_dot95.parquet`
  - QUERY_ID: uint64
  - DOCUMENT_ID: uint64
  - RELEVANCE: int8

### Step 4: Combining The Data

Since we have carefully selected negative examples via hard-negative mining, we do not need to use in-batch negative examples when fine-tuning on our prepared data. As such, it simplest to merge all of our examples together ino a single dataset before tokenization and batching.

``` shell
# Change the example path below to wherever you have the code on your machine.
cd path/to/arctic_embed/examples/finetune_models

python data_processing_scripts/stage_4_combine.py \
  --in_dirs="['data/fever','data/hotpotqa','data/msmarco','data/nq','data/stackexchange']" \
  --out_dir="data/combined" \
  --sub_dirs="['text/queries.parquet','text/documents.parquet','text/labels.parquet','mined/arcticembed2l_example_dot95.parquet']"
```

### Step 5: Tokenize And Batch

Arctic Embed uses pre-batched, pre-tokenized data. To get our data into this format, we execute pre-tokenization. This is a CPU-heavy process which may take 10-20 minutes. We shall elect to allow up to 3 positive documents per query and require exactly 10 negative documents.

```shell
# Change the example path below to wherever you have the code on your machine.
cd path/to/arctic_embed/examples/finetune_models

# Write out a configuration file for the pre-tokenization and pre-batching.
# NOTE: Adjust the paths in the config to the paths on your machine!
echo '{
  "tokenizer_id": "intfloat/e5-base-v2",
  "query_loc": "/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/combined/text/queries.parquet",
  "document_loc": "/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/combined/text/documents.parquet",
  "labels_loc": "/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/combined/mined/arcticembed2l_example_dot95.parquet",
  "output_loc": "/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/combined/pretokenized/example_dot95",
  "query_prefix": "query: ",
  "document_prefix": "passage: ",
  "random_seed": 0,
  "num_query_uses": 1,
  "num_negatives_per_query_per_batch": 10,
  "max_pos_per_query": 3,
  "queries_per_batch": 256,
  "max_seq_length": 1024,
  "pre_truncate_max_chars_per_token": 8,
  "in_query_id_col": "QUERY_ID",
  "in_document_id_col": "DOCUMENT_ID",
  "in_query_text_col": "QUERY_TEXT",
  "in_document_text_col": "DOCUMENT_TEXT",
  "in_label_relation_col": "RELEVANCE",
  "out_query_id_col": "BATCH_QUERY_ID",
  "out_document_id_col": "BATCH_DOCUMENT_ID",
  "out_query_tokens_col": "QUERY_TOKEN_ID_LIST",
  "out_document_tokens_col": "DOCUMENT_TOKEN_ID_LIST",
  "out_relevance_col": "RELEVANCE"
}' > /tmp/stage5_config_combined.json

# Run the tokenization and batching from this config.
python data_processing_scripts/stage_5_tokenize_and_batch.py /tmp/stage5_config_combined.json

# Optionally remove our config.
rm /tmp/stage5_config_combined.json
```

You may also batch the individual datasets as well, e.g. for ablation studies that vary the training data source (this will take an additional 10-20 minutes or so).

``` shell
cd path/to/arctic_embed/examples/finetune_models

# NOTE: This loop will run all tasks in parallel.
# Omit the `&` and `wait` to run in sequence (slower but with nicer terminal output).
for ds in fever hotpotqa msmarco nq stackexchange; do
    (echo "{
        \"tokenizer_id\": \"intfloat/e5-base-v2\",
        \"query_loc\": \"/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/$ds/text/queries.parquet\",
        \"document_loc\": \"/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/$ds/text/documents.parquet\",
        \"labels_loc\": \"/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/$ds/mined/arcticembed2l_example_dot95.parquet\",
        \"output_loc\": \"/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/$ds/pretokenized/example_dot95\",
        \"query_prefix\": \"query: \",
        \"document_prefix\": \"passage: \",
        \"random_seed\": 0,
        \"num_query_uses\": 1,
        \"num_negatives_per_query_per_batch\": 10,
        \"max_pos_per_query\": 3,
        \"queries_per_batch\": 256,
        \"max_seq_length\": 1024,
        \"pre_truncate_max_chars_per_token\": 8,
        \"in_query_id_col\": \"QUERY_ID\",
        \"in_document_id_col\": \"DOCUMENT_ID\",
        \"in_query_text_col\": \"QUERY_TEXT\",
        \"in_document_text_col\": \"DOCUMENT_TEXT\",
        \"in_label_relation_col\": \"RELEVANCE\",
        \"out_query_id_col\": \"BATCH_QUERY_ID\",
        \"out_document_id_col\": \"BATCH_DOCUMENT_ID\",
        \"out_query_tokens_col\": \"QUERY_TOKEN_ID_LIST\",
        \"out_document_tokens_col\": \"DOCUMENT_TOKEN_ID_LIST\",
        \"out_relevance_col\": \"RELEVANCE\"
    }" > /tmp/stage5_config_$ds.json;
    python data_processing_scripts/stage_5_tokenize_and_batch.py /tmp/stage5_config_$ds.json;
    rm /tmp/stage5_config_$ds.json;) &
done
wait
```

### Step 6: Eval Splits

In order to get more information during training about model performance, we can select a few batches of data from each source dataset to serve as evaluation sets. In this example we re-use data which is also used as part of the training data, so the evaluation is not out-of-sample, but it does serve as a nice low-noise way to measure error across each data source.

``` shell
cd path/to/arctic_embed/examples/finetune_models

python data_processing_scripts/stage_6_get_dev_splits.py
```

### Aside: Storing hard-negative mined data on S3, HuggingFace, etc.

The Arctic Embed dataloading code or this data preparation code is built around the generic abstraction of a *filesystem* (e.g. local filesystem, or a remote "filesystem-like" abstraction like AWS S3). This makes it very straightforward to move data around and stream it from remote system. For example, you can use the AWS CLI to copy data to S3 and even data-load it directly from S3 without pre-downloading it during training (`aws s3 sync /path/to/msmarco/pretokenized/example s3://my-bucket/example-hard-negative-msmarco` to upload and `aws s3 sync s3://my-bucket/example-hard-negative-msmarco /path/to/example-hard-negative-msmarco` to download).

Some domain-specific abstraction, including HuggingFace Datasets, can also be used as a backend for file storage and sharing. If you would like to leverage HF Datasets for storage/sharing, here are some steps which may be helpful to follow.

Upload steps:

1. Create a new empty dataset using the HuggingFace web application ([link](https://huggingface.co/new-dataset))
2. Create a huggingface access token with write permissions to this new dataset ([link to access tokens settings](https://huggingface.co/settings/tokens)) (see reference below for docs)
3. In your terminal, install the huggingface-cli cli (see reference below for docs)
4. Log in to the huggingface cli using a token with write persmissions via `huggingface-cli login`
5. Push the dataset using the cli, e.g. `huggingface-cli upload-large-folder Snowflake/ft-dataset-example --repo-type=dataset --num-workers=4 /path/to/data/ft-dataset-example/`

References:

1. [HuggingFace access token docs](https://huggingface.co/docs/hub/en/security-tokens)
2. [HuggingFace CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)


Downloading:

- Like with uploading, you will need to install the HuggingFace CLI
- If the dataset is private, you will need to log into the CLI with an appropriately authorized token
- You can use the CLI to download a dataset to a custom folower like this: `huggingface-cli download --repo-type=dataset --local-dir=/path/to/data_local Snowflake/ft-dataset-example`
- You can also use git and git LFS, as shown in the quickstart instructions above
