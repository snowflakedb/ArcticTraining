# Arctic Embed In Arctic Training

This directory implements the next generation of Arctic Embed training built on top of the Arctic Training package.

## Installation

To make importing and using Arctic Embed code easier, the codebase is organized as a minimalistic Python package. Assuming you have already installed Arctic Training, you just need to run `pip install -e /path/to/ArcticTraining/projects/arctic_embed`. From there, you should be able to run all the examples, `import arctic_embed`, and otherwise be off to the races running with Arctic Embed.

There is currently no plan on publishing Arctic Embed as its own package on PyPI, but we find that leveraging the Python packaging system makes structuring the code for easy import much simpler than the alternatives, and thus we have chosen to piggyback on the packaging system for this express purpose.

If you'd prefer not install the package, you can also simply set your `PYTHONPATH` environment variable manually, e.g. `PYTHONPATH=/path/to/ArcticTraining/projects/arctic_embed/src python /path/to/ArcticTraining/projects/arctic_embed/examples/<some_example.py>`.

## Running Examples

The examples directory contains Python scripts you can run directly. These examples do not leverage the YAML-driven Arctic Training launcher CLI. In order to smoothly launch a distributed multi-GPU job, use the `deepspeed` CLI in place of directly calling the `python` CLI, e.g. `deepspeed examples/finetune_e5_base_v2.py`. If you are running on a machine which only has one GPU, or you would like to force single-GPU execution for debugging, then calling the Python interpreter directly is the way to go, e.g. `python examples/finetune_e5_base_v2.py`.

## Data Loading And Training Data Format

Arctic Embed has its own destinctive data batch format, data loading implementation, and on-disk data storage conventions. While this pushes a substantial amount of complexity into the data preprocessing step, it makes the training code itself straightforward and highly readable. To make it easy to work in this paradigm, Arctic Embed offers a set of tools for preprocessing data to feed into Arctic Embed training.

### Arctic Embed's Data Format

The Arctic Embed data format is designed to be incredibly straightforward while also allowing for complex relations between queries and documents to be encoded. A training dataset is a single directory containing a number of subdirectories, one for each batch of the pre-batched data.

``` text
example_dataset/
├── batch_00000000
│   ├── documents.parquet
│   ├── queries.parquet
│   └── relations.parquet
├── batch_00000001
│   ├── documents.parquet
│   ├── queries.parquet
│   └── relations.parquet
├── batch_00000002
│   ├── documents.parquet
│   ├── queries.parquet
│   └── relations.parquet
├── batch_00000003
│   ├── documents.parquet
│   ├── queries.parquet
│   └── relations.parquet
...
```

Each batch directory contains three Parquet files, one for queries, one for documents, and one for the relations between the queries and documents. The Arrow schemas for these files is as follows:

```
# queries.parquet
BATCH_QUERY_ID: uint64
QUERY_TOKEN_ID_LIST: large_list<element: uint16>
  child 0, element: uint16

# documents.parquet
BATCH_DOCUMENT_ID: uint64
DOCUMENT_TOKEN_ID_LIST: large_list<element: uint16>
  child 0, element: uint16

# relations.parquet
BATCH_QUERY_ID: uint64
BATCH_DOCUMENT_ID: uint64
RELEVANCE: int8
```

What information should these columns hold?

- BATCH_QUERY_ID and BATCH_DOCUMENT_ID serve to define the relationship between the queries, documents, and relations.
  - Each ID must identify a unique item within the batch, but ids need not be unique across batches.
  - The `queries.parquet` and `documents.parquet` files must not contain duplicate ids -- data ought to be deduped at batch construction
- *_TOKEN_ID_LIST fields contain the integer ids of pre-tokenized texts.
  - Any instruction prefixes should be pre-tokenized and included in these fields.
  - No padding tokens are needed to be included in these fields (that would be a waste of space)
- RELEVANCE defines the relationship between a query and a document.
  - A negative value indicates knowledge of no relevance of the document to the query.
  - A positive value indicates knowledge of relevance.
  - A zero value indicates no known label.
  - All query-document pairs in the batch which do not have an entry in `relations.parquet` are implied to have a relevance value of 0.
  - During training, in batch negative training replaces all zero values with a value of -1.


The integer type of the query and document ids, token id lists, and relevance scores is flexible so long as they are consistent with one another. Due to the possibility of the relations table being quite long, the format requires integer IDs to avoid cases of accidentally wasting a lot of storage space by storing many copies of long string ids or the like. If you have string ids, hashing is an easy way to get integer ids from them, e.g. via the `hash_array` function from Pandas (`from pandas.util import hash_array`).

See the content of `pretokenize.py` for tools and examples for converting from common data formats like tables of query-doc pairs (e.g. weakly supervised pretraining data) to the Arctic Embed data format.

### Data Loading And Usage

To avoid bottlenecks and address the unique complexities of distributed contrastive dataloading (namely the need for pairwise relevance labels to span across examples which are sharded to different workers), Arctic Embed brings its own custom dataloading.

- The `ContrastivePretokenizedDataFactory` in `contrastive_dataloader.py` provides access to a `ContrastiveLearningBatchDataset` (implemented in `core/pretokenized_batch_loader.py`) which can be streamed from local storage or an object storage like AWS S3.
- Data is expected to be tokenized and batched ahead of training as part of this format.
  - The pre-tokenization helps avoid slowdowns that arise when tokenizing on the fly
  - Pre-tokenized data can also be streamed (e.g. from S3), mitigating the need to pre-download large datasets during large-scale training
  - Since there can be up to N^2 contrastive relevance labels for any group of N samples, it is infeasible to attempt to compute and store relevance judgements across an entire largescale dataset, hence the need for pre-batching
- To enable more flexible training runs and minimize the need to re-batch data repeatedly, each batch of pre-batched data can be split into smaller batches during dataloading simply by setting the `ContrastivePretokenizedDataConfig.split_factor` property, e.g. a dataset pre-tokenized to batches of 64,000 examples can be data-loaded in batches of 8,000 examples by setting the `split_factor` to 8.


**See the `data_prep` directory for more documentation, tooling, and examples regarding the Arctic Embed data format.**


### Rationale

#### Why Pre-Tokenization?

In order to ensure fast dataloading throughput, Arctic Embed's data loading operates on pre-tokenized data. Thus this directory contains tools for pre-tokenizing your data into the appropriate format.

#### Why Pre-Batching?

Arctic Embed is ultimately a contrastive learning toolkit, which means that the labels used to supervise learning are, in the general case, representable as a Q-by-D matrix of similarity annotations between each query and document in a batch of Q queries and D documents. For large datasets (e.g. millions of queries and documents), it is generally infeasible to exhaustively annotate all relationships between queries and documents and construct this matrix, however (and even if we could get exhaustive pairwise relation labels, the storage cost of storing these relation labels could quickly become cumbersome).

Many other contrastive learning projects solve this problem by treating large-scale weakly-supervised pretraining and smaller-scale fine-tuning as separate, special cases for data handling and adopt specific data formats and data loading logic for each. Arctic Embed, by contrast, takes the approach of requiring training data to be pre-batched in its storage format. This allows each batch to define arbitrary relationships between all queries and documents within the batch while also sidestepping the problem of exhaustive annotation by not requiring any relevance to be labeled for items outside of the batch.

The benefit from this design decision is flexibility with regards to the types of training you can dream up -- the Arctic Training codebase can handle whatever relevance patterns you want to throw at it, from query-doc pair relevance patterns to hard-negative mined negative examples. The cost of this design decision is that we need to pre-batch our data. It is sometimes a bit tricky to go from data in one format (e.g. query-doc pairs) into batches with arbitrary query-doc labels, but this directory contains the tools needed to make the common cases easy!


## Modeling

Though Arctic Training comes with Huggingface model support, wrapping a model into a bi-encoder architecture, the built-in HuggingFace `ModelFactory` loads `*ForCausalLM` models, which are not suitable for embedding applications. Arctic Embed thus introduces a small custom `ModelFactory` implementation and subclasses the `HFCheckpointEngine`. If you would like to work with models that do not follow the Huggingface `transformers` paradigm, you may do so by implementing your own model factory. You will likely also need to implement your own checkpointing engine.

## Training

Arctic Embed training is carried out in a custom subclass of the Arctic Training `Trainer` class.

## Tests

You can run the Arctic Embed unit tests like this:

``` shell
pytest -o pythonpath=projects/arctic_embed projects/arctic_embed/tests
```
