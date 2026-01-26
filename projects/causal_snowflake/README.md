# Causal Training with Snowflake Data Sources

This project demonstrates causal language model training using data stored in Snowflake. It includes examples of all three Snowflake data source modes supported by Arctic Training.

## Snowflake Data Source

The unified `snowflake` data source type supports three mutually exclusive modes:

| Mode | Config Key | Description |
|------|------------|-------------|
| SQL Query | `sql` | Execute arbitrary SQL queries against Snowflake |
| Table | `table_name` | Load data directly from a Snowflake table |
| Dataset | `dataset_uri` | Load data from a versioned Snowflake Dataset |

**Note:** Exactly one of `sql`, `table_name`, or `dataset_uri` must be specified.

## Prerequisites

### 1. Install Dependencies

Install Arctic Training with Snowflake support:

```bash
pip install 'arctic_training[snowflake]'
```

### 2. Configure Snowflake Credentials

The Snowflake data sources use `Session.builder.getOrCreate()` which supports multiple authentication methods:

**Option A: Environment Variables**
```bash
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_USER="your_username"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_WAREHOUSE="your_warehouse"  # optional
export SNOWFLAKE_DATABASE="your_database"    # optional
export SNOWFLAKE_SCHEMA="your_schema"        # optional
```

**Option B: Connections Config File**

Create `~/.snowflake/connections.toml`:
```toml
[default]
account = "your_account"
user = "your_username"
password = "your_password"
warehouse = "your_warehouse"
```

You can also specify a non-default connection:
```bash
export SNOWFLAKE_DEFAULT_CONNECTION_NAME="my_connection"
```

## Setting Up Snowflake with Training Data

Before running training, you need to populate your Snowflake account with the training data.

### Run the Set Up Script

```bash
cd projects/causal_snowflake
python setup_snowflake.py
```

This will:
1. Download the `stas/gutenberg-100` dataset from HuggingFace
2. Create the `ARCTIC_TRAINING.CAUSAL_DEMO` database and schema
3. Upload the data to a `GUTENBERG_100` table
4. Create a versioned `GUTENBERG_DATASET` Snowflake Dataset

### Set Up Script Options

```bash
python setup_snowflake.py --help

Options:
  --database TEXT        Snowflake database name (default: ARCTIC_TRAINING)
  --schema TEXT          Snowflake schema name (default: CAUSAL_DEMO)
  --table-name TEXT      Snowflake table name (default: GUTENBERG_100)
  --dataset-name TEXT    Snowflake Dataset name (default: GUTENBERG_DATASET)
  --dataset-version TEXT Snowflake Dataset version (default: v1)
  --hf-dataset TEXT      HuggingFace dataset to download (default: stas/gutenberg-100)
  --sample-count INT     Number of samples to upload (default: 100)
  --drop-existing        Drop existing table and dataset if they exist
```

### Expected Snowflake Resources

After set up, you should have:

| Resource | Full Name |
|----------|-----------|
| Database | `ARCTIC_TRAINING` |
| Schema | `ARCTIC_TRAINING.CAUSAL_DEMO` |
| Table | `ARCTIC_TRAINING.CAUSAL_DEMO.GUTENBERG_100` |
| Dataset | `snow://dataset/ARCTIC_TRAINING.CAUSAL_DEMO.GUTENBERG_DATASET/versions/v1` |

## Running Training

### Using SQL Query Mode

Load data via a custom SQL query:

```bash
arctic_training run-causal-snowflake-sql.yml
```

Config snippet:
```yaml
data:
  sources:
    - type: snowflake
      sql: "SELECT TEXT FROM ARCTIC_TRAINING.CAUSAL_DEMO.GUTENBERG_100"
      column_mapping: {"TEXT": "text"}
```

### Using Table Name Mode

Load data directly from a table (auto-generates `SELECT * FROM table_name`):

```bash
arctic_training run-causal-snowflake-table.yml
```

Config snippet:
```yaml
data:
  sources:
    - type: snowflake
      table_name: ARCTIC_TRAINING.CAUSAL_DEMO.GUTENBERG_100
      column_mapping: {"TEXT": "text"}
```

### Using Dataset URI Mode

Load data from a versioned Snowflake Dataset:

```bash
arctic_training run-causal-snowflake-dataset.yml
```

Config snippet:
```yaml
data:
  sources:
    - type: snowflake
      dataset_uri: "snow://dataset/ARCTIC_TRAINING.CAUSAL_DEMO.GUTENBERG_DATASET/versions/v1"
      column_mapping: {"TEXT": "text"}
```

## Configuration Options

All modes support these common options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `column_mapping` | dict | `{}` | Rename columns (e.g., `{"SRC": "dst"}`) |
| `limit` | int | None | Maximum rows to load |
| `batch_size` | int | 1024 | Batch size for data retrieval |

### Mode-Specific Options

Exactly one of the following must be specified:

| Option | Description |
|--------|-------------|
| `sql` | SQL query to execute |
| `table_name` | Table reference as `[[db.]schema.]table_name` |
| `dataset_uri` | Dataset URI as `snow://dataset/<name>/versions/<version>` |

## Troubleshooting

### Connection Issues

If you get authentication errors:
1. Verify your credentials are correct
2. Check that your account identifier is in the correct format (e.g., `orgname-accountname`)
3. Ensure your IP is allowlisted if network policies are enabled

### Missing snowflake-ml-python

If you see `ImportError: snowflake-ml-python is required`:
```bash
pip install 'arctic_training[snowflake]'
```

### Table/Dataset Not Found

Ensure you've run the set up script first:
```bash
python setup_snowflake.py
```

Or verify the resources exist in Snowflake:
```sql
SHOW TABLES IN ARCTIC_TRAINING.CAUSAL_DEMO;
```
