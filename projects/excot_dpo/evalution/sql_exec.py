# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import sqlglot
import yaml
from datasets import load_dataset


@dataclass
class EvalConfig:
    task: str = ""
    train_dir: str = ""
    train_db_folder_path: str = ""
    tables_json_path: str = ""
    train_question_file_path: str = ""
    dev_dir: str = ""
    dev_db_folder_path: str = ""
    dev_tables_json_path: str = ""
    dev_question_file_path: str = ""
    test_dir: str = ""
    test_db_folder_path: str = ""
    test_tables_json_path: str = ""
    test_question_file_path: str = ""
    cache_dir: str = ""

    def load_yaml(self, yaml_path: str):
        # Load the yaml file to initialize the configuration fields
        with open(yaml_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        self.cache_dir = yaml_data["cache_dir"]
        if "bird" in yaml_data:
            self.task = "bird"
            # Now update the attributes from the loaded YAML data
            self.train_dir = yaml_data["bird"]["train"]["data_dir"]
            self.train_db_folder_path = Path(os.path.join(self.train_dir, yaml_data["bird"]["train"]["db_folder"]))
            self.tables_json_path = Path(os.path.join(self.train_dir, yaml_data["bird"]["train"]["tables_json_name"]))
            self.train_question_file_path = Path(
                os.path.join(self.train_dir, yaml_data["bird"]["train"]["question_file"])
            )

            self.dev_dir = yaml_data["bird"]["dev"]["data_dir"]
            self.dev_db_folder_path = Path(os.path.join(self.dev_dir, yaml_data["bird"]["dev"]["db_folder"]))
            self.dev_tables_json_path = Path(os.path.join(self.dev_dir, yaml_data["bird"]["dev"]["tables_json_name"]))
            self.dev_question_file_path = Path(os.path.join(self.dev_dir, yaml_data["bird"]["dev"]["question_file"]))

            self.test_dir = yaml_data["bird"]["test"]["data_dir"]
            self.test_db_folder_path = Path(os.path.join(self.test_dir, yaml_data["bird"]["test"]["db_folder"]))
            self.test_tables_json_path = Path(
                os.path.join(self.test_dir, yaml_data["bird"]["test"]["tables_json_name"])
            )
            self.test_question_file_path = Path(
                os.path.join(self.test_dir, yaml_data["bird"]["test"]["question_file"])
            )
        elif "spider" in yaml_data:
            self.task = "spider"
            self.train_dir = yaml_data["spider"]["train"]["data_dir"]
            self.train_db_folder_path = Path(os.path.join(self.train_dir, yaml_data["spider"]["train"]["db_folder"]))
            self.tables_json_path = Path(
                os.path.join(self.train_dir, yaml_data["spider"]["train"]["tables_json_name"])
            )
            self.train_question_file_path = Path(
                os.path.join(self.train_dir, yaml_data["spider"]["train"]["question_set_file"])
            )
            self.train_question_other_file_path = Path(
                os.path.join(self.train_dir, yaml_data["spider"]["train"]["question_other_file"])
            )

            self.dev_dir = yaml_data["spider"]["dev"]["data_dir"]
            self.dev_db_folder_path = Path(os.path.join(self.dev_dir, yaml_data["spider"]["dev"]["db_folder"]))
            self.dev_tables_json_path = Path(
                os.path.join(self.dev_dir, yaml_data["spider"]["dev"]["tables_json_name"])
            )
            self.dev_question_file_path = Path(os.path.join(self.dev_dir, yaml_data["spider"]["dev"]["question_file"]))

            self.test_dir = yaml_data["spider"]["test"]["data_dir"]
            self.test_db_folder_path = Path(os.path.join(self.test_dir, yaml_data["spider"]["test"]["db_folder"]))
            self.test_tables_json_path = Path(
                os.path.join(self.test_dir, yaml_data["spider"]["test"]["tables_json_name"])
            )
            self.test_question_file_path = Path(
                os.path.join(self.test_dir, yaml_data["spider"]["test"]["question_file"])
            )


class SqlEnv:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.query_start_time = None
        self.query_timeout = 5  # seconds

    def started(self):
        return self.conn is not None

    def start_db(self, readonly=True):
        if readonly:
            readonly_path = "file:" + self.db_path + "?mode=ro"
            self.conn = sqlite3.connect(readonly_path, uri=True, timeout=5)
        else:
            self.conn = sqlite3.connect(self.db_path, timeout=5)

    def close_db(self):
        if self.conn is None:
            return
        self.conn.close()
        self.conn = None

    def _clean_sql(self, sql_command: str) -> str:
        cleaned = sql_command.replace("`", '"')
        cleaned = cleaned.rstrip(";")
        return cleaned

    def progress_handler(self):
        # Check if the elapsed time has exceeded our query_timeout.
        if time.time() - self.query_start_time > self.query_timeout:
            return 1  # Returning non-zero value aborts the query.
        return 0

    def exec_sql(self, sql_command: str) -> Union[str, List]:
        if not self.started():
            self.start_db()

        sql_command = self._clean_sql(sql_command)
        cur = self.conn.cursor()

        # Set up the progress handler with a callback frequency.
        self.query_start_time = time.time()
        # Call the handler every 1000 virtual machine instructions.
        self.conn.set_progress_handler(self.progress_handler, 1000)

        try:
            try:
                statements = sqlglot.parse(sql_command)
                if len(statements) > 1:
                    return "You can only execute one statement at a time"
            except Exception as e:
                return f"Error: {e}\n"

            cur.execute(sql_command)
            output = cur.fetchall()
        except sqlite3.Error as e:
            output = [str(e)]
        finally:
            cur.close()
            # Remove the progress handler after execution.
            self.conn.set_progress_handler(None, 0)
        return output


def _gen_data_fetch(col_name: str, table_name: str):
    col_name = col_name.replace('"', '""')
    table_name = table_name.replace('"', '""')

    query = f'SELECT "{col_name}" FROM "{table_name}" LIMIT 3'
    return query


def create_db_schema(db_metadata: Dict[str, Any], db_path: str) -> str:
    table_names = db_metadata["table_names_original"]
    column_names = db_metadata["column_names_original"]
    column_types = db_metadata["column_types"]
    primary_keys = db_metadata["primary_keys"]
    foreign_keys = db_metadata["foreign_keys"]

    # Create table columns and primary key set
    table_columns = [[] for _ in range(len(table_names))]
    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx != -1:  # Ignore the '*' wildcard
            table_columns[table_idx].append((idx, col_name))

    flattened_pks = []
    for pk in primary_keys:
        if isinstance(pk, list):
            flattened_pks.extend(pk)
        else:
            flattened_pks.append(pk)

    primary_key_set = set(flattened_pks)

    # Create a foreign key mapping for quick lookup
    foreign_key_map = {}
    for fk_idx, ref_idx in foreign_keys:
        foreign_key_map[fk_idx] = ref_idx

    # Open database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    output = []

    for table_idx, table_name in enumerate(table_names):
        output.append(f"{table_name} :")
        columns = table_columns[table_idx]

        for col_idx, col_name in columns:
            col_type = column_types[col_idx]

            # Fetch sample data
            cursor.execute(_gen_data_fetch(col_name, table_name))
            sample_data = [str(row[0]) for row in cursor.fetchall()]

            # Format the sample data as string
            sample_str = ", ".join(f'"{value}"' for value in sample_data)

            # Check if the column is a primary key
            is_primary_key = col_idx in primary_key_set
            primary_key_text = "primary_key" if is_primary_key else ""

            # Check if the column references another table
            ref_text = ""
            if col_idx in foreign_key_map:
                ref_idx = foreign_key_map[col_idx]
                ref_table_idx, ref_col_name = column_names[ref_idx]
                ref_table_name = table_names[ref_table_idx]
                ref_text = f"{table_name}.{col_name}={ref_table_name}.{ref_col_name}"

            # Append formatted column description
            column_desc = f"{col_name} [ {col_type.upper()} ] ( {sample_str} ) {primary_key_text} {ref_text}"
            output.append(column_desc.strip())

        output.append("")  # Add a blank line between tables

    conn.close()
    return "\n".join(output).strip()


def _load_db_metadata(db_metadata_path: Path) -> Dict[str, Any]:
    if not os.path.exists(db_metadata_path):
        raise FileNotFoundError(f"tables.json not found in {db_metadata_path}")
    with open(db_metadata_path, "r", encoding="utf-8") as f:
        raw_metadata = json.load(f)
    db_metadata = {r["db_id"]: r for r in raw_metadata}
    return db_metadata


def get_db_path(db_folder: Path, db_name: str) -> str:
    db_dir = db_folder / db_name
    db_path = db_dir / (db_name + ".sqlite")
    return db_path.as_posix()


def load_bird_dataset(data_config: EvalConfig, mode: str, cache_dir: str):
    if mode == "train":
        db_folder = data_config.train_db_folder_path
        tables_json_path = data_config.tables_json_path
        question_set_path = data_config.train_question_file_path
    elif mode == "dev":
        db_folder = data_config.dev_db_folder_path
        tables_json_path = data_config.dev_tables_json_path
        question_set_path = data_config.dev_question_file_path
    else:
        raise ValueError(f"Invalid mode: {mode}")

    raw_metadata = _load_db_metadata(tables_json_path)
    questions = load_dataset(
        "json",
        data_files=question_set_path.as_posix(),
        cache_dir=cache_dir,
        split="train",
    )
    db_schema_generator = create_db_schema
    db_desc_str = {
        db_id: db_schema_generator(raw_metadata[db_id], get_db_path(db_folder, db_id)) for db_id in raw_metadata
    }

    return db_desc_str, questions, db_folder


def load_spider_dataset(data_config: EvalConfig, mode: str, cache_dir: str):
    if mode == "train":
        db_folder = data_config.train_db_folder_path
        tables_json_path = data_config.tables_json_path
        question_set_path = data_config.train_question_file_path
        question_other_path = data_config.train_question_other_file_path

    elif mode == "dev":
        db_folder = data_config.dev_db_folder_path
        tables_json_path = data_config.dev_tables_json_path
        question_set_path = data_config.dev_question_file_path
    elif mode == "test":
        db_folder = data_config.test_db_folder_path
        tables_json_path = data_config.test_tables_json_path
        question_set_path = data_config.test_question_file_path
    else:
        raise ValueError(f"Invalid mode: {mode}")

    raw_metadata = _load_db_metadata(tables_json_path)
    with open(question_set_path, "r") as f:
        questions = json.load(f)
    if mode == "train":
        with open(question_other_path, "r") as f:
            questions_other = json.load(f)
        questions += questions_other

    questions = [{"db_id": q["db_id"], "question": q["question"], "SQL": q["query"]} for q in questions]
    db_schema_generator = create_db_schema
    db_desc_str = {
        db_id: db_schema_generator(raw_metadata[db_id], get_db_path(db_folder, db_id)) for db_id in raw_metadata
    }

    return db_desc_str, questions, db_folder


class SqlTask:
    """Wrapper for SQL environment, question, ground truth, and LLM template."""

    def __init__(self, db_id: str, db_desc: str, db_path: str, ground_truth: str = None):
        self.db_id = db_id
        self.ground_truth = ground_truth
        self.db_desc = db_desc
        self.db_path = db_path

        # TODO: -1 is the assistant's header, -2 is the user's question. We should
        # ensure this somewhere, or enable a more general template.

        self.sql_env: Optional[SqlEnv] = None
        self.answer: Set[Any] = None

    def launch_env(self):
        self.sql_env = SqlEnv(self.db_path)
        self.sql_env.start_db()
        if self.ground_truth:
            self.answer = self.exec_sql(self.ground_truth)

    def close_env(self):
        self.sql_env.close_db()

    def exec_sql(self, sql: str):
        return self.sql_env.exec_sql(sql.strip())
