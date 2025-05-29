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

import argparse
import json
import os
import shutil
import sqlite3

from func_timeout import func_set_timeout


def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print(f"Opening a new connection {sqlite_path}")
        conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    except Exception:
        # will re-raise whatever exception occurred,
        # printing the path to help debug
        print(sqlite_path)
        raise
    conn.text_factory = lambda b: b.decode(errors="ignore")
    return conn.cursor()


@func_set_timeout(3600)
def execute_sql(cursor, sql):
    cursor.execute(sql)
    return cursor.fetchall()


def remove_contents_of_a_folder(index_path):
    os.makedirs(index_path, exist_ok=True)
    for filename in os.listdir(index_path):
        file_path = os.path.join(index_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_content_index(db_file_path, index_path, temp_dir, threads):
    cursor = get_cursor_from_path(db_file_path)
    # fetch table names
    results = execute_sql(cursor, "SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [r[0] for r in results]

    all_column_contents = []
    for tname in table_names:
        if tname == "sqlite_sequence":
            continue
        cols = execute_sql(cursor, f"PRAGMA TABLE_INFO('{tname}')")
        col_names = [c[1] for c in cols]
        for cname in col_names:
            try:
                sql = f"SELECT DISTINCT `{cname}` FROM `{tname}` WHERE `{cname}` IS NOT NULL;"
                results = execute_sql(cursor, sql)
                contents = [r[0] for r in results if isinstance(r[0], str) and not is_number(r[0])]
                for cid, content in enumerate(contents):
                    if 0 < len(content) <= 40:
                        all_column_contents.append({"id": f"{tname}-**-{cname}-**-{cid}", "contents": content})
            except Exception as e:
                print(f"Error on {tname}.{cname}: {e}")

    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, "contents.json")
    with open(temp_file, "w") as f:
        json.dump(all_column_contents, f, indent=2, ensure_ascii=True)

    cmd = (
        "python -m pyserini.index.lucene --collection JsonCollection "
        f"--input {temp_dir} --index {index_path} "
        f"--generator DefaultLuceneDocumentGenerator --threads {threads} "
        "--storePositions --storeDocvectors --storeRaw"
    )
    ret = os.system(cmd)
    print(f"Indexing returned: {ret}")

    os.remove(temp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25-style content indices for a directory of SQLite DBs")
    parser.add_argument("--db-root", required=True, help="Root directory containing subdirectories of .sqlite files")
    parser.add_argument("--index-root", required=True, help="Root output directory for per-DB Lucene indexes")
    parser.add_argument("--temp-dir", default="/data-fast/temp_db_index", help="Temporary folder for JSON collection")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads for indexing")
    args = parser.parse_args()

    # clear root index folder
    remove_contents_of_a_folder(args.index_root)

    for db_id in os.listdir(args.db_root):
        db_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
        idx_path = os.path.join(args.index_root, db_id)
        if os.path.isfile(db_path):
            print(f"Processing {db_path} -> {idx_path}")
            build_content_index(db_path, idx_path, args.temp_dir, args.threads)
        else:
            print(f"Skipping missing or non-file {db_path}")
