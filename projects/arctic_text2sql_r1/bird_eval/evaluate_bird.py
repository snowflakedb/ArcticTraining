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
import multiprocessing as mp
import os
import random
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd
from func_timeout import FunctionTimedOut
from func_timeout import func_timeout
from self_consistency import calculate_similarity_matrix

warnings.simplefilter(action="ignore", category=FutureWarning)


random.seed(42)

execution_results = None
evaluation_results = None

DO_PRINT = False
SELF_CONSISTENCY = "Snow"  # or OmniSQL


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default="predict_dev.json")
    parser.add_argument("--gold", type=str, default="./bird/dev/dev.json")
    parser.add_argument("--db_path", type=str, default="./bird/dev/dev_databases")
    parser.add_argument("--mode", type=str, default="greedy_search")

    opt = parser.parse_args()

    return opt


def execute_sql(data_idx, db_file, sql):
    """
    Executes `sql` against the SQLite database at `db_file`.

    Returns:
        (data_idx, db_file, sql, execution_res, success_flag)
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        if SELF_CONSISTENCY == "OmniSQL":
            conn.execute("BEGIN TRANSACTION;")
            cursor.execute(sql)
            rows = cursor.fetchall()
            execution_res = frozenset(rows)  # make the result hashable
            conn.rollback()
            return data_idx, db_file, sql, execution_res, 1
        else:
            try:
                df = pd.read_sql_query(sql, conn)
                execution_res = df
                success = True
            except pd.io.sql.DatabaseError as db_err:
                if DO_PRINT:
                    print(f"Could not execute SQL (pandas): {db_err}")
                execution_res = None
                success = False

            # success flag is 1 if we got a non-empty DataFrame, else 0
            return data_idx, db_file, sql, execution_res, int(success and len(execution_res) > 0)

    except sqlite3.DatabaseError as sql_err:
        if DO_PRINT:
            print(f"Database error during execution: {sql_err}")
        return data_idx, db_file, sql, None, 0

    finally:
        conn.close()


def compare_sql(question_id, db_file, question, ground_truth, pred_sql):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    correctness = 0

    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(pred_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        if DO_PRINT:
            print("Successfully executed")
        if set(predicted_res) == set(ground_truth_res):
            correctness = 1
        conn.rollback()
    except sqlite3.DatabaseError:
        conn.rollback()
    finally:
        conn.close()
    return question_id, db_file, question, ground_truth, pred_sql, correctness


def compare_sql_wrapper(args, timeout):
    """Wrap execute_sql for timeout"""
    try:
        result = func_timeout(timeout, compare_sql, args=args)
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = (*args, 0)
    except Exception:
        result = (*args, 0)
    return result


def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Data index:{data_idx}\nSQL:\n{sql}\nTime Out!")
        print("-" * 30)
        res = (data_idx, db_file, sql, None, 0)
    except Exception:
        res = (data_idx, db_file, sql, None, 0)

    return res


def execute_callback_evaluate_sql(result):
    """Store the execution result in the collection"""
    question_id, db_file, question, ground_truth, pred_sql, correctness = result
    evaluation_results.append(
        {
            "question_id": question_id,
            "db_file": db_file,
            "question": question,
            "ground_truth": ground_truth,
            "pred_sql": pred_sql,
            "correctness": correctness,
        }
    )
    if DO_PRINT:
        print("Done:", question_id, correctness)  # Print the progress
    sys.stdout.flush()
    sys.stderr.flush()


def execute_callback_execute_sqls(result):
    data_idx, db_file, sql, query_result, valid = result
    if DO_PRINT:
        print("Done:", data_idx)  # Print the progress

    execution_results.append(
        {"data_idx": data_idx, "db_file": db_file, "sql": sql, "query_result": query_result, "valid": valid}
    )


def evaluate_sqls_parallel(db_files, questions, pred_sqls, ground_truth_sqls, num_cpus=1, timeout=1):
    """Execute the sqls in parallel"""
    pool = mp.Pool(processes=num_cpus)
    for question_id, db_file, question, pred_sql, ground_truth in zip(
        [x for x in range(len(db_files))], db_files, questions, pred_sqls, ground_truth_sqls
    ):
        pool.apply_async(
            compare_sql_wrapper,
            args=((question_id, db_file, question, ground_truth, pred_sql), timeout),
            callback=execute_callback_evaluate_sql,
        )
    pool.close()
    pool.join()


def execute_sqls_parallel(db_files, sqls, num_cpus=1, timeout=1):
    pool = mp.Pool(processes=num_cpus)
    for data_idx, db_file, sql in zip(list(range(len(sqls))), db_files, sqls):
        pool.apply_async(
            execute_sql_wrapper, args=(data_idx, db_file, sql, timeout), callback=execute_callback_execute_sqls
        )
    pool.close()
    pool.join()


def mark_invalid_sqls(db_files, sqls):
    global execution_results
    execution_results = []
    execute_sqls_parallel(db_files, sqls, num_cpus=20, timeout=10)
    execution_results = sorted(execution_results, key=lambda x: x["data_idx"])

    for idx, res in enumerate(execution_results):
        if res["valid"] == 0:
            sqls[idx] = "Error SQL"
    return sqls


def major_voting(db_files, pred_sqls, sampling_num, return_random_one_when_all_errors=True):
    global execution_results
    mj_pred_sqls = []
    execution_results = []
    # execute all sampled SQL queries to obtain their execution results
    execute_sqls_parallel(db_files, pred_sqls, num_cpus=20, timeout=10)
    execution_results = sorted(execution_results, key=lambda x: x["data_idx"])
    if DO_PRINT:
        print("len(execution_results):", len(execution_results))

    # perform major voting
    for result_idx in range(0, len(execution_results), sampling_num):
        major_voting_counting = dict()

        execution_results_of_one_sample = execution_results[result_idx : result_idx + sampling_num]

        if SELF_CONSISTENCY == "OmniSQL":
            # if no predicted SQLs are valid
            if sum([res["valid"] for res in execution_results_of_one_sample]) == 0:
                if return_random_one_when_all_errors:
                    mj_pred_sql = random.choice(execution_results_of_one_sample)[
                        "sql"
                    ]  # select a random one to return
                else:
                    mj_pred_sql = "Error SQL"
                mj_pred_sqls.append(mj_pred_sql)
                continue

            for res in execution_results_of_one_sample:
                if res["valid"] == 1:  # skip invalid SQLs
                    if res["query_result"] in major_voting_counting:
                        major_voting_counting[res["query_result"]]["votes"] += 1
                    else:
                        major_voting_counting[res["query_result"]] = {"votes": 1, "sql": res["sql"]}

            # find the SQL with the max votes
            major_vote = max(major_voting_counting.values(), key=lambda x: x["votes"])
            mj_pred_sql = major_vote["sql"]
            mj_pred_sqls.append(mj_pred_sql)
        else:
            results = [res["query_result"] for res in execution_results_of_one_sample]
            similarity_matrix = calculate_similarity_matrix(results)
            scores = np.sum(similarity_matrix, -1)
            best_idx = np.argmax(scores)
            mj_pred_sql = execution_results_of_one_sample[best_idx]["sql"]
            mj_pred_sqls.append(mj_pred_sql)

    return mj_pred_sqls


def run_eval(gold_file, pred_file, db_path, mode, save_pred_sqls, num_cpus=20, timeout=10):
    global evaluation_results
    gold = json.load(open(gold_file))
    pred_results = json.load(open(pred_file))
    db_files = [os.path.join(db_path, data["db_id"], data["db_id"] + ".sqlite") for data in gold]
    questions = [data["question"] for data in gold]
    pred_sql_key = "pred_sqls"
    # pred_sql_key = "responses"

    if "bird" in gold_file:
        ground_truth_sqls = [data["SQL"] for data in gold]
    else:
        ground_truth_sqls = [data["query"] for data in gold]

    if mode == "greedy_search":
        pred_sqls = [res[pred_sql_key][0] for res in pred_results]

        # save the (greedy-search) predicted SQL so we can check it out later
        if save_pred_sqls:
            with open(pred_file[:-5] + "_pred_greedy_search_sqls.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(pred_sqls, indent=2, ensure_ascii=False))

        assert len(pred_results) == len(pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        evaluate_sqls_parallel(db_files, questions, pred_sqls, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)

        # sort evaluation_results by question_id
        evaluation_results = sorted(evaluation_results, key=lambda x: x["question_id"])
        evaluation_scores = [res["correctness"] for res in evaluation_results]
        for res in evaluation_results:
            if res["correctness"] == 0:
                if DO_PRINT:
                    print("question:", res["question"])
                    print("GT:", res["ground_truth"])
                    print("Pred:", res["pred_sql"])
                    print("-" * 30)
        print("EX Accuracy (greedy search):", sum(evaluation_scores) / len(evaluation_scores))

        return sum(evaluation_scores) / len(evaluation_scores), pred_sqls
    elif mode == "major_voting":
        sampling_num = len(pred_results[0][pred_sql_key])
        print("sampling_num:", sampling_num)

        db_files = []
        for gold_data in gold:
            db_files.extend([os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite")] * sampling_num)

        pred_sqls = []
        for pred_data in pred_results:
            pred_sqls.extend(pred_data[pred_sql_key])
        assert len(pred_sqls) == len(db_files)

        mj_pred_sqls = major_voting(db_files, pred_sqls, sampling_num)

        # save the (major-voting) predicted SQL so we can check it out later
        if save_pred_sqls:
            with open(pred_file[:-5] + "_pred_major_voting_sqls.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(mj_pred_sqls, indent=2, ensure_ascii=False))

        # reset db_files
        db_files = []
        for gold_data in gold:
            db_files.append(os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite"))

        assert len(mj_pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        evaluate_sqls_parallel(
            db_files, questions, mj_pred_sqls, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout
        )

        # sort evaluation_results by question_id
        evaluation_results = sorted(evaluation_results, key=lambda x: x["question_id"])
        evaluation_scores = [res["correctness"] for res in evaluation_results]
        print("EX Accuracy (major voting):", sum(evaluation_scores) / len(evaluation_scores))

        return sum(evaluation_scores) / len(evaluation_scores), mj_pred_sqls
    elif mode == "pass@k":
        all_scores = []
        sampling_num = len(pred_results[0][pred_sql_key])

        db_files = []
        for gold_data in gold:
            db_files.append(os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite"))

        for sample_idx in range(sampling_num):
            pred_sqls_for_specific_sample_idx = [pred_data[pred_sql_key][sample_idx] for pred_data in pred_results]
            evaluation_results = []
            evaluate_sqls_parallel(
                db_files,
                questions,
                pred_sqls_for_specific_sample_idx,
                ground_truth_sqls,
                num_cpus=num_cpus,
                timeout=timeout,
            )
            evaluation_results = sorted(evaluation_results, key=lambda x: x["question_id"])
            evaluation_scores = [res["correctness"] for res in evaluation_results]
            all_scores.append(evaluation_scores)
        pass_at_k_scores = [1 if any(column) else 0 for column in zip(*all_scores)]
        print(f"EX Accuracy (pass@{sampling_num}):", sum(pass_at_k_scores) / len(pass_at_k_scores))
        return sum(pass_at_k_scores) / len(pass_at_k_scores), None
    else:
        raise ValueError("mode should be in [greedy_search, major_voting, pass@k]")


"""
python evaluate_bird.py --pred ./results/spider_dev_greedy_search_ckpt-5306.json --gold ../data/spider/dev.json --db_path ../data/spider/database/
python evaluate_bird.py --pred ./results/bird_dev_greedy_search_ckpt-5306.json --gold ../data/bird/dev_20240627/dev.json --db_path ../data/bird/dev_20240627/dev_databases/
"""

if __name__ == "__main__":
    opt = parse_option()
    run_eval(opt.gold, opt.pred, opt.db_path, opt.mode, False)
