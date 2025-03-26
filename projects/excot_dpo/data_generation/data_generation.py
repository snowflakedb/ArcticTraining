import os
import yaml

from arctic_training.synth import OpenAISynth, AzureOpenAISynth, VllmSynth, CortexSynth
from prompts.divide_and_conquer import messages as dnc_messages
from datasets import load_from_disk
from datasets import Dataset
from dataclasses import dataclass
from datasets import load_dataset

from utils.sql_exec import SqlTask, get_db_path
from utils.sql_exec import create_db_schema
from utils.sql_exec import _load_db_metadata
from typing import Dict
from typing import Union
from typing import List
from pathlib import Path


@dataclass
class DataGenerationConfig:
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
    n: int = 1

    def load_yaml(self, yaml_path: str):
        # Load the yaml file to initialize the configuration fields
        with open(yaml_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        sample_params = yaml_data["sample_params"]
        self.n = sample_params["n"]
        self.cache_dir = yaml_data["cache_dir"]
        if "bird" in yaml_data["data"]:
            self.task = "bird"
            # Now update the attributes from the loaded YAML data
            self.train_dir = yaml_data["data"]["bird"]["train"]["data_dir"]
            self.train_db_folder_path = Path(
                os.path.join(self.train_dir, yaml_data["data"]["bird"]["train"]["db_folder"])
            )
            self.tables_json_path = Path(
                os.path.join(
                    self.train_dir, yaml_data["data"]["bird"]["train"]["tables_json_name"]
                )
            )
            self.train_question_file_path = Path(
                os.path.join(
                    self.train_dir, yaml_data["data"]["bird"]["train"]["question_file"]
                )
            )

            self.dev_dir = yaml_data["data"]["bird"]["dev"]["data_dir"]
            self.dev_db_folder_path = Path(
                os.path.join(self.dev_dir, yaml_data["data"]["bird"]["dev"]["db_folder"])
            )
            self.dev_tables_json_path = Path(
                os.path.join(self.dev_dir, yaml_data["data"]["bird"]["dev"]["tables_json_name"])
            )
            self.dev_question_file_path = Path(
                os.path.join(self.dev_dir, yaml_data["data"]["bird"]["dev"]["question_file"])
            )

            self.test_dir = yaml_data["data"]["bird"]["test"]["data_dir"]
            self.test_db_folder_path = Path(
                os.path.join(self.test_dir, yaml_data["data"]["bird"]["test"]["db_folder"])
            )
            self.test_tables_json_path = Path(
                os.path.join(
                    self.test_dir, yaml_data["data"]["bird"]["test"]["tables_json_name"]
                )
            )
            self.test_question_file_path = Path(
                os.path.join(self.test_dir, yaml_data["data"]["bird"]["test"]["question_file"])
            )
        elif "spider" in yaml_data["data"]:
            self.task = "spider"
            self.train_dir = yaml_data["data"]["spider"]["train"]["data_dir"]
            self.train_db_folder_path = Path(
                os.path.join(self.train_dir, yaml_data["data"]["spider"]["train"]["db_folder"])
            )
            self.tables_json_path = Path(
                os.path.join(
                    self.train_dir, yaml_data["data"]["spider"]["train"]["tables_json_name"]
                )
            )
            self.train_question_file_path = Path(
                os.path.join(
                    self.train_dir, yaml_data["data"]["spider"]["train"]["question_set_file"]
                )
            )
            self.train_question_other_file_path = Path(
                os.path.join(
                    self.train_dir, yaml_data["data"]["spider"]["train"]["question_other_file"]
                )
            )

            self.dev_dir = yaml_data["data"]["spider"]["dev"]["data_dir"]
            self.dev_db_folder_path = Path(
                os.path.join(self.dev_dir, yaml_data["data"]["spider"]["dev"]["db_folder"])
            )
            self.dev_tables_json_path = Path(
                os.path.join(
                    self.dev_dir, yaml_data["data"]["spider"]["dev"]["tables_json_name"]
                )
            )
            self.dev_question_file_path = Path(
                os.path.join(self.dev_dir, yaml_data["data"]["spider"]["dev"]["question_file"])
            )

            self.test_dir = yaml_data["data"]["spider"]["test"]["data_dir"]
            self.test_db_folder_path = Path(
                os.path.join(self.test_dir, yaml_data["data"]["spider"]["test"]["db_folder"])
            )
            self.test_tables_json_path = Path(
                os.path.join(
                    self.test_dir, yaml_data["data"]["spider"]["test"]["tables_json_name"]
                )
            )
            self.test_question_file_path = Path(
                os.path.join(
                    self.test_dir, yaml_data["data"]["spider"]["test"]["question_file"]
                )
            )

def load_bird_dataset(data_config: DataGenerationConfig, mode: str, cache_dir: str):
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
        db_id: db_schema_generator(raw_metadata[db_id], get_db_path(db_folder, db_id))
        for db_id in raw_metadata
    }

    return db_desc_str, questions, db_folder


def load_spider_dataset(data_config: DataGenerationConfig, mode: str, cache_dir: str):
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

    questions = [
        {"db_id": q["db_id"], "question": q["question"], "SQL": q["query"]}
        for q in questions
    ]
    db_schema_generator = create_db_schema
    db_desc_str = {
        db_id: db_schema_generator(raw_metadata[db_id], get_db_path(db_folder, db_id))
        for db_id in raw_metadata
    }

    return db_desc_str, questions, db_folder

def dataset_conversion(data_config):
    if data_config.task == "bird":
        db_desc_str, questions, db_folder = load_bird_dataset(data_config, "train",
                                                                data_config.cache_dir)
    elif data_config.task == "spider":
        db_desc_str, questions, db_folder = load_spider_dataset(data_config, "train",
                                                                    data_config.cache_dir)
    # import pdb; pdb.set_trace()
    all_rows = []
    for i, question in enumerate(questions):
        db_id = question['db_id']
        db_desc = db_desc_str[db_id]
        if 'evidence' in question:
            all_rows.append({"schema": db_desc, "question": question['question'], "evidence": question["evidence"], "ground_truth": question['SQL']})
        else:
            all_rows.append({"schema": db_desc, "question": question['question'], "ground_truth": question['SQL']})

    if 'evidence' in questions[0]:
        new_dataset = Dataset.from_dict(
            {
                "schema": [d["schema"] for d in all_rows],
                "question": [d["question"] for d in all_rows],
                "evidence": [d["evidence"] for d in all_rows],
                "expected_answer": [d["ground_truth"] for d in all_rows],
            }
        )
    else:

        new_dataset = Dataset.from_dict(
            {
                "schema": [d["schema"] for d in all_rows],
                "question": [d["question"] for d in all_rows],
                "expected_answer": [d["ground_truth"] for d in all_rows],
            }
        )
    return new_dataset



def submit_requests(loaded_dataset, client, task_name="demo", model="gpt-4o", n=32):
    for row in loaded_dataset:
        client.add_chat_to_batch_task(
            task_name=task_name,
            model=model,
            messages=row["messages"],
            n=n
        )
    extracted_messages = client.extract_messages_from_responses(
        client.execute_batch_task(task_name)
    )

    return extracted_messages


def construct_gpt_prompt(row: Union[Dict, List]):
    schema, question = row["schema"], row["question"]
    if ("evidence" in row) and not (row["evidence"] is None):
        question = question + " " + row["evidence"]
    prompt = f"""
Database Info
{schema}
**************************
Question
Question: {question}
**************************
""".strip()
    rt_messages = dnc_messages + [
        {"role": "user", "content": prompt},
    ]
    return rt_messages


def main():
    # GPT data generation
    # client = AzureOpenAISynth(api_key=...)
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    client = AzureOpenAISynth(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-07-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    # Load data config
    bird_config_path = "/code/users/bohanzhai/ArcticTraining/projects/excot_dpo/data_generation/configs/bird_config.yaml"
    data_gen_config = DataGenerationConfig()
    data_gen_config.load_yaml(bird_config_path)
    loaded_dataset = dataset_conversion(data_gen_config)
    processed_dataset = loaded_dataset.map(
        lambda row: {"messages": construct_gpt_prompt(row)}
    )
    processed_dataset = processed_dataset.select(range(10))
    # GPT batch job
    bird_result = submit_requests(processed_dataset, client)
    import pdb; pdb.set_trace()
    # spider_result = submit_requests(spider_dataset, client)


if __name__ == "__main__":
    main()
