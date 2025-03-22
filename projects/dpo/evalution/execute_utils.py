import re
from typing import Union
from typing import Dict
from typing import List

from prompts.divide_and_conquer import messages as PROMPT_MESSAGES

_TEXT2SQL_COT_INSTRUCTION = (
    "Below I will provide a DB schema and a question that can be answered by querying the provided DB."
    "The SQL query must be enclosed in ```sql ...``` that answers the question"
)

_TEXT2SQL_NON_COT_INSTRUCTION = (
    "Below I will provide a DB schema and a question that can be answered by querying the provided DB. "
    "You will then write a SQL query enclosed in ```sql ...``` that answers the question "
    "(and nothing else)."
)

_SYSTEM_PROMPT = "You are an AI assistant helping a data analyst write SQL queries to answer questions. "


def _extract_sql(message):
    """
    Given a message, this function extracts the last SQL code by regex matching.
    The code should be wrapped by the MarkDown code block
    """
    pattern = r"```sql(.*?)```$"
    pattern2 = r"sql(.*?)```"

    matches = re.findall(pattern, message, flags=re.DOTALL | re.MULTILINE)
    matches2 = re.findall(pattern2, message, flags=re.DOTALL | re.MULTILINE)
    if matches:
        return matches[-1].strip()
    else:
        if matches2:
            return matches2[-1].strip()
        print("\nno matched code found...\n")
        return None


def construct_dnc_prompt(schema, question, evidence=None):
    if not (evidence is None):
        question = question + " " + evidence
    prompt = f"""
Database Info
{schema}
**************************
Question
Question: {question}
**************************
""".strip()
    rt_messages = PROMPT_MESSAGES + [
        {"role": "user", "content": prompt},
    ]
    rt_messages_instr = [rt_messages[0], rt_messages[-1]]
    return rt_messages_instr


def check_blob_pattern(text):
    pattern = r"(\[ BLOB \]\s*)\([^)]*\)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches

def remove_blob_content(text):
    # Regular expression pattern to match [BLOB](...)
    pattern = r"(\[ BLOB \]\s*)\([^)]*\)"

    # Replace the matched pattern with an empty string
    cleaned_text = re.sub(pattern, r"\1()", text, flags=re.DOTALL)
    return cleaned_text



def construct_prompt_cot(schema, question, evidence=None):
    if not (evidence is None):
        question = question + " " + evidence
    prompt = f"""{_TEXT2SQL_COT_INSTRUCTION}

# # Schema
# {schema}

# # Question
# {question}

# Please provide chain of thought and final sql answer:
# """
    rt_messages = [
        {"role": "system", "content": f"""{_SYSTEM_PROMPT}"""},
        {"role": "user", "content": prompt},
    ]
    return rt_messages


def construct_long_cot(schema, question, evidence=None):

    evidence = '' if evidence is None else evidence
    messages = [
            {
                'role': 'system',
                'content': "You are a SQL expert to help solve users' Text2SQL problems based on the provided database scheme. Please think step by step and write your answer in the form of ```sql ...```.",
            },
            {
                'role': 'user',
                'content': f"Database Schema: {schema.strip()}\n\nQuestion: {question + ' ' + evidence}",
            }
        ]
    return messages


def construct_prompt_non_cot(schema, question, evidence=None):
    if not (evidence is None):
        question = question + " " + evidence
    prompt = f"""{_TEXT2SQL_NON_COT_INSTRUCTION}

Snowflake database schema:
{schema}

The question:{question}"""
    rt_messages = [
        {"role": "system", "content": f"""{_SYSTEM_PROMPT}"""},
        {"role": "user", "content": prompt},
    ]
    return rt_messages
