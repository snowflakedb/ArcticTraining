import json
import logging
import re
from pathlib import Path

from snow_rlhf.verl_comp.utils.dataset.long_system_instruction import long_system_instruction

logger = logging.getLogger(__name__)

# Global flag to log template rendering only once
_template_logged = False


def load_template(template_name: str) -> str:
    """Load a template file from the same directory as this module."""
    current_dir = Path(__file__).parent
    template_path = current_dir / template_name
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def render_autocomplete_template(prefix: str, suffix: str) -> str:
    """Render the autocomplete prompt template with the provided values."""
    template_content = load_template("autocomplete_prompt.tmpl")
    
    # Use a simple placeholder replacement approach similar to Go templates
    template_content = template_content.replace("{{.PREFIX}}", prefix)
    template_content = template_content.replace("{{.SUFFIX}}", suffix)
    
    return template_content


def render_autocomplete_system_prompt() -> str:
    """Render the static system prompt for autocomplete"""
    return """/* You are a Snowflake SQL completion model. Fill in missing SQL code between prefix and suffix.
Output ONLY the missing tokens with no prose, backticks, or explanations.

RULES:

0) NEW QUERY (when prefix and suffix are empty or whitespace):
   - Pattern detection: If last_few_executed has queries, use it to detect patterns. If the same query 
     structure appears in multiple queries in last_few_executed with changes in LIMIT values, dates, 
     columns, filters, etc., predict the NEXT value by continuing the progression (10→1000→100000, '08-04'→'08-05'→'08-06').
   - Keep query structure unchanged, modify only the varying element
   - MUST use the same table, MUST differ from all previous queries
   - NEVER copy the last query verbatim
   - If no pattern detected in last_few_executed, use other context sources to suggest the next query. If uncertain, return "".

1) Context: Use only tables/columns/literals from prefix/suffix, last_few_executed, sql_history, ddl_context, 
   or file_content. Use aliased columns when aliases exist (e.g., FROM orders o → use o.order_id).

2) Comment-to-SQL: If comment has trailing \n and suffix is empty, translate to SQL. Without \n, return <no_suggestion> as response.

3) Literals: Prefer values from the provided context. Fallback to plausible integers (1, 100000000), 
   short strings ('value'), known Snowflake literals, or dates (CURRENT_DATE, '2025-01-01').

4) No duplicate tokens: Don't repeat operators or SQL tokens already typed. 
   Example: "WHERE id = " → "12345" (not "= 12345").

5) Empty output: Return <no_suggestion> as response if query is complete, no context match exists, comment lacks trailing newline, or if unsure in any cases.

6) Output Format: Output only raw new SQL tokens. No backticks, no prose, no extra whitespace.

7) Code style: Match the formatting style from prefix, suffix, file_content or other context sources (e.g., indentation, spacing, casing).


EXAMPLES:

MID-QUERY:
- prefix="SELECT * FROM orders;" suffix="" → "<no_suggestion>"
- prefix="SELECT * FROM orders o JOIN customers c ON " suffix="= c.customer_id" → "o.customer_id"
- prefix="SELECT region, product, SUM(revenue) FROM sales GROUP BY region" suffix="" → ", product"
- prefix="WHERE id = " suffix="" → "12345"

NEW QUERY (prefix/suffix empty):
- last_few_executed: [SELECT * FROM t LIMIT 10; SELECT * FROM t LIMIT 1000;]
  → SELECT * FROM t LIMIT 100000; (continue progression, NOT copy last)

- last_few_executed: [SELECT * FROM e WHERE d='2024-08-05'; SELECT * FROM e WHERE d='2024-08-06';]
  → SELECT * FROM e WHERE d='2024-08-07'; (next date, same table)

- last_few_executed: [SELECT fv['a'] FROM t1; SELECT fv['b'] FROM t1;] sql_history: [SELECT * FROM t2;]
  → SELECT fv['c'] FROM t1; (use last_few_executed, ignore sql_history)

- last_few_executed: [] sql_history: [SELECT * FROM c WHERE x='US'; SELECT * FROM c WHERE x='CA';]
  → SELECT * FROM c WHERE x='UK'; (fallback to sql_history)
  */"""


def render_autocomplete_user_prompt(
    prefix: str,
    suffix: str,
    sql_history: str = "",
    ddl_context: str = "",
    last_few_executed_context: str = "",
    file_content: str = "",
) -> str:
    """
    Render the user prompt with context AND FIM tokens.
    
    Use this for inference where user message contains everything.
    For training with FIM in assistant response, use render_autocomplete_user_context instead.
    """
    context_parts = []
    
    # 1. SQL history context
    if sql_history:
        context_parts.append(f"""/* sql_history (most recent queries appear last in this block)
{sql_history}
*/""")
    
    # 2. DDL context
    if ddl_context:
        context_parts.append(f"""/* ddl_context (Snowflake objects, columns, types)
{ddl_context}
*/""")
    
    # 3. Last few executed context
    if last_few_executed_context:
        context_parts.append(f"""/* last_few_executed_context (recently executed queries)
{last_few_executed_context}
*/""")
    
    # 4. File content (the entire worksheet)
    if file_content:
        context_parts.append(f"""/* file_content (entire worksheet)
{file_content}
*/""")
    
    # 5. FIM tokens with prefix/suffix (the SQL block being edited)
    template_content = load_template("autocomplete_prompt.tmpl")
    template_content = template_content.replace("{{.PREFIX}}", prefix)
    template_content = template_content.replace("{{.SUFFIX}}", suffix)
    
    # Combine context sections with FIM content
    if context_parts:
        context_block = "\n\n".join(context_parts)
        return f"""# ------------------ FULL CONTEXT  ------------------
{context_block}
# ------------------ END FULL CONTEXT --------------------------------------------------

{template_content}"""
    return template_content


def render_autocomplete_user_context(
    sql_history: str = "",
    ddl_context: str = "",
    last_few_executed_context: str = "",
    file_content: str = "",
) -> str:
    """
    Render ONLY the context for the user message (no FIM tokens).
    
    Use this for training where FIM tokens should be in the assistant response.
    
    Returns:
        str: Context block, or a minimal message if no context provided.
              Never returns empty string to ensure valid training examples.
    """
    context_parts = []
    
    # 1. SQL history context
    if sql_history:
        context_parts.append(f"""/* sql_history (most recent queries appear last in this block)
{sql_history}
*/""")
    
    # 2. DDL context
    if ddl_context:
        context_parts.append(f"""/* ddl_context (Snowflake objects, columns, types)
{ddl_context}
*/""")
    
    # 3. Last few executed context
    if last_few_executed_context:
        context_parts.append(f"""/* last_few_executed_context (recently executed queries)
{last_few_executed_context}
*/""")
    
    # 4. File content (the entire worksheet)
    if file_content:
        context_parts.append(f"""/* file_content (entire worksheet)
{file_content}
*/""")
    
    # Return context block, or minimal message if no context
    if context_parts:
        context_block = "\n\n".join(context_parts)
        return f"""# ------------------ FULL CONTEXT  ------------------
{context_block}
# ------------------ END FULL CONTEXT --------------------------------------------------"""
    
    # No context available - return minimal message to avoid empty user turns
    # This ensures the model learns the task structure even without SQL history
    return "/* No additional context available. Complete the SQL based on the prefix and suffix. */"


def render_autocomplete_assistant_response(prefix: str, suffix: str, output: str) -> str:
    """
    Render the assistant response with FIM tokens wrapping prefix/suffix and the completion.
    
    Format: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{output}
    
    Args:
        prefix: Text before the completion point
        suffix: Text after the completion point  
        output: The actual completion (what replaces <fillMe>)
        
    Returns:
        str: FIM-formatted assistant response
    """
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{output}"


def get_messages(
    data: dict, tokenizer, message_type="arctic_text_to_sql_r1", schema_name="omnisql_schema", engine="SQLite"
) -> list[dict[str, str]]:
    if message_type == "arctic_text_to_sql_r1":
        # cot_info = "Let me solve this step by step. \n<think>"
        instruct_info = """
Please provide a detailed chain-of-thought reasoning process and include your thought process within `<think>` tags. Your final answer should be enclosed within `<answer>` tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

```sql
Correct SQL query here
```
</answer>""".strip()
        messages = [
            {
                "role": "system",
                "content": "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.",
            },
            {
                "role": "user",
                "content": f"""
Database Engine:
{engine}

Database Schema:
{data[schema_name]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
{instruct_info}
    """.strip(),
            },
        ]
    elif message_type == "base_instruction":
        # cot_info = "Let me solve this step by step. \n<think>"
        messages = [
            {
                "role": "system",
                "content": "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.",
            },
            {
                "role": "user",
                "content": f"""
Database Engine:
{engine}

Database Schema:
{data[schema_name]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- Ensure that your SQL query follows the correct syntax and is formatted as follows:
```sql
-- Your SQL query here
```'''.strip()
    """.strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_r1_qwen3":
        messages = [
            {
                "role": "system",
                "content": "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.",
            },
            {
                "role": "user",
                "content": f"""
Database Engine:
{engine}

Database Schema:
{data[schema_name]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- Ensure that your final output SQL query follows the correct syntax and is in the form as follows: ```sql ... ```.
""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_r1_qwen3_long_system_instruction":
        messages = [
            {
                "role": "system",
                "content": f"""You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question. Besides, you are also provided with a detailed SQLite syntax reference, which you can refer to when writing the SQL query.

                {long_system_instruction}
                """.strip(),
            },
            {
                "role": "user",
                "content": f"""
Database Engine:
{engine}

Database Schema:
{data[schema_name]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- Ensure that your final output SQL query follows the correct syntax and is in the form as follows: ```sql ... ```.
""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_t1":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can execute SQL queries on SQLite databases to help answer user's questions.",
            },
            {
                "role": "user",
                "content": f"""Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
{engine}

Database Schema:
{data[schema_name]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query and use the tool to help your thinking process.
- If you are not sure which columns to use due to the ambiguity in the question, please use the tool to help you figure out the correct columns.
- You can at most call the tool 5 times and give the final SQL query.

Final Output Format:
When you are confident to output your final answer, please enclose the generated SQL query in a code block instead of giving a plain text answer or the SQL execution result:
```sql
```
""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_t1_v2":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can execute SQL queries on SQLite databases to help answer user's questions.",
            },
            {
                "role": "user",
                "content": f"""
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
{engine}

Database Schema:
{data[schema_name]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query and use the tool to help your thinking process.
- If you are not sure which columns to use due to the ambiguity in the question, please use the tool to help you figure out the correct columns.
- You should at least call the tool once to help you make sure the syntax of the SQL query is correct, aka it should be executable, and returns the expected results.
- You should decompose the question into smaller steps and use the tool to help you make sure each sub-step is corectly executed and returns the expected results.
- You can at most call the tool 5 times and give the final SQL query.

Final Output Format:
When you are confident to output your final answer, please enclose the generated SQL query in a code block instead of giving a plain text answer or the SQL execution result:
```sql
```
""".strip(),
            },
        ]

    elif message_type == "arctic_text_to_sql_t1_v3":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can execute SQL queries on SQLite databases to help answer user's questions.",
            },
            {
                "role": "user",
                "content": f"""
**Task Overview:**
You are an expert data scientist. You are provided with a database schema and a natural language question. Your task is to carefully analyze the schema and generate a correct and precise SQL query to answer the question.

**Database Engine:**
{engine}

**Database Schema:**
{data[schema_name]}
This schema defines the database structure, including tables, columns, primary keys, foreign keys, and relevant relationships or constraints.

**Question:**
{data["question"]}

**Instructions:**

* Your SQL query **must retrieve only** the information requested in the question. If the question specifies certain columns, your `SELECT` clause must include *only* those columns—nothing more, nothing less.
* The query should return complete and accurate results, with no missing, redundant, or irrelevant information.

**Reasoning Process:**

* Before writing the final SQL query, you are expected to **explicitly think through the query step by step**, reasoning about table selection, join conditions, filtering, grouping, and aggregation (if needed).
* You are encouraged to first use the provided tool to assist your reasoning process. **Always attempt at least one tool call before writing the final query.**
* If the question contains ambiguity regarding which columns to use or which tables are relevant, use the tool to explore and resolve this.
* You may call the tool **up to 5 times** to support your reasoning.

**Output Policy:**

* **Do not output partial or intermediate queries**. Only provide the final complete SQL query in your final answer.
* To maintain benchmark consistency, always format the final SQL query using the following exact format:

```sql
[Your SQL query here]
```

* **Always follow this format even if the final answer is a simple SELECT or requires no joins or filters.**

**Additional Notes for Benchmark Consistency:**

* Maintain consistent column ordering in SELECT clauses where the question does not specify ordering.
* Do not add implicit default clauses (e.g., `ORDER BY ROWID` or `LIMIT`) unless explicitly requested.
* Avoid use of aliases unless needed to resolve ambiguity.
* Use fully qualified table and column names where necessary to disambiguate.
""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_t1_v4":
        messages = [
            {
                "role": "system",
                "content": """## Data Scientist Task: Natural Language to SQL

Your mission is to act as an expert data scientist. You will be given a target SQL engine,  a database schema, and a natural language question. Your goal is to write a precise and efficient query that correctly answers the question.

### 1. Analyze the Provided Information

* **Database Schema:** You will receive a schema that details the database's tables, columns, primary keys, and foreign keys. This is your map to the data.
* **The Question:** Carefully examine the user's question to understand exactly what information is being requested.

### 2. The Reasoning Process

Before writing the final query, you must first reason through the problem.

* **Think Step-by-Step:** Verbally outline your plan. Identify the necessary tables, the required joins, any filtering conditions (`WHERE`), and if aggregation (`GROUP BY`) or ordering (`ORDER BY`) is needed.
* **Use Your Tools:** You have a tool to help you explore the schema and test your assumptions.
    * You are encouraged to use the tool to resolve any ambiguity in the question or schema.
    * You may call the tool up to **5 times**.
    * **Crucially, you must attempt at least one tool call before providing the final query, to help ensure the SQL you provide is executable without error.**

### 3. SQL Query Requirements

Your final query must be accurate and adhere to these rules:

* **Select Only What's Asked:** The `SELECT` statement must retrieve *only* the columns specified or implied in the question.
* **Complete and Accurate:** Ensure the query returns all required results without missing data or including irrelevant information.
* **Avoid Aliases:** Do not use table or column aliases unless they are necessary to resolve ambiguity.
* **No Implicit Clauses:** Do not add clauses like `ORDER BY` or `LIMIT` unless the question explicitly asks for them.

### 4. Final Output Format

Your final response **must only contain the complete SQL query**. Adhere strictly to the following format:

```sql
-- Your SQL query here
```""".strip(),
            },
            {
                "role": "user",
                "content": f"""
**Database Engine:**
{engine}

**Database Schema:**
{data[schema_name]}
This schema defines the database structure, including tables, columns, primary keys, foreign keys, and relevant relationships or constraints.

**Question:**
{data["question"]}
""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_t1_v5":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can execute SQL queries on SQLite databases to help answer user's questions.",
            },
            {
                "role": "user",
                "content": f"""
### Task
You are an **expert SQL engineer**. Using the *SQLite* schema provided below, write a single SQL query that **fully and only** answers the question.

---
#### Database Schema
{data[schema_name]}

#### Question
{data["question"]}
---

### Constraints
1. **Column fidelity** — select *only* the columns explicitly requested.
2. **No extras** — do not include unnecessary columns, tables, or rows.
3. **No `SELECT *`** — never use a wildcard selector unless the question explicitly asks for all columns.
4. **Precision joins & filters** — join and filter strictly as required.
5. **Deterministic ordering** — use `ORDER BY` only when the question demands a specific order.
6. **Alias discipline** — employ aliases solely to resolve ambiguity.
7. **Implicit clauses** — avoid defaults such as `LIMIT` or `ROWID` unless explicitly requested.

### Tool-Usage Requirement
You **must** call tool **at least once and no more than five times**. Calling it fewer than once or more than five times will incur a negative reward.

### Thinking & Tool-Call Guidelines
1. **Plan before you code.** Inside `<think>` tags, lay out a step-by-step plan:
    * Paraphrase what the question is asking.
    * Map each requested output column to the table(s) containing it.
    * Identify join paths via primary/foreign keys.
    * Specify filter conditions, grouping, and aggregation logic.
    * Decide whether an `ORDER BY` clause is needed.
2. **Exploratory tool call (required).** Use tool to inspect tables/columns or schema relationships.
3. **Optional verification calls.** You may make additional calls to confirm your join keys or test the query, provided the total call count stays ≤ 5.
4. **Update your plan** based on tool feedback, still inside `<think>` tags.
5. **Write the final SQL query**.

### Output Protocol
1. Output your detailed reasoning inside `<think>` tags *after* each tool call. Do **not** include any partial or intermediate SQL queries in these tags.
2. After the closing `</think>`, output **only** the final SQL query wrapped in a code block exactly like this:
```sql
-- your query here
```
""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_t1_v6":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can execute SQL queries on SQLite databases to help answer user's questions.",
            },
            {
                "role": "user",
                "content": f"""
**Task Overview:**
You are an expert data scientist. You are provided with a database schema and a natural language question. Your task is to carefully analyze the schema and generate a correct and precise SQL query to answer the question.

**Database Engine:**
SQLite

**Database Schema:**
{data[schema_name]}
This schema defines the database structure, including tables, columns, primary keys, foreign keys, and relevant relationships or constraints.

**Question:**
{data["question"]}

**Instructions:**

* Your SQL query **must retrieve only** the information requested in the question. If the question specifies certain columns, your `SELECT` clause must include *only* those columns—nothing more, nothing less.
* The query should return complete and accurate results, with no missing, redundant, or irrelevant information.

**Reasoning Process:**

* Before writing the final SQL query, you are expected to **explicitly think through the query step by step**, reasoning about table selection, join conditions, filtering, grouping, and aggregation (if needed).
* You are encouraged to first use the provided tool to assist your reasoning process. **Always attempt at least one tool call before writing the final query.** Failure to do so will incur a negative reward.
* If the question contains ambiguity regarding which columns to use or which tables are relevant, use the tool to explore and resolve this.
* You may call the tool **up to 5 times** to support your reasoning.
* Use the feedback from the tool calls to refine/modify your thinking and ensure the final SQL query is executable without error.

**Output Policy:**

* **Do not output partial or intermediate queries**. Only provide the final complete SQL query in your final answer.
* To maintain benchmark consistency, always format the final SQL query using the following exact format:

```sql
[Your SQL query here]
```

* **Always follow this format even if the final answer is a simple SELECT or requires no joins or filters.**

**Additional Notes for Benchmark Consistency:**

* Maintain consistent column ordering in SELECT clauses where the question does not specify ordering.
* Do not add implicit default clauses (e.g., `ORDER BY ROWID` or `LIMIT`) unless explicitly requested.
* Avoid use of aliases unless needed to resolve ambiguity.
* Use fully qualified table and column names where necessary to disambiguate.
""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_t1_v7":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can execute SQL queries on SQLite databases to help answer user's questions.",
            },
            {
                "role": "user",
                "content": f"""
**Task Overview:**
You are an expert data scientist. You are provided with a database schema and a natural language question. Your task is to carefully analyze the schema and generate a correct and precise SQL query to answer the question.

**Database Engine:**
SQLite

**Database Schema:**
{data[schema_name]}
This schema defines the database structure, including tables, columns, primary keys, foreign keys, and relevant relationships or constraints.

**Question:**
{data["question"]}

**Instructions:**

* Your SQL query **must retrieve only** the information requested in the question. If the question specifies certain columns, your `SELECT` clause must include *only* those columns—nothing more, nothing less.
* The query should return complete and accurate results, with no missing, redundant, or irrelevant information.

**Reasoning Process:**

* Before writing the final SQL query, you are expected to **explicitly think through the query step by step**, reasoning about table selection, join conditions, filtering, grouping, and aggregation (if needed).
* You are encouraged to first use the provided tool to assist your reasoning process. **Always attempt at least one tool call before writing the final query.**
* If the question contains ambiguity regarding which columns to use or which tables are relevant, use the tool to explore and resolve this.
* You may call the tool **up to 5 times** to support your reasoning.
* You **must enclose your reasoning process within <think> and </think> tags** before proceeding with any tool calls or writing the final query.

**Output Policy:**

* **Do not output partial or intermediate queries**. Only provide the final complete SQL query in your final answer.
* To maintain benchmark consistency, always format the final SQL query using the following exact format:

```sql
[Your SQL query here]
```

* **Always follow this format even if the final answer is a simple SELECT or requires no joins or filters.**


""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_r1_qwen3-sft":
        messages = [
            {
                "role": "system",
                "content": "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.",
            },
            {
                "role": "user",
                "content": f"""
Database Engine:
{engine}

Database Schema:
{data[schema_name]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
Please directly output the SQL query in the form of ```sql ... ``` without any thinking process.
""".strip(),
            },
        ]
    elif message_type == "arctic_text_to_sql_r1_qwen3-sft-cot":
        messages = [
            {
                "role": "system",
                "content": "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.",
            },
            {
                "role": "user",
                "content": f"""
Database Engine:
{engine}

Database Schema:
{data[schema_name]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
Please output the SQL query in the form of ```sql ... ```.
""".strip(),
            },
        ]
    elif message_type == "only_user_turn":
        messages = [
            {
                "role": "user",
                "content": data[schema_name],
            },
        ]
    elif message_type == "auto_complete_prompt":
        # This message type is designed for autocomplete training data (e.g., test.csv)
        # Input format: JSON array with two objects:
        #   1. {"type": "text", "text": "<sql_history>...SQL history..."} - provides context
        #   2. {"type": "text", "text": "...SQL query with <fillMe> placeholder..."} - query to complete
        # Process: Extract context and query, split around <fillMe>, render Qwen3-Coder FIM template
        # Parse the INPUT data (should be JSON string from CSV)
        input_data_raw = data.get("INPUT", "")
        try:
            input_data = json.loads(input_data_raw)
        except (json.JSONDecodeError, TypeError):
            # Fallback if not JSON
            input_data = [{"type": "text", "text": str(input_data_raw)}]

        # Extract context and query with <fillMe> from separate objects
        context_text = ""
        fillme_text = ""

        for item in input_data:
            if isinstance(item, dict) and item.get("type") == "text":
                text_content = item.get("text", "")

                # Check if this contains <sql_history> - this is the SQL history context
                if "<sql_history>" in text_content:
                    # Strip both opening and closing tags to get the raw content
                    context_text = text_content.replace("<sql_history>", "").replace("</sql_history>", "").strip()
                # Check if this contains <fillMe> (case insensitive) - this is the completion point
                elif re.search(r'<fillme>', text_content, re.IGNORECASE):
                    fillme_text = text_content
                # If neither <sql_history> nor <fillMe>, check if it's the first item (context) or second item (query)
                else:
                    # If we haven't found context yet and this is the first item, treat it as context
                    if not context_text and not fillme_text:
                        context_text = text_content
                    # If we have context but no fillMe text, treat this as the query
                    elif context_text and not fillme_text:
                        fillme_text = text_content

        # Extract prefix and suffix directly from fillMe text
        prefix_text = ""
        suffix_text = ""
        
        if fillme_text:
            if re.search(r'<fillme>', fillme_text, re.IGNORECASE):
                # Split the text around <fillMe> (case insensitive) to get prefix and suffix directly
                parts = re.split(r'<fillme>', fillme_text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    prefix_text = parts[0]
                    suffix_text = parts[1]
                else:
                    # Fallback if multiple <fillMe> or other issues
                    prefix_text = re.sub(r'<fillme>', '', fillme_text, flags=re.IGNORECASE)
                    suffix_text = ""
            else:
                # If no <fillMe> found, treat the whole text as prefix
                prefix_text = fillme_text
                suffix_text = ""
        
        # If no fillMe text found, use the first text item as fallback
        elif input_data:
            for item in input_data:
                if isinstance(item, dict) and item.get("type") == "text":
                    prefix_text = item.get("text", "")
                    suffix_text = ""
                    break

        sql_history = context_text
        ddl_context = ""  # placeholder for now
        last_few_executed_context = ""  # placeholder for now
        file_content = ""  # placeholder for now (entire worksheet content)
        
        # Render the static system prompt (no context)
        system_message = render_autocomplete_system_prompt()
        
        # Render the user message with all context
        # Order: sql_history, ddl_context, last_few_executed_context, file_content, FIM tokens
        user_message = render_autocomplete_user_prompt(
            prefix=prefix_text,
            suffix=suffix_text,
            sql_history=sql_history,
            ddl_context=ddl_context,
            last_few_executed_context=last_few_executed_context,
            file_content=file_content,
        )
        
        # Log the rendered template once for debugging
        global _template_logged
        if not _template_logged:
            logger.info("=== Training record example starts ===")
            logger.info(f"System: {system_message[:200]}...")
            logger.info(f"User: {user_message}")
            logger.info("=== Training record example ends ===")
            _template_logged = True

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    else:
        raise ValueError(f"message_type: {message_type} is not supported!")
    # TODO: we may want to modify this part for snowflake sql dataset
    return messages
