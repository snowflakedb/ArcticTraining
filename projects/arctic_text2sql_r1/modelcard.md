## Overview
Arctic-Text2SQL-R1-7B is a 7-billion-parameter Text-to-SQL model fine-tuned using Group Relative Policy Optimization (GRPO) with a simple execution-based reward signal. It converts natural language questions into executable SQL queries.

## Code & Resources

- **BIRD Benchmark Evaluation**
  The evaluation scripts and examples for the BIRD benchmark are available in our GitHub repo:
  [snowflakedb/ArcticTraining – projects](https://github.com/snowflakedb/ArcticTraining/tree/main/projects)

## Key Features

- **Lightweight RL formulation**: Uses only execution correctness and syntax validity as rewards.
- **State-of-the-art performance**: Achieves 68.9% execution accuracy on BIRD-dev and 68.5% on BIRD-test, with an average of 57.2% across six benchmarks (BIRD, Spider, Spider2.0, Spider-DK, EHRSQL, ScienceBenchmark)
- **Efficiency**: Outperforms many 70B+ models with only 7B parameters.

## Intended Use

This model is designed for:

- Interactive natural language interfaces to relational databases.
- Data analytics tools enabling non-technical users to query databases.

### Not intended for:
- Generation of non-SQL text or free-form natural language tasks.
- Production systems without validation, especially in safety-critical domains.

| Benchmark        | Dev/Test Accuracy |
| ---------------- | ----------------- |
| BIRD-dev         | 68.9%             |
| BIRD-test        | 68.5%             |
| Spider-test      | 88.8%             |
| Spider2.0-DK     | 15.6%             |
| EHRSQL           | 36.7%             |
| ScienceBenchmark | 51.8%             |
| **Average**      | **57.2%**         |

## Ethical Considerations
- Avoid using for private or sensitive data without proper oversight.
- Validate generated SQL to prevent data leakage or unauthorized access.
