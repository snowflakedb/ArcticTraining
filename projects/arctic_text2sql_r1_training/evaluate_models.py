#!/usr/bin/env python3
"""
Evaluate and Compare Models: Baseline vs GRPO-Trained

This script tests both the baseline model and GRPO-trained model
on the same test queries to demonstrate improvement.

Usage:
    python evaluate_models.py

Results:
    - Execution accuracy comparison
    - Sample outputs side-by-side
    - Performance metrics
"""

import torch
import sqlite3
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Tuple
import re
from tqdm import tqdm


class ModelEvaluator:
    """Evaluate Text-to-SQL models"""

    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"):
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def load_baseline_model(self):
        """Load baseline model (before GRPO training)"""
        print("\n" + "="*60)
        print("Loading BASELINE model...")
        print("="*60)

        self.baseline_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print(f"✅ Baseline model loaded: {self.base_model_name}")

    def load_trained_model(self, checkpoint_path: str = "grpo_3b_trained"):
        """Load GRPO-trained model"""
        print("\n" + "="*60)
        print("Loading TRAINED model (after GRPO)...")
        print("="*60)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Load LoRA adapters
        try:
            self.trained_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            self.trained_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            print(f"✅ Trained model loaded from: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Could not load trained model from {checkpoint_path}")
            print(f"   Error: {e}")
            print(f"   Will skip trained model evaluation")
            self.trained_model = None
            self.trained_tokenizer = None

    def generate_sql(self, model, tokenizer, prompt: str, num_samples: int = 5) -> List[str]:
        """Generate SQL queries from model"""
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        candidates = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                sql = text[len(prompt):].strip()
                candidates.append(sql)

        return candidates

    def extract_sql(self, text: str) -> str:
        """Extract SQL from model output"""
        # Look for SQL in code blocks
        sql_match = re.search(r'```sql\n(.*?)\n```', text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        # Look for SELECT, INSERT, UPDATE, DELETE statements
        sql_patterns = [
            r'(SELECT\s+.*?(?:;|$))',
            r'(INSERT\s+.*?(?:;|$))',
            r'(UPDATE\s+.*?(?:;|$))',
            r'(DELETE\s+.*?(?:;|$))',
        ]

        for pattern in sql_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().rstrip(';')

        # Fallback: return as-is
        return text.strip()

    def test_sql_execution(self, sql: str, db_path: str) -> Tuple[bool, str]:
        """
        Test if SQL executes successfully

        Returns:
            (success, result/error_message)
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return True, str(result)
        except Exception as e:
            return False, str(e)

    def compare_results(self, pred_result: str, gold_result: str) -> bool:
        """Compare query results"""
        return pred_result == gold_result

    def evaluate_on_test_set(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate both models on test set

        Args:
            test_data: List of test examples with 'prompt', 'gold_sql', 'database_path'

        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATION STARTED")
        print("="*60)

        results = {
            'baseline': {'correct': 0, 'executable': 0, 'total': 0, 'examples': []},
            'trained': {'correct': 0, 'executable': 0, 'total': 0, 'examples': []},
        }

        for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
            prompt = example['prompt']
            gold_sql = example['gold_sql']
            db_path = example.get('database_path', 'test.db')

            # Create simple test database if doesn't exist
            self._ensure_test_db(db_path)

            # Get gold result
            gold_success, gold_result = self.test_sql_execution(gold_sql, db_path)

            # Evaluate baseline model
            baseline_sqls = self.generate_sql(
                self.baseline_model,
                self.baseline_tokenizer,
                prompt,
                num_samples=3
            )

            baseline_best = self._evaluate_candidates(
                baseline_sqls, gold_sql, gold_result, db_path
            )

            results['baseline']['total'] += 1
            if baseline_best['executable']:
                results['baseline']['executable'] += 1
            if baseline_best['correct']:
                results['baseline']['correct'] += 1

            # Evaluate trained model (if available)
            if self.trained_model is not None:
                trained_sqls = self.generate_sql(
                    self.trained_model,
                    self.trained_tokenizer,
                    prompt,
                    num_samples=3
                )

                trained_best = self._evaluate_candidates(
                    trained_sqls, gold_sql, gold_result, db_path
                )

                results['trained']['total'] += 1
                if trained_best['executable']:
                    results['trained']['executable'] += 1
                if trained_best['correct']:
                    results['trained']['correct'] += 1
            else:
                trained_best = None

            # Store example for reporting
            results['baseline']['examples'].append({
                'prompt': prompt[:100] + "...",
                'gold_sql': gold_sql,
                'generated_sql': baseline_best['sql'],
                'executable': baseline_best['executable'],
                'correct': baseline_best['correct'],
            })

            if trained_best:
                results['trained']['examples'].append({
                    'prompt': prompt[:100] + "...",
                    'gold_sql': gold_sql,
                    'generated_sql': trained_best['sql'],
                    'executable': trained_best['executable'],
                    'correct': trained_best['correct'],
                })

        return results

    def _evaluate_candidates(
        self,
        candidates: List[str],
        gold_sql: str,
        gold_result: str,
        db_path: str
    ) -> Dict:
        """Evaluate list of candidate SQLs and return best one"""
        best = {
            'sql': candidates[0] if candidates else "",
            'executable': False,
            'correct': False
        }

        for candidate in candidates:
            sql = self.extract_sql(candidate)
            success, result = self.test_sql_execution(sql, db_path)

            if success:
                best['executable'] = True
                best['sql'] = sql

                if self.compare_results(result, gold_result):
                    best['correct'] = True
                    return best  # Found correct one, return immediately

        return best

    def _ensure_test_db(self, db_path: str):
        """Create simple test database if doesn't exist"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create simple table for testing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    department TEXT,
                    salary INTEGER
                )
            """)

            # Insert test data
            cursor.execute("SELECT COUNT(*) FROM employees")
            if cursor.fetchone()[0] == 0:
                test_data = [
                    (1, 'Alice', 'Engineering', 120000),
                    (2, 'Bob', 'Engineering', 110000),
                    (3, 'Carol', 'Sales', 95000),
                ]
                cursor.executemany(
                    "INSERT INTO employees VALUES (?, ?, ?, ?)",
                    test_data
                )

            conn.commit()
            conn.close()
        except:
            pass

    def print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        # Calculate metrics
        baseline_exec_acc = (results['baseline']['executable'] / results['baseline']['total'] * 100) if results['baseline']['total'] > 0 else 0
        baseline_correct_acc = (results['baseline']['correct'] / results['baseline']['total'] * 100) if results['baseline']['total'] > 0 else 0

        print("\n📊 BASELINE MODEL (Before GRPO):")
        print(f"   Executable SQLs: {results['baseline']['executable']}/{results['baseline']['total']} ({baseline_exec_acc:.1f}%)")
        print(f"   Correct Results: {results['baseline']['correct']}/{results['baseline']['total']} ({baseline_correct_acc:.1f}%)")

        if self.trained_model is not None:
            trained_exec_acc = (results['trained']['executable'] / results['trained']['total'] * 100) if results['trained']['total'] > 0 else 0
            trained_correct_acc = (results['trained']['correct'] / results['trained']['total'] * 100) if results['trained']['total'] > 0 else 0

            print("\n📊 TRAINED MODEL (After GRPO):")
            print(f"   Executable SQLs: {results['trained']['executable']}/{results['trained']['total']} ({trained_exec_acc:.1f}%)")
            print(f"   Correct Results: {results['trained']['correct']}/{results['trained']['total']} ({trained_correct_acc:.1f}%)")

            print("\n📈 IMPROVEMENT:")
            exec_improvement = trained_exec_acc - baseline_exec_acc
            correct_improvement = trained_correct_acc - baseline_correct_acc
            print(f"   Executable: {exec_improvement:+.1f} percentage points")
            print(f"   Correct: {correct_improvement:+.1f} percentage points")

        # Show example comparisons
        print("\n" + "="*60)
        print("EXAMPLE COMPARISONS")
        print("="*60)

        for i in range(min(3, len(results['baseline']['examples']))):
            example_baseline = results['baseline']['examples'][i]

            print(f"\nExample {i+1}:")
            print(f"Prompt: {example_baseline['prompt']}")
            print(f"\nGold SQL:")
            print(f"  {example_baseline['gold_sql']}")

            print(f"\n📌 BASELINE:")
            print(f"  Generated: {example_baseline['generated_sql'][:100]}")
            print(f"  Executable: {'✅' if example_baseline['executable'] else '❌'}")
            print(f"  Correct: {'✅' if example_baseline['correct'] else '❌'}")

            if self.trained_model is not None and i < len(results['trained']['examples']):
                example_trained = results['trained']['examples'][i]
                print(f"\n📌 TRAINED:")
                print(f"  Generated: {example_trained['generated_sql'][:100]}")
                print(f"  Executable: {'✅' if example_trained['executable'] else '❌'}")
                print(f"  Correct: {'✅' if example_trained['correct'] else '❌'}")

            print("-" * 60)


def create_test_data() -> List[Dict]:
    """Create test dataset"""
    return [
        {
            'prompt': "Question: How many employees are in the Engineering department?\nTable: employees(id, name, department, salary)\n\nSQL:",
            'gold_sql': "SELECT COUNT(*) FROM employees WHERE department = 'Engineering'",
            'database_path': 'test_eval.db'
        },
        {
            'prompt': "Question: What is the average salary in Engineering?\nTable: employees(id, name, department, salary)\n\nSQL:",
            'gold_sql': "SELECT AVG(salary) FROM employees WHERE department = 'Engineering'",
            'database_path': 'test_eval.db'
        },
        {
            'prompt': "Question: List all employee names\nTable: employees(id, name, department, salary)\n\nSQL:",
            'gold_sql': "SELECT name FROM employees",
            'database_path': 'test_eval.db'
        },
        {
            'prompt': "Question: How many unique departments are there?\nTable: employees(id, name, department, salary)\n\nSQL:",
            'gold_sql': "SELECT COUNT(DISTINCT department) FROM employees",
            'database_path': 'test_eval.db'
        },
        {
            'prompt': "Question: Who earns more than 100000?\nTable: employees(id, name, department, salary)\n\nSQL:",
            'gold_sql': "SELECT name FROM employees WHERE salary > 100000",
            'database_path': 'test_eval.db'
        },
    ]


def main():
    """Main evaluation script"""
    print("="*60)
    print("MODEL EVALUATION: Baseline vs GRPO-Trained")
    print("="*60)

    # Initialize evaluator
    evaluator = ModelEvaluator(base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct")

    # Load models
    evaluator.load_baseline_model()
    evaluator.load_trained_model(checkpoint_path="grpo_3b_trained")

    # Create test data
    print("\n📋 Creating test dataset...")
    test_data = create_test_data()
    print(f"✅ Created {len(test_data)} test examples")

    # Run evaluation
    results = evaluator.evaluate_on_test_set(test_data)

    # Print results
    evaluator.print_results(results)

    # Save results
    with open('evaluation_results.json', 'w') as f:
        # Remove examples for cleaner JSON
        save_results = {
            'baseline': {k: v for k, v in results['baseline'].items() if k != 'examples'},
            'trained': {k: v for k, v in results['trained'].items() if k != 'examples'},
        }
        json.dump(save_results, f, indent=2)

    print("\n✅ Results saved to: evaluation_results.json")
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
