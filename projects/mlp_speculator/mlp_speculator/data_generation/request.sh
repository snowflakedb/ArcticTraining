curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    "prompt": "What are other names for Mount Denali?"
  }'