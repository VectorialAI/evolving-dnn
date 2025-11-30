import json
from statistics import median

with open("experiments/experiment3/experiment_summary.json") as f:
    data = json.load(f)

individuals = data["individuals"]
tokens = [individuals[k]["evaluation"]["metrics"]["total_tokens_processed"] for k in individuals if "total_tokens_processed" in individuals[k].get("evaluation", {}).get("metrics", {})]
total_params = [individuals[k]["evaluation"]["param_count"] for k in individuals if "param_count" in individuals[k].get("evaluation", {})]

print(f"Avg tokens: {sum(tokens) / len(tokens):,.0f}, Median: {median(tokens):,}")
print(f"Avg params: {sum(total_params) / len(total_params):,.0f}, Median: {median(total_params):,}")
print(f"Average tokens per param: {sum(tokens) / sum(total_params):,.2f}, Median: {median(tokens) / median(total_params):,.2f}")