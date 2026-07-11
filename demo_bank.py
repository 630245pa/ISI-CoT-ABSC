
from eval import Evaluator
import json

with open("/Users/pranathi/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Masters/Thesis/code_draft/Thesis_draft/results_mistral_rest/mistral_zs_scot_si_checkpoint.json") as f:
    results = json.load(f)

Evaluator().iteration_stats(results, label="zs_scot_si (mistral partial)")