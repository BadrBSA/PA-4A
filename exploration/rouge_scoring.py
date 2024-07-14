from rouge_score import rouge_scorer

with open("../data/summary/HP/fine_tuned.txt", "r", encoding="utf-8") as f:
    fine_tuned_summary = f.read()

with open("../data/summary/HP/base_model.txt", "r", encoding="utf-8") as f:
    base_model_summary = f.read()

with open("../data/summary/HP/target.txt", "r", encoding="utf-8") as f:
    target_summary = f.read()


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

scores = scorer.score(target_summary, fine_tuned_summary)

print(f"ROUGE-1: {scores['rouge1']}")
print(f"ROUGE-2: {scores['rouge2']}")
print(f"ROUGE-L: {scores['rougeL']}")


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

scores = scorer.score(target_summary, base_model_summary)

print(f"ROUGE-1: {scores['rouge1']}")
print(f"ROUGE-2: {scores['rouge2']}")
print(f"ROUGE-L: {scores['rougeL']}")
