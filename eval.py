import json
import os
import threading
import time
from collections import defaultdict, Counter
from tqdm.auto import tqdm                          # auto handles Jupyter vs terminal
from sklearn.metrics import f1_score, classification_report
from sentiment_predictor import SentimentPredictor
from concurrent.futures import ThreadPoolExecutor, as_completed

class Evaluator:
   
    LABELS = ["positive", "neutral", "negative"]

    
    def evaluate(self, results: list[dict]) -> dict:
        
        gold      = [r["gold"]      for r in results]
        predicted = [r["predicted"] for r in results]

        n_total   = len(gold)
        n_correct = sum(g == p for g, p in zip(gold, predicted))

        return {
            "accuracy":     n_correct / n_total if n_total else 0.0,
            "macro_f1":     f1_score(gold, predicted, labels=self.LABELS,
                                     average="macro",    zero_division=0),
            "weighted_f1":  f1_score(gold, predicted, labels=self.LABELS,
                                     average="weighted", zero_division=0),
            "n_total":      n_total,
            "n_correct":    n_correct,
        }

   
    def evaluate_by_split(self, results: list[dict],
                      splits: bool = True) -> dict:
        
        out = {"overall": self.evaluate(results)}
        if not splits:
            return out
        buckets = defaultdict(list)
        for r in results:
            key = "ISE" if r.get("implicit_sentiment") == "True" else "ESE"
            buckets[key].append(r)

        for split_name, split_results in buckets.items():
            out[split_name] = self.evaluate(split_results)

        return out

    
    def report(self, results: list[dict]) -> str:
        gold      = [r["gold"]      for r in results]
        predicted = [r["predicted"] for r in results]
        return classification_report(gold, predicted,
                                     labels=self.LABELS, zero_division=0)


    def compare(self, results_dict: dict[str, list[dict]]) -> dict:
        comparison = {}
        for method_name, results in results_dict.items():
            comparison[method_name] = self.evaluate(results)
        return comparison

  
    def print_evaluate(self, results: list[dict], label: str = "") -> None:
        metrics = self.evaluate(results)
        header  = f"── {label} ──" if label else "── Results ──"
        print(header)
        print(f"  Accuracy:     {metrics['accuracy']:.4f}  "
              f"({metrics['n_correct']}/{metrics['n_total']})")
        print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")

    def print_compare(self, results_dict: dict[str, list[dict]]) -> None:
        comparison = self.compare(results_dict)
        print(f"\n{'Method':<20} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
        print("─" * 55)
        for method, metrics in comparison.items():
            print(f"{method:<20} "
                  f"{metrics['accuracy']:>10.4f} "
                  f"{metrics['macro_f1']:>10.4f} "
                  f"{metrics['weighted_f1']:>12.4f}")

    def print_by_split(self, results: list[dict],
                   label: str = "",
                   splits: bool = True) -> None:
        split_data = self.evaluate_by_split(results, splits=splits)
        header = f"── {label} ──" if label else "── Split Results ──"
        print(header)
        for split_name, metrics in split_data.items():
            print(f"\n  [{split_name}]  n={metrics['n_total']}")
            print(f"    Accuracy:    {metrics['accuracy']:.4f}")
            print(f"    Macro F1:    {metrics['macro_f1']:.4f}")
            print(f"    Weighted F1: {metrics['weighted_f1']:.4f}")

    def iteration_stats(self, results: list[dict], label: str = "") -> None:
        si_results = [r for r in results if "iterations" in r]

        if not si_results:
            print(f"[{label or 'results'}] No iteration data found "
                f"— only SI methods track iterations.")
            return

        counts = Counter(r["iterations"] for r in si_results)
        total  = len(si_results)

        header = f"── {label} iteration stats ──" if label else "── Iteration stats ──"
        print(header)
        print(f"  Total items with iteration data: {total}\n")

        for n_iter in sorted(counts.keys()):
            count = counts[n_iter]
            pct   = count / total * 100
            bar   = "█" * int(pct / 2)  
            print(f"  {n_iter} iteration(s): {count:>4}  ({pct:5.1f}%)  {bar}")

       
        STEP_TAG_METHODS = {"zs_hcot_si", "zs_scot_si"}

        # get the method name from the first result
        method_name = si_results[0].get("method", "") if si_results else ""

        if method_name in STEP_TAG_METHODS:
            step_counts = Counter()
            for r in si_results:
                for entry in r.get("history", []):
                    tag = entry.get("restart_from")
                    step_counts[tag] += 1

            print(f"\n  Step tag distribution (across all iterations):")
            tag_labels = {1: "[STEP 1]", 2: "[STEP 2]", 3: "[STEP 3]", None: "[STOP/malformed]"}
            for tag in [1, 2, 3, None]:
                count = step_counts[tag]
                pct   = count / sum(step_counts.values()) * 100 if step_counts else 0
                print(f"  {tag_labels[tag]:>20}: {count:>4}  ({pct:5.1f}%)")
    

    def refinement_stats(self, results: list[dict], label: str = "") -> None:
        si_results = [r for r in results if "history" in r]

        if not si_results:
            print(f"[{label or 'results'}] No history data found "
                f"— only SI methods track refinement history.")
            return

        def _refinements(history: list[dict]) -> int:
            n = 0
            for h in history:
                if "restart_from" in h:
                    if h["restart_from"] is not None:
                        n += 1
                elif "[STOP]" not in h.get("feedback", ""):
                    n += 1
            return n

        counts = Counter(_refinements(r["history"]) for r in si_results)
        total  = len(si_results)
        total_refinements = sum(n * c for n, c in counts.items())

        header = f"── {label} refinement stats ──" if label else "── Refinement stats ──"
        print(header)
        print(f"  Items:                {total}")
        print(f"  Total refinements:    {total_refinements}")
        print(f"  Avg refinements/item: {total_refinements/total:.3f}\n")

        for n_refine in sorted(counts.keys()):
            count = counts[n_refine]
            pct   = count / total * 100
            bar   = "█" * int(pct / 2)
            print(f"  {n_refine} refinement(s): {count:>4}  ({pct:5.1f}%)  {bar}")

        



class Runner:
  
    SI_CACHE_MAP = {
            "zs_cot_si":  "zs_cot",
            "zs_hcot_si": "zs_hcot",
            "zs_scot_si": "zs_scot",
            "vanilla_si": "vanilla",
            "fs_cot_si":  "fs_cot",
        }

    def __init__(self, predictor: SentimentPredictor,
                 output_dir: str = "results",
                 model: str = None):
        self.predictor  = predictor
        self.output_dir = output_dir
        self.model      = model or predictor.default_model
        os.makedirs(output_dir, exist_ok=True)

    def run(self, data: list[dict], method: str) -> list[dict]:
        checkpoint_path = self._checkpoint_path(method)

        completed = self._load_checkpoint(checkpoint_path)
        done_ids  = {r["sentence_id"] + "|" + r["term"] for r in completed}
        remaining = [d for d in data
                    if d["sentence_id"] + "|" + d["term"] not in done_ids]

        if completed:
            print(f"[{method}] Resuming — "
                f"{len(completed)} done, {len(remaining)} remaining.")
        else:
            print(f"[{method}] Starting fresh — {len(data)} items.")

        results = list(completed)
        base_method = self.SI_CACHE_MAP.get(method)
        cache = self.load_cache(base_method) if base_method else None

        for item in tqdm(remaining, desc=method, unit="item"):
            try:
                result = self.predictor.predict(item, method=method,
                                                model=self.model,
                                                cache=cache)
                result["llm_model"]          = self.model
                result["implicit_sentiment"] = item["implicit_sentiment"]
                result["sentence_id"]        = item["sentence_id"]
                results.append(result)
                self._save_checkpoint(checkpoint_path, results)
            except Exception as e:
                print(f"\n[{method}] Error on item "
                    f"{item['sentence_id']} / '{item['term']}': {e}")
                self._save_checkpoint(checkpoint_path, results)
                continue

        print(f"[{method}] Done — {len(results)} items.")
        return results


    def run_parallel(self, data: list[dict], method: str,
                    max_workers: int = 4) -> list[dict]:
        checkpoint_path = self._checkpoint_path(method)

        completed = self._load_checkpoint(checkpoint_path)
        done_ids  = {r["sentence_id"] + "|" + r["term"] for r in completed}
        remaining = [d for d in data
                    if d["sentence_id"] + "|" + d["term"] not in done_ids]

        if completed:
            print(f"[{method}] Resuming — "
                f"{len(completed)} done, {len(remaining)} remaining.")
        else:
            print(f"[{method}] Starting fresh — "
                f"{len(data)} items, {max_workers} workers.")

        results    = list(completed)
        lock       = threading.Lock()
        wall_start = time.perf_counter()

        base_method = self.SI_CACHE_MAP.get(method)
        cache = self.load_cache(base_method) if base_method else None

        def process_item(item: dict) -> dict | None:
            try:
                item_start = time.perf_counter()
                result = self.predictor.predict(item, method=method,
                                                model=self.model,
                                                cache=cache)
                result["llm_model"]          = self.model
                result["inference_time"]     = round(time.perf_counter() - item_start, 4)
                result["implicit_sentiment"] = item["implicit_sentiment"]
                result["sentence_id"]        = item["sentence_id"]
                return result
            except Exception as e:
                print(f"\n[{method}] Error on {item['sentence_id']} "
                    f"/ '{item['term']}': {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, item): item
                    for item in remaining}

            with tqdm(as_completed(futures), total=len(remaining),
                    desc=f"{method} (x{max_workers})", unit="item") as pbar:
                for future in pbar:
                    result = future.result()
                    if result is not None:
                        with lock:
                            results.append(result)
                            self._save_checkpoint(checkpoint_path, results)

        wall_time     = round(time.perf_counter() - wall_start, 4)
        n             = len(results)
        wall_per_item = round(wall_time / n, 4) if n else 0.0

        for r in results:
            r["wall_time_total"]    = wall_time
            r["wall_time_per_item"] = wall_per_item

        self._save_checkpoint(checkpoint_path, results)

        print(f"[{method}] Done — {n} items | "
            f"wall-clock: {wall_time:.1f}s | "
            f"per item: {wall_per_item:.2f}s")

        return results


    
   
    def run_all(self, data: list[dict],
                methods: list[str] = None) -> dict[str, list[dict]]:
       
        methods = methods or ["vanilla", "zs_scot", "zs_hcot", "zs_cot_si"]
        all_results = {}

        for method in methods:
            print(f"\n{'─'*50}")
            all_results[method] = self.run(data, method)

        return all_results

   
    def _checkpoint_path(self, method: str) -> str:
        #return os.path.join(self.output_dir, f"{method}_checkpoint.json")
        return os.path.join(self.output_dir, f"{self.model}_{method}_checkpoint.json")

    def _load_checkpoint(self, path: str) -> list[dict]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _save_checkpoint(self, path: str, results: list[dict]) -> None:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    
    def load_results(self, method: str) -> list[dict]:
        """Load a previously completed run from disk."""
        path = self._checkpoint_path(method)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No checkpoint found for method '{method}' at {path}"
            )
        return self._load_checkpoint(path)

    def load_all_results(self, methods: list[str]) -> dict[str, list[dict]]:
        """Load all completed runs from disk."""
        return {m: self.load_results(m) for m in methods}
    

    def load_cache(self, method: str) -> dict:
        path = self._checkpoint_path(method)
        assert os.path.exists(path), (
            f"\n[CHECKPOINT] {method} cache not found at '{path}'.\n"
            f"Run runner.run(data, '{method}') to completion before launching "
            f"{method}_si, otherwise every item will fall back to a live call.\n"
        )
        records = self._load_checkpoint(path)
        assert len(records) > 0, (
            f"\n[CHECKPOINT] {method} checkpoint at '{path}' is empty.\n"
            f"Re-run {method} first.\n"
        )
        cache = {(r["text"], r["term"]): r for r in records}
        print(f"[Runner] {method} cache loaded — {len(cache)} entries.")
        return cache

    def run_all_parallel(self, data: list[dict],
                        methods: list[str] = None,
                        max_workers: int = 4) -> dict[str, list[dict]]:
        methods = methods 
        all_results = {}
        for method in methods:
            print(f"\n{'─'*50}")
            all_results[method] = self.run_parallel(data, method,
                                                    max_workers=max_workers)
        return all_results
    



    