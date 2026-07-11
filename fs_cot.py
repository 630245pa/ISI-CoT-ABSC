
import json
import os
import random
import xml.etree.ElementTree as ET
from typing import Literal
from tqdm import tqdm

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

VALID_LABELS = {"positive", "neutral", "negative"}


def _parse_label(raw: str) -> str:
    for token in raw.lower().split():
        token = token.strip(".,!?:")
        if token in VALID_LABELS:
            return token
    return "neutral"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _load_train_xml(path: str) -> list[dict]:
    tree = ET.parse(path)
    root = tree.getroot()
    records = []
    for sentence in root.findall("sentence"):
        text = sentence.find("text").text
        for aspect in sentence.findall(".//aspectTerm"):
            impl = aspect.get("implicit_sentiment")
            if impl is None or impl == "None":
                continue
            records.append({
                "text":               text,
                "term":               aspect.get("term"),
                "polarity":           aspect.get("polarity"),
                "implicit_sentiment": impl,
            })
    return records



class DemoBank:
    def __init__(self, train_xml: str, cache_path: str, N: int = 5):
        self.train_xml  = train_xml
        self.cache_path = cache_path
        self.N          = N
        self.records: list[dict] = []  
        self._centroid_sbert = None

    
    def build(self, predictor, model: str) -> None:
        raw_data = _load_train_xml(self.train_xml)
        print(f"[DemoBank] {len(raw_data)} training instances loaded.")

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                cached = json.load(f)
            if len(cached) == len(raw_data):
                self.records = cached
                print(f"[DemoBank] Cache found -- skipping chain generation.")
                print(f"[DemoBank] {len(self.records)} pre-built chains loaded "
                    f"from '{os.path.basename(self.cache_path)}'.")
                print(f"[DemoBank] Proceeding directly to FS-CoT prediction.")
                return
            elif len(cached) > 0:
                self.records = cached
                completed_texts = {(r["text"], r["term"]) for r in cached}
                raw_data = [
                    item for item in raw_data
                    if (item["text"], item["term"]) not in completed_texts
                ]
                print(f"[DemoBank] Partial cache found -- "
                    f"{len(self.records)} done, {len(raw_data)} remaining.")
            else:
                print(f"[DemoBank] Empty cache found. Starting fresh.")
        if len(raw_data) == 0:          
                print(f"[DemoBank] All chains already cached. Skipping generation.")
                return                       

        print(f"[DemoBank] Generating CoT chains "
            f"(N={self.N} per instance) -- this runs once and is cached.")

        for i, item in enumerate(tqdm(raw_data, desc="Building chains", unit="instance")):
            chain, source = self._build_chain(item, predictor, model)  
            self.records.append({
                "text":     item["text"],
                "term":     item["term"],
                "polarity": item["polarity"],
                "chain":    chain,
                "source":   source,   
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(raw_data):  
                self._save_cache()

        print(f"[DemoBank] Done. {len(self.records)} chains saved to "
            f"'{self.cache_path}'.")

        from collections import Counter
        sources = Counter(r.get("source", "cached") for r in self.records)
        total = len(self.records)
        print(f"\n[DemoBank] Chain source distribution:")
        for src, count in sources.items():
            print(f"  {src:>10}: {count:>4}  ({count/total*100:.1f}%)")


    def _build_chain(self, item: dict, predictor, model: str) -> tuple[str, str]:

        text, term, gold = item["text"], item["term"], item["polarity"]
        chains = []

        prompt = (
            f'Given the sentence: "{text}"\n'
            f'Aspect term: "{term}"\n'
            f"Let's think step by step about the sentiment polarity "
            f'towards "{term}".'
        )

        for _ in range(self.N):
            try:
                response = predictor.client.chat(
                    model=model,
                    options={"temperature": 0.7, "num_predict": 200},
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_chain = response["message"]["content"].strip()
            except Exception:
                raw_chain = predictor._chat(prompt, model, max_tokens=200)

            predicted = _parse_label(raw_chain)
            chains.append({"chain": raw_chain, "predicted": predicted})

        r_plus = [c["chain"] for c in chains if c["predicted"] == gold]

        if len(r_plus) > 1:
            return self._select_by_centroid(r_plus), "centroid"

        if len(r_plus) == 1:
            return r_plus[0], "direct"


        chain = predictor._chat(
            f'Given the sentence: "{text}"\n'
            f'Aspect term: "{term}"\n'
            f'The correct sentiment polarity towards "{term}" is {gold}.\n'
            f'Explain step by step why the polarity is {gold}.',
            model,
            max_tokens=200,
        )
        return chain, "fallback"


    def _select_by_centroid(self, chains: list[str]) -> str:

        if not _SBERT_AVAILABLE:
            print("[DemoBank] sentence-transformers not available -- "
                  "falling back to first chain in R+.")
            return chains[0]

        
        if self._centroid_sbert is None:
            self._centroid_sbert = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self._centroid_sbert.encode(
            chains, convert_to_numpy=True, normalize_embeddings=True
        )
        centroid = embeddings.mean(axis=0)
        sims     = [_cosine_sim(e, centroid) for e in embeddings]
        best_idx = int(np.argmax(sims))
        return chains[best_idx]


    def _save_cache(self) -> None:
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.records, f, indent=2)


class Retriever:
    def __init__(
        self,
        demo_bank: DemoBank,
        method: Literal["random", "bm25", "simcse"] = "random",
        K: int = 4,
    ):
        if method not in {"random", "bm25", "simcse"}:
            raise ValueError(
                f"Unknown retrieval method '{method}'. "
                f"Choose from: random, bm25, simcse."
            )
        self.demo_bank = demo_bank
        self.method    = method
        self.K         = K

        
        self._bm25_index   = None
        self._sbert_model  = None
        self._sbert_embeds = None   

        if method == "bm25":
            self._build_bm25_index()
        elif method == "simcse":
            self._build_simcse_index()

   
    def retrieve(self, text: str, term: str) -> list[dict]:
        if self.method == "random":
            return self._retrieve_random()
        elif self.method == "bm25":
            return self._retrieve_bm25(text, term)
        else:  # simcse
            return self._retrieve_simcse(text, term)

    def _retrieve_random(self) -> list[dict]:
        return random.sample(
            self.demo_bank.records,
            min(self.K, len(self.demo_bank.records)),
        )


    def _build_bm25_index(self) -> None:
        if not _BM25_AVAILABLE:
            raise ImportError(
                "rank_bm25 is required for BM25 retrieval. "
                "Install with: pip install rank-bm25"
            )
        corpus = [
            (r["text"] + " " + r["term"]).lower().split()
            for r in self.demo_bank.records
        ]
        self._bm25_index = BM25Okapi(corpus)
        print("[Retriever] BM25 index built.")

    def _retrieve_bm25(self, text: str, term: str) -> list[dict]:
        query   = (text + " " + term).lower().split()
        scores  = self._bm25_index.get_scores(query)
        indices = np.argsort(scores)[::-1][: self.K]
        return [self.demo_bank.records[i] for i in indices]


    def _build_simcse_index(self) -> None:
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SimCSE retrieval. "
                "Install with: pip install sentence-transformers"
            )
        self._sbert_model  = SentenceTransformer("all-MiniLM-L6-v2")
        docs               = [r["text"] + " " + r["term"]
                               for r in self.demo_bank.records]
        self._sbert_embeds = self._sbert_model.encode(
            docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        print("[Retriever] SimCSE index built.")

    def _retrieve_simcse(self, text: str, term: str) -> list[dict]:
        query_embed = self._sbert_model.encode(
            [text + " " + term],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        sims    = self._sbert_embeds @ query_embed
        indices = np.argsort(sims)[::-1][: self.K]
        return [self.demo_bank.records[i] for i in indices]



class FsCoT:
    

    def __init__(
        self,
        predictor,
        train_xml:  str,
        retrieval:  Literal["random", "bm25", "simcse"] = "random",
        K:          int = 4,
        N:          int = 5,
        cache_dir:  str = None,
    ):
        self.predictor = predictor
        self.retrieval = retrieval
        self.K         = K
        self.N         = N

        
        base_name  = os.path.splitext(os.path.basename(train_xml))[0]
        cache_dir  = cache_dir or os.path.dirname(train_xml)
        cache_path = os.path.join(cache_dir, f"{base_name}_chains_N{N}.json")

        self._demo_bank  = DemoBank(train_xml, cache_path, N=N)
        self._retriever: Retriever = None  

   
    def build(self, model: str) -> None:
        self._demo_bank.build(self.predictor, model)
        self._retriever = Retriever(
            self._demo_bank, method=self.retrieval, K=self.K
        )
        print(f"[FsCoT] Ready -- retrieval={self.retrieval}, K={self.K}, "
              f"N={self.N}, pool size={len(self._demo_bank.records)}.")

    def predict(self, text: str, term: str, model: str) -> dict:

        if self._retriever is None:
            raise RuntimeError(
                "FsCoT.build() must be called before predict()."
            )

       
        demos = self._retriever.retrieve(text, term)


        demo_block = self._format_demos(demos)

        reasoning = self.predictor._chat(
            demo_block
            + f'Now analyse the following:\n'
              f'Sentence: "{text}"\n'
              f'Aspect term: "{term}"\n'
              f"Let's think step by step about the sentiment polarity "
              f'towards "{term}".',
            model,
            max_tokens=200,
        )

        raw = self.predictor._chat(
            f'Sentence: "{text}"\n'
            f'Aspect term: "{term}"\n\n'
            f'Reasoning:\n{reasoning}\n\n'
            f'Therefore, the sentiment polarity towards "{term}" is '
            f'(one word -- positive, neutral, or negative):',
            model,
            max_tokens=10,
        )

        return {
            "predicted":  _parse_label(raw),
            "reasoning":  reasoning,
            "demos_used": [
                {"text": d["text"], "term": d["term"], "polarity": d["polarity"]}
                for d in demos
            ],
        }

    
    @staticmethod
    def _format_demos(demos: list[dict]) -> str:
        
        lines = []
        for i, d in enumerate(demos, start=1):
            lines.append(f"Example {i}:")
            lines.append(f'Sentence: "{d["text"]}"')
            lines.append(f'Aspect term: "{d["term"]}"')
            lines.append(f'Reasoning: {d["chain"]}')
            lines.append(f'Polarity: {d["polarity"]}')
            lines.append("")   
        lines.append("")     
        return "\n".join(lines)
