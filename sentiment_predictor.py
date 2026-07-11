# sentiment_predictor.py
import ollama
import spacy
import os
#from groq import Groq
from dotenv import load_dotenv
import time
from fs_cot import FsCoT ###here for fs_cot

load_dotenv(dotenv_path=".env")  # add this before Groq is initialized


VALID_LABELS = {"positive", "neutral", "negative"}
nlp = spacy.load("en_core_web_sm")

CONLLU_EXPLANATION = (
    "The CoNLL-U format has the following columns:\n"
    "- ID: position of the word in the sentence\n"
    "- TEXT: the word as it appears in the sentence\n"
    "- LEMMA: the base/dictionary form of the word\n"
    "- POS: part-of-speech tag (e.g. NOUN, ADJ, VERB, ADV)\n"
    "- TAG: fine-grained part-of-speech tag (e.g. NN, JJ, VBZ)\n"
    "- HEAD: the ID of the governing word that the current word depends on\n"
    "- DEPREL: the syntactic dependency relation to the head word "
    "(e.g. nsubj=nominal subject, amod=adjectival modifier, "
    "advmod=adverbial modifier, ROOT=root of the sentence)\n"
)


class SentimentPredictor:
    def __init__(self, model: str = "llama3.2"):
        self.client = ollama.Client()          # connects to localhost:11434
        #self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))##change 2
        self.default_model = model
        self.fs_cot_runner = FsCoT(  ##here for fs_cot
            self,
            train_xml="/Users/pranathi/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Masters/Thesis/After_meeting/THOR-ISA-main/data/laptops/Laptops_Train_v2_Implicit_Labeled.xml",
            #train_xml="/Users/pranathi/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Masters/Thesis/After_meeting/THOR-ISA-main/data/restaurants/Restaurants_Train_v2_Implicit_Labeled.xml",
            retrieval="bm25",   # or "random" / "simcse"
            K=3, N=5,
        )
        self._token_log    = {          # ← ADD THIS
        "calls":      0,
        "prompt":     0,
        "completion": 0,
        "total":      0,
    }
    def _fs_cot_predict(self, text, term, model):   ###here for fs_cot
        if self.fs_cot_runner._retriever is None:
            self.fs_cot_runner.build(model=model)
        return self.fs_cot_runner.predict(text, term, model)

    def predict(self, item: dict, method: str = "vanilla", cache: dict = None, model: str = None) -> dict:
        
        model = model or self.default_model
        text, term = item["text"], item["term"]

        METHOD_MAP = {
            "vanilla": self.vanilla,
            "zs_cot": self.zs_cot,
            "zs_scot": self.zs_scot,
            "zs_hcot": self.zs_hcot,
            "fs_cot": lambda t, te, m: self._fs_cot_predict(t, te, m),
            "zs_cot_si":  lambda t, te, m: self.zs_cot_si(t, te, m, cot_cache=cache),
            "zs_hcot_si": lambda t, te, m: self.zs_hcot_si(t, te, m, hcot_cache=cache),
            "zs_scot_si": lambda t, te, m: self.zs_scot_si(t, te, m, scot_cache=cache),
            "vanilla_si": lambda t, te, m: self.vanilla_si(t, te, m,  vanilla_cache=cache),
            "fs_cot_si":  lambda t, te, m: self.fs_cot_si(t, te, m,   fs_cache=cache),

        }

        if method not in METHOD_MAP:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Choose from: {list(METHOD_MAP.keys())}")

        result = METHOD_MAP[method](text, term, model)
        result["gold"] = item["polarity"]
        result["text"] = text
        result["term"] = term
        result["method"] = method
        return result


    def _chat(self, prompt: str, model: str, max_tokens: int = 150) -> str:  ####for OLLAMA!! 
        response = self.client.chat(
            model=model,
            options={"temperature": 0.0, "num_predict": max_tokens},
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    

    def _parse_label(self, raw: str) -> str:
        for token in raw.lower().split():
            token = token.strip(".,!?:")
            if token in VALID_LABELS:
                return token
        return "neutral"

    def _get_conllu(self, text: str) -> str:
        doc = nlp(text)
        lines = ["ID\tTEXT\tLEMMA\tPOS\tTAG\tHEAD\tDEPREL"]
        for token in doc:
            lines.append(
                f"{token.i+1}\t{token.text}\t{token.lemma_}\t"
                f"{token.pos_}\t{token.tag_}\t{token.head.i+1}\t{token.dep_}"
            )
        return "\n".join(lines)
    

    def token_summary(self) -> None:
        log = self._token_log
        print("\n── Token Usage Summary ──────────────────────────────")
        print(f"  Total calls:        {log['calls']:>8,}")
        print(f"  Prompt tokens:      {log['prompt']:>8,}")
        print(f"  Completion tokens:  {log['completion']:>8,}")
        print(f"  Total tokens:       {log['total']:>8,}")
        print(f"  Avg tokens/call:    {log['total']//log['calls'] if log['calls'] else 0:>8,}")
        print(f"  Daily limit:        {'500,000':>8}")
        print(f"  Remaining (approx): {max(0, 500000 - log['total']):>8,}")
        print("─────────────────────────────────────────────────────")

    def vanilla(self, text: str, term: str, model: str) -> dict:
        prompt = (
            f'Given the sentence: "{text}"\n'
            f'What is the sentiment polarity towards the aspect term "{term}"?\n'
            f'Answer with exactly one word: positive, neutral, or negative.'
        )
        raw = self._chat(prompt, model, max_tokens=10)
        return {"predicted": self._parse_label(raw)}
    

    def zs_cot(self, text: str, term: str, model: str) -> dict:
        # C1 — reasoning chain
        reasoning = self._chat(
            f'Given the sentence: "{text}"\n'
            f'What is the sentiment polarity towards the aspect term "{term}"?\n'
            f"Let's think step by step.",
            model, max_tokens=200
        )

        # C2 — extract label conditioned on reasoning
        raw = self._chat(
            f'Given the sentence: "{text}"\n'
            f'What is the sentiment polarity towards the aspect term "{term}"?\n'
            f"Let's think step by step.\n"
            f"{reasoning}\n\n"
            f"Therefore, the answer is one word only (positive, neutral, or negative):",
            model, max_tokens=10
        )

        return {
            "predicted": self._parse_label(raw),
            "reasoning": reasoning,
        }

    def zs_scot(self, text: str, term: str, model: str) -> dict:
        # C1 — opinion extraction
        opinion = self._chat(
            f'Given the sentence: "{text}"\n'
f'Aspect term: "{term}"\n'
f'What is the opinion or sentiment signal expressed towards "{term}"? '
f'If no explicit opinion word is present, reason from context and common sense '
f'to identify what the sentence implies about "{term}". '
f'Answer in one or two sentences.',
            model, max_tokens=150
            )
        
#         C1 — opinion extraction

        # C2 — justification
        justification = self._chat(
            f'Given the sentence: "{text}"\n'
            f'Aspect term: "{term}"\n'
            f'Opinion or contextual cue: "{opinion}"\n\n'
f'Based on this, what does common sense tell us about whether the speaker '
f'views "{term}" positively, negatively, or neutrally, and why?\n'
f'Be concise.',
            model, max_tokens=150
        )

        # C3 — final polarity
        raw = self._chat(
            f'Given the sentence: "{text}"\n'
            f'Aspect term: "{term}"\n'
            f'Opinion or contextual cue: "{opinion}"\n'
            f'Justification: "{justification}"\n\n'
            f'Based on the opinion and justification, what is the sentiment polarity '
            f'towards "{term} in sentence "{text}". "?\n'        ##in sentence "{text}". was absent in the first run!!
            f'Answer with exactly one word (positive, neutral, or negative):',
            model, max_tokens=5
        )

        return {
            "predicted":     self._parse_label(raw),
            "opinion":       opinion,
            "justification": justification,
        }

    
    def zs_hcot(self, text: str, term: str, model: str) -> dict:
        conllu = self._get_conllu(text)

        # Full context — only for C1
        syntax_context_full = (
            f'Sentence: "{text}"\n'
            f'Aspect term: "{term}"\n\n'
            f'Syntactic dependency information (CoNLL-U format):\n{conllu}\n\n'
            f'Format explanation:\n{CONLLU_EXPLANATION}\n'
        )

        # Short context — X only, for C2 and C3
        sentence_context = (
            f'Sentence: "{text}"\n'
            f'Aspect term: "{term}"\n\n'
        )

        # C1 — P(R1 | X, S, E, Q1) ✓
        syntactic_analysis = self._chat(
            syntax_context_full +
            f'Based on the syntactic dependency information above, analyze the '
            f'syntactic relationships related to "{term}" in the sentence. '
            f'Identify which words modify, describe, or are syntactically related to "{term}".\n'
            f'Answer in one or two sentences.',
            model, max_tokens=150
        )

        # C2 — P(R2 | X, R1, Q2) 
        opinion = self._chat(
            sentence_context +
            f'Syntactic analysis: {syntactic_analysis}\n\n'
            f'Considering the syntactic context of "{term}", '
            f'what is the speaker\'s opinion towards "{term}"?\n'
            f'Answer in one or two sentences.',
            model, max_tokens=150
        )

        # C3 — P(R3 | X, R2, Q3) 
        raw = self._chat(
            sentence_context +
            f'Speaker\'s opinion: {opinion}\n\n'
            f'Based on the speaker\'s opinion towards "{term}", '
            f'what is the sentiment polarity?\n'
            f'Answer with exactly one word (positive, neutral, or negative):',
            model, max_tokens=5
        )

        return {
            "predicted":          self._parse_label(raw),
            "conllu":             conllu,
            "syntactic_analysis": syntactic_analysis,
            "opinion":            opinion,
        }

    def vanilla_si(self, text: str, term: str, model: str,
               max_iterations: int = 2,
               vanilla_cache: dict = None) -> dict:
        implicit_note = (
    f'Note: the sentiment towards "{term}" may be implicit — '
    f'inferred from contextual cues rather than expressed through '
    f'explicit opinion words.'
)

        cached = vanilla_cache.get((text, term)) if vanilla_cache is not None else None

        if cached is not None:
            predicted = cached["predicted"]
        else:
            initial   = self.vanilla(text, term, model)
            predicted = initial["predicted"]

        history = []

        for t in range(max_iterations):

            feedback_prompt = (
                f'Sentence: "{text}"\n'
                f'Aspect term: "{term}"\n'
                f'Predicted polarity: {predicted}\n\n'
                f'{implicit_note}\n\n'   
                f'Evaluate the predicted polarity "{predicted}" towards "{term}" '
                f'against two criteria:\n\n'
                f'Criterion 1 — Aspect grounding: Is "{predicted}" specifically about '
                f'"{term}", or does it reflect the overall sentence sentiment or a '
                f'different aspect?\n\n'
                f'Criterion 2 — Evidence consistency: Is there a cue in the sentence '
    f'that justifies "{predicted}" towards "{term}"?\n\n'
                f'If either criterion is violated in a way that would change the polarity '
                f'label, name the criterion, describe the error, and prescribe a correction.\n\n'
                f'End with exactly one tag and nothing after it:\n'
                f'[STOP] if both criteria are satisfied.\n'
                f'[CONTINUE] if there is a clear, specific error that would change the label.'
            )

            feedback = self._chat(feedback_prompt, model, max_tokens=300)

            history.append({
                "iteration": t,
                "predicted": predicted,
                "feedback":  feedback,
            })

            
            if "[STOP]" in feedback:
                break

            history_context = "".join(
                f'--- Iteration {h["iteration"]} ---\n'
                f'Prediction: {h["predicted"]}\n'
                f'Feedback: {h["feedback"]}\n\n'
                for h in history
            )

            raw = self._chat(
                f'Sentence: "{text}"\n'
                f'Aspect term: "{term}"\n\n'
                f'Below is the full history of predictions and feedback.\n'
                f'Do NOT reintroduce errors already identified below.\n\n'
                f'{history_context}'
                f'Based on the feedback above, what is the corrected sentiment '
                f'polarity towards "{term}"?\n'
                f'Answer with exactly one word (positive, neutral, or negative):',
                model, max_tokens=5
            )
            predicted = self._parse_label(raw)

        return {
            "predicted":  predicted,
            "history":    history,
            "iterations": len(history),
        }



    
    def zs_cot_si(self, text, term, model,
              max_iterations=2,
              cot_cache: dict = None):   # add cache parameter
        cached = cot_cache.get((text, term)) if cot_cache is not None else None

        if cached is not None:
            predicted = cached["predicted"]
            reasoning = cached["reasoning"]
        else:
            initial   = self.zs_cot(text, term, model)
            predicted = initial["predicted"]
            reasoning = initial["reasoning"]


        history = []  

        for t in range(max_iterations):


            feedback_prompt = (
    f'Sentence: "{text}"\n'
    f'Aspect term: "{term}"\n\n'
    f'A two-step process was used to predict the sentiment polarity '
    f'towards "{term}":\n'
    f'  Step 1 (Reasoning): A free-form chain-of-thought reasoning was '
    f'generated.\n'
    f'  Step 2 (Prediction): Based on that reasoning, a final polarity '
    f'label was assigned.\n\n'
    f'Current reasoning (Step 1):\n{reasoning}\n\n'
    f'Current prediction (Step 2): {predicted}\n\n'
    f'Evaluate the output against exactly two criteria:\n\n'
    f'Criterion 1 — Aspect grounding: Does the reasoning specifically '
     f'engage with aspect "{term}" '
    
    f'The reasoning must be anchored to "{term}" throughout.\n\n'
    f'  Criterion 2 — Internal consistency: Does the predicted polarity '
    f'"{predicted}" follow from what is actually argued in the reasoning? '
    f'Evaluate only whether the label is consistent with the reasoning '
    f'as written — do not independently re-evaluate the sentence.\n\n'
    f'If you identify a violation of either criterion, name the criterion, '
    f'describe the specific mistake, and prescribe exactly what a corrected '
    f'version should express.\n\n'
    f'You MUST end your response with exactly one of these two tokens '
f'and nothing after it:\n'
f'[STOP] — both criteria are satisfied and the label is correct.\n'
f'[CONTINUE] — at least one criterion is violated and the label would change.\n'
f'Choose based only on whether the label would change.'
)
            feedback = self._chat(feedback_prompt, model, max_tokens=300)

            history.append({
                "iteration": t,
                "reasoning": reasoning,
                "predicted": predicted,
                "feedback":  feedback,
            })

           
            if "[STOP]" in feedback:
                break
             

            history_context = ""
            for h in history:
                history_context += (
                    f'--- Iteration {h["iteration"]} ---\n'
                    f'Reasoning: {h["reasoning"]}\n'
                    f'Prediction: {h["predicted"]}\n'
                    f'Feedback: {h["feedback"]}\n\n'
                )

            reasoning = self._chat(
                f'Sentence: "{text}"\n'
                f'Aspect term: "{term}"\n\n'
                f'Below is the full history of reasoning chains and feedback.\n'
                f'Do NOT reintroduce errors that have already been identified.\n\n'
                f'{history_context}'
                f'Now provide a revised reasoning chain that addresses all feedback above.\n'
                f"Let's think step by step.",
                model, max_tokens=300
            )
            raw = self._chat(
                f'Sentence: "{text}"\n'
                f'Aspect term: "{term}"\n\n'
                f'Reasoning:\n{reasoning}\n\n'
                f'Therefore, the answer is one word only (positive, neutral, or negative):',
                model, max_tokens=5
            )
            predicted = self._parse_label(raw)

        return {
            "predicted":  predicted,
            "reasoning":  reasoning,
            "history":    history,
            "iterations": len(history) 
        }
    



    def zs_hcot_si(self, text: str, term: str, model: str,
                   max_iterations: int = 2,
                   hcot_cache: dict = None) -> dict:
    
        cached = hcot_cache.get((text, term)) if hcot_cache is not None else None


        if cached is not None:
            r1        = cached["syntactic_analysis"]
            r2        = cached["opinion"]
            predicted = cached["predicted"]
            conllu    = cached["conllu"]
        else:
            conllu    = self._get_conllu(text)
            initial   = self.zs_hcot(text, term, model)
            r1        = initial["syntactic_analysis"]
            r2        = initial["opinion"]
            predicted = initial["predicted"]
 
       
        syntax_context_full = (
    f'Sentence: "{text}"\n'
    f'Aspect term: "{term}"\n\n'
    f'Syntactic parse (CoNLL-U):\n{conllu}\n\n'
    f'{CONLLU_EXPLANATION}'
)

        syntax_context = (
    f'Sentence: "{text}"\n'
    f'Aspect term: "{term}"\n\n'
)
 
        history = []  
 
        for t in range(max_iterations):
 
            
            feedback = self._chat(
                syntax_context_full +
                f'\nThe following structured output was produced for "{term}":\n\n'
                f'R1 (syntactic analysis): {r1}\n'
                f'R2 (opinion extraction): {r2}\n'
                f'Prediction (y_hat): {predicted}\n\n'
                f'Evaluate each step strictly against the CoNLL-U parse above:\n\n'
                f'Your sole task is to identify errors that would change the polarity label. '
f'Do not flag stylistic issues or weaker-but-valid inferences.\n\n'
                f'Step 1 — Does R1 correctly identify the syntactic relationships '
                f'in the parse that are most relevant to "{term}"? '
                f'Verify every claimed dependency relation (head, deprel, POS) '
                f'exists in the parse table. Flag any relation that is absent, '
                f'misread, or misattributed.\n\n'
                f'Step 2 — Does R2 derive the opinion from the '
                f'syntactic context established in R1? '
    f'Flag any lexical cue introduced in R2 that is not grounded in R1.\n\n'
                f'Step 3 — Is y_hat fully consistent with the opinion stated in R2? '
                f'A polarity that does not follow from R2 is an error even if R2 '
                f'itself is correct.\n\n'
                f'Identify the EARLIEST step where the model\'s use of the '
                f'syntactic evidence breaks down. Correcting a downstream step '
                f'while leaving an upstream error intact cannot produce a coherent '
                f'chain.\n\n'
                f'After your brief 2-3 sentence critique, end your response in a new line with EXACTLY ONE '
                f'of the following tags '
                f'and nothing after it:\n'
                f'[STEP 1] — R1 misreads or misapplies the parse.\n'
                f'[STEP 2] — R1 is correct but R2 is not grounded in R1\'s '
                f'syntactic evidence.\n'
                f'[STEP 3] — R1 and R2 are correct but y_hat is inconsistent '
                f'with R2.\n'
                f'[STOP]   — all three steps are correct; no revision needed.\n',
                model, max_tokens=500
            )
 
            last_line = feedback.strip().split('\n')[-1].upper()
            if "[STOP]" in last_line:
                restart_from = None
            elif "[STEP 1]" in last_line or "[STEP1]" in last_line:
                restart_from = 1
            elif "[STEP 2]" in last_line or "[STEP2]" in last_line:
                restart_from = 2
            elif "[STEP 3]" in last_line or "[STEP3]" in last_line:
                restart_from = 3
            else:
                restart_from = 3  
            history.append({
                "iteration":    t,
                "r1":           r1,
                "r2":           r2,
                "predicted":    predicted,
                "feedback":     feedback,
                "restart_from": restart_from,  
            })

            if restart_from is None:
                break
 
            history_block = "".join(
                f'[Iteration {h["iteration"]}]\n'
                f'R1: {h["r1"]}\n'
                f'R2: {h["r2"]}\n'
                f'Prediction: {h["predicted"]}\n'
                f'Feedback: {h["feedback"]}\n\n'
                for h in history
            )
            no_regress = (
                f'Prior attempts and feedback (do NOT reintroduce any error '
                f'already identified below):\n{history_block}'
            )
 
            if restart_from == 1:
                r1 = self._chat(
                    syntax_context_full +
                    f'\n{no_regress}\n'
                    f'The feedback above identified an error in Step 1 (R1).\n'
                    f'Re-read the CoNLL-U parse carefully and produce a corrected R1:\n'
                    f'Identify the syntactic relationships in the parse that are most '
                    f'relevant to "{term}". For each relationship, cite the exact '
                    f'dependency relation (head token, deprel label) from the parse '
                    f'table. Where sentiment is not explicitly encoded in the parse, you may draw on lexical '
                    f'knowledge, but anchor that inference to a specific token that appears in a dependency relation with "{term}".\n'
                    f'Answer in one or two sentences.',
                    model, max_tokens=200
                )
 
            if restart_from in (1, 2):
                r2 = self._chat(
                    syntax_context +
                    f'\n{no_regress}\n'
                    f'Updated R1 (syntactic analysis): {r1}\n\n'
                    f'{"The feedback identified an error in Step 2 (R2)." if restart_from == 2 else "R1 has been corrected; now produce a corrected R2."}\n'
                    f'Based strictly on the syntactic context established in R1 above, '
                    f'what is the speaker\'s opinion towards "{term}"? '
                    f'Do not introduce lexical cues that are not grounded in R1\'s '
                    f'syntactic evidence. If the sentiment is implicit, anchor the '
                    f'inference explicitly in the dependency relations identified in R1.\n'
                    f'Answer in one or two sentences.',
                    model, max_tokens=200
                )
 
            raw = self._chat(
                syntax_context +
                f'\n{no_regress}\n'
                f'Updated R1 (syntactic analysis): {r1}\n'
                f'Updated R2 (opinion extraction): {r2}\n\n'
                f'{"The feedback identified an error in Step 3." if restart_from == 3 else "R1 and R2 have been revised."}\n'
                f'Based solely on the opinion stated in R2 above, what is the '
                f'sentiment polarity towards "{term}"?\n'
                f'Answer with exactly one word: positive, neutral, or negative.',
                model, max_tokens=5
            )
            predicted = self._parse_label(raw)
        return {
            "predicted":          predicted,
            "conllu":             conllu,
            "syntactic_analysis": r1,
            "opinion":            r2,
            "history":            history,
            "iterations":         len(history),
        }
    



    def zs_scot_si(self, text: str, term: str, model: str,
               max_iterations: int = 2,
               scot_cache: dict = None) -> dict:
       
        implicit_note = (
            f'Note: the sentiment towards "{term}" may be implicit — '
            f'inferred from contextual cues rather than expressed through '
            f'explicit opinion words.'
        )
        cached = scot_cache.get((text, term)) if scot_cache is not None else None

        if cached is not None:
            opinion       = cached["opinion"]
            justification = cached["justification"]
            predicted     = cached["predicted"]
        else:
            initial       = self.zs_scot(text, term, model)
            opinion       = initial["opinion"]
            justification = initial["justification"]
            predicted     = initial["predicted"]

        history = []  

        for t in range(max_iterations):

            
            feedback = self._chat(
    f'Sentence: "{text}"\n'
    f'Aspect term: "{term}"\n'
    f'O: {opinion}\n'
    f'J: {justification}\n'
    f'y_hat: {predicted}\n\n'
    f'Check for label-changing errors ONLY — ignore style or quality.\n'
    f'Check 1: Does O identify a cue grounded in the sentence for "{term}" specifically?\n'
    f'Check 2: Does J connect O to a polarity direction in a defensible way?\n'
    f'Check 3: Does y_hat match the polarity direction in J?\n\n'
    f'A "defensible" inference need not be the only valid one — '
    f'just plausible given the sentence.\n\n'
    f'After your brief 2-3 sentence critique, end your response in a new line with EXACTLY ONE '
    f'of the following tags '
    f'End with ONE tag, nothing after:\n'
    f'[STOP] if all checks pass or errors would not change the label.\n'
    f'[STEP 1] if O is factually wrong for "{term}".\n'
    f'[STEP 2] if J directly contradicts O (not merely weaker).\n'
    f'[STEP 3] if y_hat contradicts J.\n',
                model, max_tokens=350
)
            


            
            last_line = feedback.strip().split('\n')[-1].upper()
            if "[STOP]" in last_line:
                restart_from = None
            elif "[STEP 1]" in last_line or "[STEP1]" in last_line:
                restart_from = 1
            elif "[STEP 2]" in last_line or "[STEP2]" in last_line:
                restart_from = 2
            elif "[STEP 3]" in last_line or "[STEP3]" in last_line:
                restart_from = 3
            else:
                restart_from = 3  
                
            history.append({
                "iteration":    t,
                "opinion":      opinion,
                "justification": justification,
                "predicted":    predicted,
                "feedback":     feedback,
                "restart_from": restart_from,
            })

            if restart_from is None:
                break

          
            history_block = "".join(
                f'[Iteration {h["iteration"]}]\n'
                f'O: {h["opinion"]}\n'
                f'J: {h["justification"]}\n'
                f'Prediction: {h["predicted"]}\n'
                f'Feedback: {h["feedback"]}\n\n'
                for h in history
            )
            no_regress = (
                f'Prior attempts and feedback (do NOT reintroduce any error '
                f'already identified below):\n{history_block}'
            )


            if restart_from == 1:
                opinion = self._chat(
                    f'Sentence: "{text}"\n'
                    f'Aspect term: "{term}"\n\n'
                    f'{implicit_note}\n\n'
                    f'{no_regress}\n'
                    f'The feedback above identified an error in Step 1 (O).\n'
                    f'Produce a corrected opinion expression: what is the opinion '
                    f'or sentiment signal expressed or implied towards "{term}" '
                    f'in the sentence? If no explicit opinion word is present, '
                    f'reason from context and common sense to identify the '
                    f'strongest available cue for "{term}" specifically.\n'
                    f'Answer in one or two sentences.',
                    model, max_tokens=150
                )

            if restart_from in (1, 2):
                justification = self._chat(
                    f'Sentence: "{text}"\n'
                    f'Aspect term: "{term}"\n\n'
                    f'{implicit_note}\n\n'
                    f'{no_regress}\n'
                    f'Updated opinion expression (O): {opinion}\n\n'
                    f'{"The feedback identified an error in Step 2 (J)." if restart_from == 2 else "O has been corrected; now produce a corrected J."}\n'
                    f'Based strictly on the opinion expression O above, what does '
                    f'common sense tell us about whether the speaker views "{term}" '
                    f'positively, negatively, or neutrally, and why? '
                    f'Do not introduce a new opinion cue not grounded in O.\n'
                    f'Answer in one or two sentences.',
                    model, max_tokens=150
                )

    
            raw = self._chat(
                f'Sentence: "{text}"\n'
                f'Aspect term: "{term}"\n\n'
                f'{no_regress}\n'
                f'Updated opinion expression (O): {opinion}\n'
                f'Updated sentiment justification (J): {justification}\n\n'
                f'{"The feedback identified an error in Step 3." if restart_from == 3 else "O and J have been revised."}\n'
                f'Based solely on O and J above, what is the sentiment polarity '
                f'towards "{term}"?\n'
                f'Answer with exactly one word: positive, neutral, or negative.',
                model, max_tokens=5
            )
            predicted = self._parse_label(raw)

        return {
            "predicted":     predicted,
            "opinion":       opinion,
            "justification": justification,
            "history":       history,
            "iterations":    len(history),
        }
    


    def fs_cot_si(self, text: str, term: str, model: str,
              max_iterations: int = 2,
              fs_cache: dict = None) -> dict:
        cached = fs_cache.get((text, term)) if fs_cache is not None else None

        if cached is not None:
            predicted = cached["predicted"]
            reasoning = cached["reasoning"]
        else:
            initial   = self._fs_cot_predict(text, term, model)
            predicted = initial["predicted"]
            reasoning = initial["reasoning"]

        history = []  
        for t in range(max_iterations):

            feedback = self._chat(
                f'Sentence: "{text}"\n'
                f'Aspect term: "{term}"\n\n'
                f'A two-step process was used to predict the sentiment polarity '
                f'towards "{term}":\n'
                f'  Step 1 (Reasoning): A few-shot chain-of-thought reasoning was '
                f'generated.\n'
                f'  Step 2 (Prediction): Based on that reasoning, a final polarity '
                f'label was assigned.\n\n'
                f'Current reasoning (Step 1):\n{reasoning}\n\n'
                f'Current prediction (Step 2): {predicted}\n\n'
                f'Evaluate the output against exactly two criteria:\n\n'
                f'  Criterion 1 — Aspect grounding: Does the reasoning specifically '
                f'engage with aspect "{term}"'
                f'The reasoning must be anchored to "{term}" throughout.\n\n'
                f'  Criterion 2 — Internal consistency: Does the predicted polarity '
                f'"{predicted}" follow from what is actually argued in the reasoning? '
                f'Evaluate only whether the label is consistent with the reasoning '
                f'as written — do not independently re-evaluate the sentence.\n\n'
                f'If you identify a violation of either criterion, name the criterion, '
                f'describe the specific mistake, and prescribe exactly what a corrected '
                f'version should express.\n\n'
                f'You MUST end your response with exactly one of these two tokens '
                f'and nothing after it:\n'
f'[STOP] — both criteria are satisfied and the label is correct.\n'
f'[CONTINUE] — at least one criterion is violated and the label would change.\n'
f'Choose based only on whether the label would change.',
                model, max_tokens=300
            )

            history.append({
                "iteration": t,
                "reasoning": reasoning,
                "predicted": predicted,
                "feedback":  feedback,
            })

            if "[STOP]" in feedback:
                break

            history_context = "".join(
                f'--- Iteration {h["iteration"]} ---\n'
                f'Reasoning: {h["reasoning"]}\n'
                f'Prediction: {h["predicted"]}\n'
                f'Feedback: {h["feedback"]}\n\n'
                for h in history
            )

            reasoning = self._chat(
                f'Sentence: "{text}"\n'
                f'Aspect term: "{term}"\n\n'
                f'Below is the full history of reasoning chains and feedback.\n'
                f'Do NOT reintroduce errors that have already been identified.\n\n'
                f'{history_context}'
                f'Now provide a revised reasoning chain that addresses all feedback above.\n'
                f"Let's think step by step.",
                model, max_tokens=300
            )

            raw = self._chat(
                f'Sentence: "{text}"\n'
                f'Aspect term: "{term}"\n\n'
                f'Reasoning:\n{reasoning}\n\n'
                f'Therefore, the answer is one word only (positive, neutral, or negative):',
                model, max_tokens=5
            )
            predicted = self._parse_label(raw)

        return {
            "predicted":  predicted,
            "reasoning":  reasoning,
            "history":    history,
            "iterations": len(history),
        }