

import xml.etree.ElementTree as ET


class DataLoader:
   

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = []

    def load_data(self) -> list[dict]:
       
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        self.data = []
        for sentence in root.findall("sentence"):
            sentence_id = sentence.get("id")
            text = sentence.find("text").text

            for aspect in sentence.findall(".//aspectTerm"):
                self.data.append({
                    "sentence_id":        sentence_id,
                    "text":               text,
                    "term":               aspect.get("term"),
                    "polarity":           aspect.get("polarity"),
                    "from":               aspect.get("from"),
                    "to":                 aspect.get("to"),
                    "opinion_words":      aspect.get("opinion_words"),
                    "implicit_sentiment": aspect.get("implicit_sentiment"),
                })
        before = len(self.data)
        self.data = [
            d for d in self.data
            if d["implicit_sentiment"] is not None
            and d["implicit_sentiment"] != "None"]
        removed = before - len(self.data)
        print(f"Removed {removed} entries with implicit_sentiment=None. "
            f"Remaining: {len(self.data)}")   

        return self.data

    def print_stats(self) -> None:
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return

        print(f"Total aspect terms: {len(self.data)}\n")

        for label, key, values in [
            ("Implicit sentiment", "implicit_sentiment", ["True", "False", "None", None]),
            ("Polarity",           "polarity",           ["positive", "negative", "neutral"]),
        ]:
            print(f"--- {label} ---")
            for v in values:
                count = sum(1 for d in self.data if d[key] == v)
                label_str = str(v) if v is not None else "None (missing)"
                print(f"  {label_str:>10}: {count}")
            print()

    def inspect_first_sentence(self) -> None:
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        sentence = root[0]
        for elem in sentence.iter():
            print(f"Tag: {elem.tag}, Attribs: {elem.attrib}, Text: {elem.text}")