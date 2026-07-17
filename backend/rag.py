"""
rag.py — Retrieval-Augmented Generation layer for the KFED AI Entrepreneur Advisor.

This is a real retrieval engine (TF-IDF + cosine similarity) built from scratch in
pure Python (no external ML dependencies), so the demo runs anywhere with just
the standard library. It indexes the KFED program knowledge base (kb/programs.json)
which was built from real, published KFED programme pages (khalifafund.ae), not
invented data.

Swap point for a real embeddings model: replace `_tokenize` + the TF-IDF vectors
in KnowledgeBase.__init__ with calls to an embedding API (e.g. Anthropic/OpenAI/
Voyage embeddings) and cosine-similarity over those vectors instead. The public
interface (KnowledgeBase.search) would not need to change.
"""

import json
import math
import os
import re
from collections import Counter, defaultdict

# Minimal bilingual stopword list (EN + AR) so common function words don't
# dominate the similarity score.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "is",
    "are", "this", "that", "by", "as", "it", "at", "be", "your", "you", "our",
    "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "و", "أو", "ما", "لا",
    "أن", "إن", "التي", "الذي", "كل", "بين", "عبر",
}

_TOKEN_RE = re.compile(r"[A-Za-zء-ي]+", re.UNICODE)


def _tokenize(text):
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class KnowledgeBase:
    """TF-IDF index over KFED programme documents, with structured filters
    (stage_fit / sector_fit) that act as retrieval boosts on top of the raw
    text similarity — this mirrors a metadata-filtered vector search."""

    def __init__(self, kb_path=None):
        if kb_path is None:
            kb_path = os.path.join(os.path.dirname(__file__), "..", "kb", "programs.json")
        with open(kb_path, encoding="utf-8") as f:
            self.docs = json.load(f)

        self._doc_tokens = []
        for doc in self.docs:
            text = " ".join([
                doc.get("name_en", ""), doc.get("name_ar", ""),
                doc.get("description_en", ""), doc.get("description_ar", ""),
                doc.get("category", ""), " ".join(doc.get("sector_fit", [])),
                " ".join(doc.get("stage_fit", [])),
            ])
            self._doc_tokens.append(_tokenize(text))

        self._build_index()

    def _build_index(self):
        n_docs = len(self._doc_tokens)
        df = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                df[term] += 1

        self._idf = {
            term: math.log((1 + n_docs) / (1 + freq)) + 1
            for term, freq in df.items()
        }

        self._doc_vectors = []
        for tokens in self._doc_tokens:
            self._doc_vectors.append(self._vectorize(tokens))

    def _vectorize(self, tokens):
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            idf = self._idf.get(term)
            if idf:
                vec[term] = (1 + math.log(count)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    @staticmethod
    def _cosine(vec_a, vec_b):
        if len(vec_b) < len(vec_a):
            vec_a, vec_b = vec_b, vec_a
        return sum(w * vec_b.get(term, 0.0) for term, w in vec_a.items())

    def search(self, query_text, stage=None, sector=None, top_k=5):
        """Retrieve the top_k KFED programmes most relevant to query_text,
        boosted by stage/sector metadata match. Returns list of
        (doc, score) tuples, grounded in real KB documents."""
        query_vec = self._vectorize(_tokenize(query_text))
        scored = []
        for doc, doc_vec in zip(self.docs, self._doc_vectors):
            score = self._cosine(query_vec, doc_vec)

            if stage and stage in doc.get("stage_fit", []):
                score += 0.35
            if sector and sector in doc.get("sector_fit", []):
                score += 0.35
            if sector and "general" in doc.get("sector_fit", []) and sector not in doc.get("sector_fit", []):
                score += 0.05

            if score > 0:
                scored.append((doc, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get(self, doc_id):
        for doc in self.docs:
            if doc["id"] == doc_id:
                return doc
        return None


if __name__ == "__main__":
    kb = KnowledgeBase()
    for query, stage, sector in [
        ("I have a greenhouse farming idea and need funding", "startup", "agri"),
        ("عندي فكرة تصنيع وأحتاج شهادة قيمة مضافة محلية", "growth", "manufacturing"),
    ]:
        print("\nQuery:", query)
        for doc, score in kb.search(query, stage=stage, sector=sector, top_k=3):
            print(f"  {score:.3f}  {doc['name_en']}")
