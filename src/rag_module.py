# src/rag_module.py
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class RagRetriever:
    def __init__(self, kb_path="data/mental_health_knowledge.json",
                 embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 index_path="data/kb_index.faiss", normalize=True):
        self.kb_path = kb_path
        self.index_path = index_path
        self.model = SentenceTransformer(embed_model_name)
        self.normalize = normalize
        self.docs, self.doc_texts = self._load_kb()
        self.index = self._load_or_build_index()

    def _load_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        texts = [f"{d['title']}\n{d['content']}" for d in docs]
        return docs, texts

    def _embed(self, texts):
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if self.normalize:
            faiss.normalize_L2(embs)
        return embs

    def _load_or_build_index(self):
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            return index
        embs = self._embed(self.doc_texts)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine if normalized
        index.add(embs)
        faiss.write_index(index, self.index_path)
        return index

    def search(self, query, k=3):
        q_emb = self._embed([query])
        scores, idxs = self.index.search(q_emb, k)  # (1, k)
        results = []
        for rank, i in enumerate(idxs[0].tolist()):
            if i < 0 or i >= len(self.docs): 
                continue
            results.append({
                "rank": rank,
                "score": float(scores[0][rank]),
                "doc": self.docs[i],
                "text": self.doc_texts[i]
            })
        return results
