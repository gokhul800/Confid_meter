import os
import glob
from typing import List, Dict, Any, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# Optional NVIDIA LLM (safe fallback included)
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False


class RAGWithConfidence:
    """
    Retrieval-Augmented Generation (RAG) system
    with Confidence Calibration Layer.
    """

    def __init__(self, data_folder: str = "./docs"):
        self.data_folder = data_folder

        # ==========================
        # 1. Embedding Model
        # ==========================
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()

        # ==========================
        # 2. FAISS Index
        # ==========================
        self.faiss_index = faiss.IndexFlatIP(self.embed_dim)

        self.chunks = []
        self.sources = []

        # ==========================
        # 3. Optional LLM Setup
        # ==========================
        if NVIDIA_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.llm = ChatNVIDIA(
                model="openai/gpt-oss-20b",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3
            )
        else:
            self.llm = None

        self._ingest_documents()

    # ==========================================
    # DOCUMENT INGESTION
    # ==========================================

    def _read_text_file(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _read_pdf_file(self, filepath: str) -> str:
        text = ""
        reader = PdfReader(filepath)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def _ingest_documents(self, chunk_size: int = 80, overlap: int = 20):
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"Created '{self.data_folder}' folder.")
            return

        files = glob.glob(os.path.join(self.data_folder, "*.*"))
        print(f"Found {len(files)} files for ingestion.")

        for file_path in files:
            filename = os.path.basename(file_path)
            ext = filename.split(".")[-1].lower()

            try:
                if ext == "txt":
                    content = self._read_text_file(file_path)
                elif ext == "pdf":
                    content = self._read_pdf_file(file_path)
                else:
                    continue
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
                continue

            if not content.strip():
                continue

            words = content.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                self.chunks.append(chunk)
                self.sources.append(filename)

        if self.chunks:
            print(f"Generating embeddings for {len(self.chunks)} chunks...")
            embeddings = self.embed_model.encode(self.chunks, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)
            print("Ingestion complete!")

    # ==========================================
    # RETRIEVAL
    # ==========================================

    def retrieve_documents(self, query: str, top_k: int = 3):
        if self.faiss_index.ntotal == 0:
            return [], [], []

        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        similarities, indices = self.faiss_index.search(query_embedding, top_k)

        retrieved_chunks = []
        retrieved_sources = []
        retrieved_scores = []

        for i, idx in enumerate(indices[0]):
            if idx != -1:
                retrieved_chunks.append(self.chunks[idx])
                retrieved_sources.append(self.sources[idx])
                retrieved_scores.append(float(similarities[0][i]))

        return retrieved_chunks, retrieved_sources, retrieved_scores

    # ==========================================
    # ANSWER GENERATION
    # ==========================================

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "No context found."

        combined_context = "\n---\n".join(context_chunks)

        prompt = f"""
        Use ONLY the following context to answer the query.
        If the answer is not in the context, say:
        "I cannot find reliable information for this query."

        Context:
        {combined_context}

        Query: {query}
        Answer:
        """

        # If LLM available
        if self.llm:
            try:
                response = self.llm.invoke(prompt)
                return response.content.strip()
            except:
                return context_chunks[0]

        # Demo-safe fallback (no LLM)
        return context_chunks[0]

    # ==========================================
    # CONFIDENCE CALIBRATION (Balanced Mode)
    # ==========================================

    def compute_confidence(self, scores, context_chunks, answer):

        if not scores or not context_chunks:
            return 0.0, "Low", True

        base_score = float(np.mean(scores))

        variance = float(np.max(scores) - np.min(scores))
        variance_penalty = variance * 0.2

        # Cross-check similarity
        answer_emb = self.embed_model.encode([answer], convert_to_numpy=True)
        faiss.normalize_L2(answer_emb)

        context_text = " ".join(context_chunks)
        context_emb = self.embed_model.encode([context_text], convert_to_numpy=True)
        faiss.normalize_L2(context_emb)

        ans_ctx_sim = float(np.dot(answer_emb, context_emb.T)[0][0])

        mismatch_penalty = 0.0
        if ans_ctx_sim < 0.3:
            mismatch_penalty = 0.2

        word_count = len(context_text.split())
        context_bonus = 0.0
        if word_count > 100:
            context_bonus = min(0.1, (word_count - 100) * 0.001)

        confidence = base_score - variance_penalty - mismatch_penalty + context_bonus
        confidence = max(0.0, min(1.0, confidence))

        BASE_THRESHOLD = 0.25
        should_fallback = base_score < BASE_THRESHOLD

        if confidence > 0.75:
            label = "High"
        elif confidence >= 0.5:
            label = "Medium"
        else:
            label = "Low"

        return confidence, label, should_fallback

    # ==========================================
    # MASTER QUERY
    # ==========================================

    def query(self, user_query: str):

        chunks, sources, scores = self.retrieve_documents(user_query)

        if not chunks:
            return {
                "answer": "No documents available in the knowledge base.",
                "confidence_score": 0.0,
                "confidence_label": "Low",
                "sources": [],
                "similarity_scores": []
            }

        raw_answer = self.generate_answer(user_query, chunks)

        confidence_score, confidence_label, should_fallback = \
            self.compute_confidence(scores, chunks, raw_answer)

        final_answer = raw_answer
        if should_fallback:
            final_answer = "I cannot find reliable information for this query."
            confidence_label = "Low"

        return {
            "answer": final_answer,
            "confidence_score": round(confidence_score, 4),
            "confidence_label": confidence_label,
            "sources": list(set(sources)),
            "similarity_scores": [round(s, 4) for s in scores]
        }


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    import json

    rag_system = RAGWithConfidence(data_folder="./docs")

    q = input("Query :")
    print(f"\nQuerying: '{q}'\n")

    result = rag_system.query(q)
    print(json.dumps(result, indent=2))