import os, pickle, json, gc, re, unicodedata
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
TOP_K = 20
BATCH_SIZE = 32
HYBRID_ALPHA = 0.7
RERANK_TOP_K = 7
CHUNKS_FOLDER = 'chunks_store'
EMBEDDING_MODEL_NAME = "Models/LaBSE"

def init_embedding(model_dir=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        base = Path(__file__).resolve().parent.parent
        model_path = Path(model_dir).expanduser().resolve() if model_dir else base / EMBEDDING_MODEL_NAME
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model folder not found: {model_path}")
        model = SentenceTransformer(str(model_path), trust_remote_code=True, local_files_only=True)
        model.max_seq_length = 512
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
        except:
            pass
        return model
    except:
        return None

def preprocess_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()

def detect_sentence_boundaries(text):
    patterns = [
        r'[।॥][\s\n]+',
        r'[።][\s\n]+',
        r'[។][\s\n]+',
        r'[۔][\s\n]+',
        r'[。！？][\s\n]+',
        r'[\.!?]+[\s\n]+(?=[A-Z\u0900-\u097F])',
    ]
    combined_pattern = '|'.join(patterns)
    sentences = re.split(f'({combined_pattern})', text)
    result = []
    for i in range(0, len(sentences), 2):
        sent = sentences[i]
        if i + 1 < len(sentences):
            sent += sentences[i + 1]
        if sent.strip():
            result.append(sent.strip())
    if not result:
        result = [s.strip() for s in text.split('\n') if s.strip()]
    return result if result else [text]

def create_chunks_streaming(txt_path, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    base = Path(__file__).resolve().parent.parent
    model_path = base / EMBEDDING_MODEL_NAME
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = preprocess_text(f.read())
        sentences = detect_sentence_boundaries(text)
        if not sentences:
            return
        chunk_id = 0
        current_chunk = []
        current_size = 0
        for sent in sentences:
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            sent_size = len(tokens)
            if sent_size > size:
                words = sent.split()
                word_chunk = []
                word_size = 0
                for word in words:
                    word_tokens = tokenizer.encode(word, add_special_tokens=False)
                    if word_size + len(word_tokens) > size and word_chunk:
                        chunk_text = ' '.join(word_chunk)
                        yield {'text': chunk_text, 'chunk_id': chunk_id, 'start_pos': chunk_id * (size - overlap), 'end_pos': (chunk_id + 1) * (size - overlap)}
                        chunk_id += 1
                        overlap_words = word_chunk[-max(1, overlap // 10):]
                        word_chunk = overlap_words
                        word_size = sum(len(tokenizer.encode(w, add_special_tokens=False)) for w in overlap_words)
                    word_chunk.append(word)
                    word_size += len(word_tokens)
                if word_chunk:
                    current_chunk = word_chunk
                    current_size = word_size
                continue
            if current_size + sent_size > size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                yield {'text': chunk_text, 'chunk_id': chunk_id, 'start_pos': chunk_id * (size - overlap), 'end_pos': (chunk_id + 1) * (size - overlap)}
                chunk_id += 1
                overlap_text = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
                    if overlap_size + s_tokens <= overlap:
                        overlap_text.insert(0, s)
                        overlap_size += s_tokens
                    else:
                        break
                current_chunk = overlap_text
                current_size = overlap_size
            current_chunk.append(sent)
            current_size += sent_size
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            yield {'text': chunk_text, 'chunk_id': chunk_id, 'start_pos': chunk_id * (size - overlap), 'end_pos': (chunk_id + 1) * (size - overlap)}
    except:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = preprocess_text(f.read())
        paragraphs = text.split('\n\n')
        step = size - overlap
        chunk_id = 0
        current_text = ''
        for para in paragraphs:
            if len(current_text) + len(para) > size * 4:
                pos = 0
                while pos < len(current_text):
                    end = min(pos + size * 4, len(current_text))
                    chunk = current_text[pos:end].strip()
                    if chunk:
                        yield {'text': chunk, 'chunk_id': chunk_id, 'start_pos': pos, 'end_pos': end}
                        chunk_id += 1
                    pos += step * 4
                current_text = para
            else:
                current_text += '\n\n' + para if current_text else para
        if current_text:
            pos = 0
            while pos < len(current_text):
                end = min(pos + size * 4, len(current_text))
                chunk = current_text[pos:end].strip()
                if chunk:
                    yield {'text': chunk, 'chunk_id': chunk_id, 'start_pos': pos, 'end_pos': end}
                    chunk_id += 1
                pos += step * 4

def save_chunks(pdf_hash, chunks):
    chunk_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_chunks.pkl")
    json_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_chunks.json")
    with open(chunk_file, 'wb') as f:
        pickle.dump(chunks, f)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def load_chunks(pdf_hash):
    chunk_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_chunks.pkl")
    if os.path.exists(chunk_file):
        try:
            with open(chunk_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_vector_store(pdf_hash, index):
    vector_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_vectors.index")
    faiss.write_index(index, vector_file)

def load_vector_store(pdf_hash):
    vector_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_vectors.index")
    if os.path.exists(vector_file):
        try:
            return faiss.read_index(vector_file)
        except:
            return None
    return None

def save_tfidf_index(pdf_hash, tfidf_matrix, tfidf_vectorizer):
    tfidf_matrix_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_tfidf_matrix.npz")
    tfidf_vectorizer_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_tfidf_vectorizer.pkl")
    sp.save_npz(tfidf_matrix_file, tfidf_matrix)
    with open(tfidf_vectorizer_file, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

def load_tfidf_index(pdf_hash):
    tfidf_matrix_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_tfidf_matrix.npz")
    tfidf_vectorizer_file = os.path.join(CHUNKS_FOLDER, f"{pdf_hash}_tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_matrix_file) and os.path.exists(tfidf_vectorizer_file):
        try:
            tfidf_matrix = sp.load_npz(tfidf_matrix_file)
            with open(tfidf_vectorizer_file, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            return tfidf_matrix, tfidf_vectorizer
        except:
            return None, None
    return None, None

def remove_index_and_chunks(pdf_hash):
    files = [f"{pdf_hash}_vectors.index", f"{pdf_hash}_chunks.pkl", f"{pdf_hash}_chunks.json", f"{pdf_hash}_tfidf_matrix.npz", f"{pdf_hash}_tfidf_vectorizer.pkl"]
    for fname in files:
        fpath = os.path.join(CHUNKS_FOLDER, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except:
                pass

def encode_in_batches(texts, embedding_model):
    embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        e = embedding_model.encode(batch, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
        embs.append(e.cpu().numpy())
    return np.vstack(embs).astype('float32')

def create_vector_store_streaming(txt_path, pdf_hash, embedding_model, update_callback=None):
    if not embedding_model:
        return None, []
    try:
        all_chunks = []
        chunk_buffer = []
        index = None
        total_processed = 0
        for chunk in create_chunks_streaming(txt_path):
            all_chunks.append(chunk)
            chunk_buffer.append(chunk['text'])
            if len(chunk_buffer) >= BATCH_SIZE:
                embeddings = encode_in_batches(chunk_buffer, embedding_model)
                if index is None:
                    dim = embeddings.shape[1]
                    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
                ids = np.arange(total_processed, total_processed + len(chunk_buffer))
                index.add_with_ids(embeddings, ids)
                total_processed += len(chunk_buffer)
                if update_callback:
                    update_callback(f"Processed {total_processed} chunks...")
                chunk_buffer = []
                embeddings = None
                gc.collect()
        if chunk_buffer:
            embeddings = encode_in_batches(chunk_buffer, embedding_model)
            if index is None:
                dim = embeddings.shape[1]
                index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            ids = np.arange(total_processed, total_processed + len(chunk_buffer))
            index.add_with_ids(embeddings, ids)
            total_processed += len(chunk_buffer)
            embeddings = None
            gc.collect()
        texts = [c['text'] for c in all_chunks]
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), analyzer='char_wb', min_df=1, max_df=0.85, sublinear_tf=True, norm='l2')
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        if update_callback:
            update_callback(f"Saving {total_processed} chunks to disk...")
        save_chunks(pdf_hash, all_chunks)
        save_vector_store(pdf_hash, index)
        save_tfidf_index(pdf_hash, tfidf_matrix, tfidf_vectorizer)
        return index, all_chunks
    except:
        return None, []

def augment_query(query):
    query = preprocess_text(query)
    variations = [query]
    normalized = re.sub(r'[^\w\s\u0900-\u097F\u0600-\u06FF\u0C00-\u0C7F]', ' ', query)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    if normalized != query and normalized:
        variations.append(normalized)
    words = query.split()
    key_terms = [w for w in words if len(w) > 2]
    if len(key_terms) >= 3:
        focused = ' '.join(key_terms[:5])
        if focused not in variations:
            variations.append(focused)
    return variations

def hybrid_search(query, store, chunks, embedding_model, pdf_hash, top_k=TOP_K):
    if not chunks or store is None:
        return []
    query_variations = augment_query(query)
    query_embs = embedding_model.encode(query_variations, convert_to_tensor=True, normalize_embeddings=True)
    import torch
    query_emb = torch.mean(query_embs, dim=0, keepdim=True).cpu().numpy()
    k = min(top_k * 5, len(chunks))
    dense_scores, dense_ids = store.search(query_emb, k)
    dense_scores = dense_scores[0]
    dense_ids = dense_ids[0]
    if dense_scores.max() > dense_scores.min():
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
    sparse_scores = np.zeros(len(chunks))
    tfidf_matrix, tfidf_vectorizer = load_tfidf_index(pdf_hash)
    if tfidf_vectorizer and tfidf_matrix is not None:
        try:
            all_query_texts = ' '.join(query_variations)
            query_tfidf = tfidf_vectorizer.transform([all_query_texts])
            sparse_sim = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            if sparse_sim.max() > sparse_sim.min():
                sparse_scores = (sparse_sim - sparse_sim.min()) / (sparse_sim.max() - sparse_sim.min())
            else:
                sparse_scores = sparse_sim
        except:
            pass
    final_scores = {}
    for i, idx in enumerate(dense_ids):
        if idx != -1 and idx < len(chunks):
            final_scores[int(idx)] = HYBRID_ALPHA * dense_scores[i]
    top_sparse_k = min(k, len(sparse_scores))
    top_sparse_idx = np.argsort(-sparse_scores)[:top_sparse_k]
    for idx in top_sparse_idx:
        idx_int = int(idx)
        if idx_int < len(chunks):
            sparse_contribution = (1 - HYBRID_ALPHA) * sparse_scores[idx]
            if idx_int in final_scores:
                final_scores[idx_int] += sparse_contribution
            else:
                final_scores[idx_int] = sparse_contribution
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

def rerank_with_cross_encoder(query, candidate_chunks, chunks, embedding_model, top_k=RERANK_TOP_K):
    if not candidate_chunks:
        return []
    query_emb = embedding_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    reranked = []
    for chunk_idx, initial_score in candidate_chunks:
        if chunk_idx >= len(chunks):
            continue
        chunk_text = chunks[chunk_idx]["text"]
        chunk_emb = embedding_model.encode(chunk_text, convert_to_tensor=True, normalize_embeddings=True)
        sim = util.pytorch_cos_sim(query_emb, chunk_emb).item()
        chunk_lower = chunk_text.lower()
        chunk_terms = set(chunk_lower.split())
        if len(query_terms) > 0:
            term_overlap = len(query_terms & chunk_terms) / len(query_terms)
        else:
            term_overlap = 0
        phrase_bonus = 0
        if query_lower in chunk_lower:
            phrase_bonus = 0.3
        final_score = sim * 0.5 + term_overlap * 0.3 + initial_score * 0.1 + phrase_bonus
        if sim < 0.05 and term_overlap < 0.1:
            continue
        reranked.append({'chunk_idx': chunk_idx, 'final_score': final_score, 'sim': sim, 'term_overlap': term_overlap, 'text': chunk_text})
    reranked.sort(key=lambda x: x['final_score'], reverse=True)
    return reranked[:top_k]

def retrieve_chunks(query, store, chunks, embedding_model, k=TOP_K, document_language='english', pdf_hash=None):
    if not embedding_model or not store or not chunks:
        return []
    query = preprocess_text(query)
    if not query:
        return []
    try:
        candidates = hybrid_search(query, store, chunks, embedding_model, pdf_hash, top_k=k*3)
        if not candidates:
            return []
        reranked = rerank_with_cross_encoder(query, candidates, chunks, embedding_model, top_k=RERANK_TOP_K)
        results = []
        for item in reranked:
            chunk_idx = item['chunk_idx']
            chunk = chunks[chunk_idx]
            results.append({'chunk_id': int(chunk['chunk_id']), 'similarity': float(item['sim']), 'final_score': float(item['final_score']), 'text': item['text'], 'term_overlap': float(item['term_overlap'])})
        return results
    except:
        return []