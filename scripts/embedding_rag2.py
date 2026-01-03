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
HYBRID_ALPHA = 0.6
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
    except Exception:
        return None

def preprocess_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()

def create_chunks_streaming(txt_path, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    
    base = Path(__file__).resolve().parent.parent
    model_path = base / EMBEDDING_MODEL_NAME
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = preprocess_text(f.read())
        
        sentences = re.split(r'[।॥\.\!\?]\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return
        
        chunk_id = 0
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            sent_size = len(tokens)
            
            if current_size + sent_size > size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                yield {'text': chunk_text, 'chunk_id': chunk_id, 'start_pos': chunk_id * (size - overlap), 'end_pos': (chunk_id + 1) * (size - overlap)}
                chunk_id += 1
                
                overlap_sents = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
                    if overlap_size + s_tokens <= overlap:
                        overlap_sents.insert(0, s)
                        overlap_size += s_tokens
                    else:
                        break
                
                current_chunk = overlap_sents
                current_size = overlap_size
            
            current_chunk.append(sent)
            current_size += sent_size
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            yield {'text': chunk_text, 'chunk_id': chunk_id, 'start_pos': chunk_id * (size - overlap), 'end_pos': (chunk_id + 1) * (size - overlap)}
            
    except Exception:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = preprocess_text(f.read())
        step = size - overlap
        start, chunk_id = 0, 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                yield {'text': chunk_text, 'chunk_id': chunk_id, 'start_pos': start, 'end_pos': end}
                chunk_id += 1
            start += step

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
    files = [
        f"{pdf_hash}_vectors.index", f"{pdf_hash}_chunks.pkl", f"{pdf_hash}_chunks.json",
        f"{pdf_hash}_tfidf_matrix.npz", f"{pdf_hash}_tfidf_vectorizer.pkl"
    ]
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
        embeddings_buffer = []
        index = None
        total_processed = 0
        
        for chunk in create_chunks_streaming(txt_path):
            all_chunks.append(chunk)
            chunk_buffer.append(chunk['text'])
            
            if len(chunk_buffer) >= BATCH_SIZE:
                embeddings = encode_in_batches(chunk_buffer, embedding_model)
                embeddings_buffer.append(embeddings)
                
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
            embeddings_buffer.append(embeddings)
            
            if index is None:
                dim = embeddings.shape[1]
                index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            
            ids = np.arange(total_processed, total_processed + len(chunk_buffer))
            index.add_with_ids(embeddings, ids)
            total_processed += len(chunk_buffer)
            embeddings = None
            gc.collect()
        
        texts = [c['text'] for c in all_chunks]
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), analyzer='word', token_pattern=r"(?u)\b\w\w+\b", max_df=0.9, min_df=1)
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
    
    normalized = re.sub(r'[^\w\s]', ' ', query)
    if normalized != query:
        variations.append(normalized)
    
    if len(query) > 20:
        words = query.split()
        if len(words) > 3:
            partial = ' '.join(words[:2] + words[-2:])
            variations.append(partial)
    
    return variations

def hybrid_search(query, store, chunks, embedding_model, pdf_hash, top_k=TOP_K):
    if not chunks or store is None:
        return []
    
    query_variations = augment_query(query)
    query_embs = embedding_model.encode(query_variations, convert_to_tensor=True, normalize_embeddings=True)
    
    import torch
    query_emb = torch.mean(query_embs, dim=0, keepdim=True).cpu().numpy()
    
    k = min(top_k * 4, len(chunks))
    dense_scores, dense_ids = store.search(query_emb, k)
    dense_scores = dense_scores[0]
    dense_ids = dense_ids[0]
    
    sparse_scores = np.zeros(len(chunks))
    tfidf_matrix, tfidf_vectorizer = load_tfidf_index(pdf_hash)
    
    if tfidf_vectorizer and tfidf_matrix is not None:
        try:
            query_tfidf = tfidf_vectorizer.transform([preprocess_text(query)])
            sparse_sim = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            sparse_scores = sparse_sim
        except:
            pass
    
    final_scores = {}
    for i, idx in enumerate(dense_ids):
        if idx != -1:
            final_scores[int(idx)] = HYBRID_ALPHA * dense_scores[i]
    
    top_sparse_idx = np.argsort(-sparse_scores)[:k]
    for idx in top_sparse_idx:
        idx_int = int(idx)
        if idx_int in final_scores:
            final_scores[idx_int] += (1 - HYBRID_ALPHA) * sparse_scores[idx]
        else:
            final_scores[idx_int] = (1 - HYBRID_ALPHA) * sparse_scores[idx]
    
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

def rerank_with_cross_encoder(query, candidate_chunks, chunks, embedding_model, top_k=RERANK_TOP_K):
    if not candidate_chunks:
        return []
    
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    
    reranked = []
    for chunk_idx, initial_score in candidate_chunks:
        chunk_text = chunks[chunk_idx]["text"]
        chunk_emb = embedding_model.encode(chunk_text, convert_to_tensor=True)
        
        sim = util.pytorch_cos_sim(query_emb, chunk_emb).item()
        
        query_terms = set(query.lower().split())
        chunk_terms = set(chunk_text.lower().split())
        term_overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        
        final_score = sim * 0.8 + term_overlap * 0.2
        
        if sim < 0.1:
            continue
        
        reranked.append((chunk_idx, final_score, sim, chunk_text))
    
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]

def retrieve_chunks(query, store, chunks, embedding_model, k=TOP_K, document_language='english', pdf_hash=None):
    if not embedding_model or not store or not chunks:
        return []
    
    query = preprocess_text(query)
    if not query:
        return []
    
    try:
        candidates = hybrid_search(query, store, chunks, embedding_model, pdf_hash, top_k=k*2)
        reranked = rerank_with_cross_encoder(query, candidates, chunks, embedding_model, top_k=RERANK_TOP_K)
        
        results = []
        for chunk_idx, final_score, sim_score, text in reranked:
            chunk = chunks[chunk_idx]
            results.append({
                'chunk_id': int(chunk['chunk_id']),
                'similarity': float(sim_score),
                'final_score': float(final_score),
                'text': text
            })
        
        return results
    except:
        return []