from flask import Flask, render_template, request, jsonify, session, Response
from flask_session import Session
import os, sys, requests, json, uuid, threading, time, gc
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

from scripts.logger_config import setup_logger
from scripts.task_manager import tasks, tasks_lock
from scripts.gemma_api import start_ollama, detect_language, build_enhanced_prompt, LANGUAGE_PROMPTS, GEMMA_API_URL, GEMMA_MODEL, MAX_CONTEXT_LENGTH, api_session
from scripts.pdf_ocr import find_tesseract, extract_text_ocr_streaming, ocr_worker, get_pdf_hash
from scripts.embedding_rag import init_embedding, create_vector_store_streaming, retrieve_chunks, load_chunks, load_vector_store, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from scripts.speech_to_text import speech_converter

logger = setup_logger()

def resource_path(relative_path):
    return os.path.join(getattr(sys, '_MEIPASS', os.path.abspath(".")), relative_path)

embedding_model = None
vector_stores, chunk_stores = {}, {}
rag_lock = threading.Lock()
pdf_metadata = {}
metadata_lock = threading.Lock()

app = Flask(__name__, template_folder=resource_path('templates'))
app.secret_key = 'prod_secret_key_2024'
UPLOAD_FOLDER = 'uploads'
EXTRACTED_FOLDER = 'extracted_texts'
CHUNKS_FOLDER = 'chunks_store'
METADATA_FILE = os.path.join(CHUNKS_FOLDER, 'metadata.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)
os.makedirs(CHUNKS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.update(SESSION_PERMANENT=True, SESSION_TYPE="filesystem", PERMANENT_SESSION_LIFETIME=timedelta(hours=1))
Session(app)

tesseract_path = find_tesseract()
if tesseract_path:
    logger.info(f"Tesseract found at: {tesseract_path}")
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    logger.warning("Tesseract not found - OCR functionality may be limited")

def load_metadata():
    global pdf_metadata
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                pdf_metadata = json.load(f)
            logger.info(f"Metadata loaded: {len(pdf_metadata)} documents")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            pdf_metadata = {}
    else:
        logger.info("No existing metadata file found")

def save_metadata():
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(pdf_metadata, f, indent=2, ensure_ascii=False)
        logger.debug("Metadata saved successfully")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")

@app.route('/')
def index():
    logger.debug("Index page accessed")
    return render_template('index.html')

@app.route('/get-documents', methods=['GET'])
def get_documents():
    try:
        with metadata_lock:
            docs = []
            for pdf_hash, data in pdf_metadata.items():
                if os.path.exists(data.get('txt_path', '')):
                    docs.append({
                        'hash': pdf_hash,
                        'filename': data.get('filename', 'Unknown'),
                        'language': data.get('detected_language', 'unknown'),
                        'cached_at': data.get('cached_at', ''),
                        'last_accessed': data.get('last_accessed', ''),
                        'access_count': data.get('access_count', 0),
                        'chunk_count': data.get('chunk_count', 0)
                    })
            docs.sort(key=lambda x: x['last_accessed'], reverse=True)
            logger.info(f"Documents retrieved: {len(docs)} documents")
            return jsonify({'success': True, 'documents': docs})
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/load-document', methods=['POST'])
def load_document():
    try:
        pdf_hash = request.json.get('hash')
        logger.info(f"Load document request: hash={pdf_hash}")
        
        if not pdf_hash:
            logger.warning("No document hash provided")
            return jsonify({'error': 'No document hash provided'}), 400
            
        with metadata_lock:
            if pdf_hash not in pdf_metadata:
                logger.warning(f"Document not found: {pdf_hash}")
                return jsonify({'error': 'Document not found'}), 404
                
            doc_data = pdf_metadata[pdf_hash]
            txt_path = doc_data.get('txt_path')
            
            if not txt_path or not os.path.exists(txt_path):
                logger.error(f"Document file not found: {txt_path}")
                return jsonify({'error': 'Document file not found'}), 404
                
            session_id = session.get('session_id', str(uuid.uuid4()))
            session['session_id'] = session_id
            session['txt_filepath'] = txt_path
            session['original_filename'] = doc_data.get('filename')
            session['pdf_hash'] = pdf_hash
            
            logger.info(f"Session created: session_id={session_id}, filename={doc_data.get('filename')}")
            
            if embedding_model:
                logger.debug(f"Loading chunks for document: {pdf_hash}")
                chunks = load_chunks(pdf_hash)
                store = load_vector_store(pdf_hash)
                
                if chunks and store:
                    with rag_lock:
                        vector_stores[session_id] = store
                        chunk_stores[session_id] = chunks
                    
                    logger.info(f"RAG data loaded successfully for session {session_id}")
                    logger.info(f"Total chunks loaded: {len(chunks)}")
                    logger.debug(f"Chunk store details: {json.dumps([{'chunk_id': c.get('chunk_id', 'unknown'), 'length': len(c.get('text', '')), 'has_embedding': 'embedding' in c} for c in chunks[:5]], indent=2)}")
                    logger.info(f"Vector store initialized with {len(chunks)} embeddings")
                else:
                    logger.warning(f"RAG data not available for document {pdf_hash}")
                    if not chunks:
                        logger.warning("Chunks could not be loaded")
                    if not store:
                        logger.warning("Vector store could not be loaded")
            else:
                logger.info("Embedding model not initialized - RAG disabled")
                
            pdf_metadata[pdf_hash]['access_count'] = doc_data.get('access_count', 0) + 1
            pdf_metadata[pdf_hash]['last_accessed'] = datetime.now().isoformat()
            save_metadata()
            
            logger.info(f"Document loaded successfully: {doc_data.get('filename')}, access_count={pdf_metadata[pdf_hash]['access_count']}")
            
            return jsonify({
                'success': True,
                'filename': doc_data.get('filename'),
                'language': doc_data.get('detected_language'),
                'rag_enabled': bool(embedding_model)
            })
    except Exception as e:
        logger.error(f"Error loading document: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.warning("Upload attempt with no file")
        return jsonify({'error': 'No file'}), 400
        
    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file upload: {file.filename}")
        return jsonify({'error': 'Invalid PDF'}), 400
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{pdf_name}")
    txt_path = os.path.join(EXTRACTED_FOLDER, f"{timestamp}_{pdf_name}.txt")
    
    file.save(pdf_path)
    file_size = os.path.getsize(pdf_path)
    logger.info(f"File uploaded: {pdf_name}, size={file_size} bytes, saved to {pdf_path}")
    
    pdf_hash = get_pdf_hash(pdf_path)
    logger.info(f"PDF hash computed: {pdf_hash}")
    
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    task_id = str(uuid.uuid4())
    session.update({'txt_filepath': txt_path, 'original_filename': file.filename, 'pdf_hash': pdf_hash})
    
    logger.info(f"OCR task created: task_id={task_id}, session_id={session_id}, file={pdf_name}")
    
    def create_vector_store_wrapper(txt_path, pdf_hash, callback):
        return create_vector_store_streaming(txt_path, pdf_hash, embedding_model, callback)
    
    threading.Thread(target=ocr_worker, args=(
        pdf_path, txt_path, task_id, tasks, tasks_lock, tesseract_path, pdf_hash, file.filename, session_id,
        embedding_model, metadata_lock, pdf_metadata, save_metadata,
        rag_lock, vector_stores, chunk_stores, load_chunks, load_vector_store,
        create_vector_store_wrapper, detect_language
    ), daemon=True).start()
    
    cached = pdf_hash in pdf_metadata
    logger.info(f"Upload response: task_id={task_id}, cached={cached}, rag_enabled={bool(embedding_model)}")
    
    return jsonify({'success': True, 'task_id': task_id, 'filename': file.filename, 'cached': cached, 'rag_enabled': bool(embedding_model)})

@app.route('/process-status')
def process_status():
    task_id = request.args.get('id')
    if not task_id:
        logger.warning("Process status check without task ID")
        return "Task ID required", 400
        
    logger.debug(f"Process status stream started for task: {task_id}")
    
    def generate():
        while True:
            with tasks_lock:
                task = tasks.get(task_id, {}).copy()
            if task:
                yield f"data: {json.dumps(task)}\n\n"
                if task.get('status') in ['complete', 'error']:
                    logger.info(f"Task completed: {task_id}, status={task.get('status')}")
                    yield f"event: complete\ndata: {json.dumps({'message': task.get('message'), 'progress': 100})}\n\n"
                    threading.Timer(60.0, lambda: tasks.pop(task_id, None)).start()
                    break
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        audio_file = request.files.get('audio')
        language = request.form.get('language', 'english')
        
        logger.info(f"Transcription request: language={language}")
        
        if not audio_file:
            logger.warning("Transcription attempt with no audio file")
            return jsonify({'success': False, 'error': 'No audio file'}), 400
        
        logger.debug(f"Starting transcription: language={language}")
        transcription = speech_converter.transcribe(audio_file, language)
        
        logger.info(f"Transcription completed: length={len(transcription)} chars, language={language}")
        return jsonify({'success': True, 'text': transcription, 'language': language})
    
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question', '')
    if not question:
        logger.warning("Empty question received")
        return jsonify({'error': 'Empty question'}), 400
        
    logger.info(f"Question received: '{question[:100]}{'...' if len(question) > 100 else ''}'")
    
    txt_path = session.get('txt_filepath')
    session_id = session.get('session_id')
    pdf_hash = session.get('pdf_hash')
    doc_data = pdf_metadata.get(pdf_hash, {})
    document_language = doc_data.get('detected_language', 'english')
    
    logger.debug(f"Session info: session_id={session_id}, pdf_hash={pdf_hash}, doc_language={document_language}")
    
    if not txt_path or not os.path.exists(txt_path):
        logger.error(f"No document loaded or file missing: {txt_path}")
        return jsonify({'error': 'No document loaded'}), 400
        
    logger.debug("Starting Ollama service")
    start_ollama()
    
    try:
        requests.get('http://localhost:11434/api/tags', timeout=5).raise_for_status()
        logger.debug("Ollama service is available")
    except Exception as e:
        logger.error(f"Ollama service unavailable: {e}")
        return jsonify({'error': 'Gemma model service is not available. Please ensure Ollama is running.'}), 503
        
    rag_used = False
    context = ""
    chunk_info = []
    
    if embedding_model and session_id:
        logger.debug(f"Attempting RAG retrieval for session: {session_id}")
        with rag_lock:
            store = vector_stores.get(session_id)
            chunks = chunk_stores.get(session_id)
            
        if store and chunks:
            logger.info(f"RAG data available: total_chunks={len(chunks)}")
            logger.debug(f"Chunk store sample (first 3): {json.dumps([{'chunk_id': c.get('chunk_id'), 'text_length': len(c.get('text', '')), 'text_preview': c.get('text', '')[:100]} for c in chunks[:3]], indent=2, ensure_ascii=False)}")
            
            logger.debug(f"Retrieving relevant chunks for query: top_k={TOP_K}")
            relevant = retrieve_chunks(question, store, chunks, embedding_model, TOP_K, document_language, pdf_hash)
            
            if relevant:
                logger.info(f"Retrieved {len(relevant)} chunks from vector store")
                
                for idx, chunk in enumerate(relevant, 1):
                    chunk_detail = {
                        "rank": idx,
                        "chunk_id": chunk.get('chunk_id', 'unknown'),
                        "similarity_score": round(chunk.get('similarity', 0), 4),
                        "final_score": round(chunk.get('final_score', chunk.get('similarity', 0)), 4),
                        "text_length": len(chunk.get('text', '')),
                        "text_preview": chunk.get('text', '')[:150],
                        "full_text": chunk.get('text', '')
                    }
                    logger.info(f"Chunk #{idx} details: {json.dumps({k: v for k, v in chunk_detail.items() if k != 'full_text'}, indent=2, ensure_ascii=False)}")
                    logger.debug(f"Chunk #{idx} full text:\n{'-'*80}\n{chunk_detail['full_text']}\n{'-'*80}")
                
                top_score = relevant[0].get('similarity', 0)
                logger.info(f"Top chunk similarity score: {top_score:.4f}")
                
                if top_score > 0:
                    rag_used = True
                    logger.info(f"RAG ACTIVATED - Score threshold met: {top_score:.4f} > 0")

                    context_parts = []
                    for idx, c in enumerate(relevant, 1):
                        section = f"[Section {idx} - Relevance: {c.get('final_score', c.get('similarity', 0)):.2f}]\n{c['text']}\n"
                        context_parts.append(section)
                        logger.debug(f"Added to context - Section {idx}: {len(c['text'])} chars, score={c.get('final_score', c.get('similarity', 0)):.4f}")
                    
                    context = "\n".join(context_parts)
                    logger.info(f"RAG context built: total_length={len(context)} chars, sections={len(relevant)}")
                    
                    chunk_info = [{
                        'id': c['chunk_id'], 
                        'similarity': float(c.get('similarity', 0)), 
                        'final_score': float(c.get('final_score', 0)), 
                        'text_preview': c['text'][:200],
                        'text_length': len(c['text'])
                    } for c in relevant]
                    
                    logger.debug(f"Chunk info summary: {json.dumps(chunk_info, indent=2)}")
                else:
                    logger.warning(f"RAG NOT ACTIVATED - Score too low: {top_score:.4f} <= 0.3")
                    logger.info("Falling back to full document context")
            else:
                logger.warning("No chunks retrieved from vector store")
        else:
            logger.warning(f"RAG data not loaded for session {session_id}: store={'present' if store else 'missing'}, chunks={'present' if chunks else 'missing'}")
    else:
        logger.debug(f"RAG disabled: embedding_model={'present' if embedding_model else 'missing'}, session_id={session_id}")
        
    if not context:
        logger.info("Loading full document context (RAG not used)")
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
            context = full_text[:MAX_CONTEXT_LENGTH]
            logger.info(f"Full text context loaded: {len(full_text)} total chars, using {len(context)} chars (max: {MAX_CONTEXT_LENGTH})")
            logger.debug(f"Context preview (first 300 chars):\n{context[:300]}...")
            
    detected_lang = detect_language(question, True)
    logger.info(f"Query language detected: {detected_lang}")
    
    logger.debug(f"Building prompt: question_lang={detected_lang}, doc_lang={document_language}, rag={rag_used}")
    prompt = build_enhanced_prompt(question, context, detected_lang, is_rag=rag_used)
    logger.info(f"Prompt built: total_length={len(prompt)} chars")
    logger.debug(f"Prompt preview (first 500 chars):\n{prompt[:500]}...")
    
    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 800,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "seed": 42
        }
    }
    
    logger.info(f"LLM request payload: model={GEMMA_MODEL}, temp={payload['options']['temperature']}, max_tokens={payload['options']['num_predict']}, top_k={payload['options']['top_k']}, top_p={payload['options']['top_p']}")
    response_text = []
    token_count = 0
    start_time = time.time()
    
    def generate():
        nonlocal token_count
        try:
            logger.info("Initiating LLM streaming request")
            with api_session.post(GEMMA_API_URL, json=payload, stream=True, timeout=3000) as r:
                r.raise_for_status()
                first_token_time = None
                
                for line in r.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get('response', '')
                            if token:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    latency = first_token_time - start_time
                                    logger.info(f"First token received: latency={latency:.3f}s")
                                    
                                response_text.append(token)
                                token_count += 1
                                
                                if token_count % 50 == 0:
                                    logger.debug(f"Streaming progress: {token_count} tokens, {len(''.join(response_text))} chars")
                                    
                            yield token
                        except:
                            continue
                            
            full_response = ''.join(response_text)
            total_time = time.time() - start_time
            tokens_per_sec = token_count / total_time if total_time > 0 else 0
            
            logger.info(f"Response generation complete:")
            logger.info(f"  - Tokens: {token_count}")
            logger.info(f"  - Characters: {len(full_response)}")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Throughput: {tokens_per_sec:.2f} tokens/s")
            logger.info(f"  - First token latency: {(first_token_time - start_time):.3f}s" if first_token_time else "  - First token latency: N/A")
            
            if not full_response.strip() or len(full_response) < 20:
                logger.warning(f"Response validation failed: length={len(full_response)} chars, is_empty={not full_response.strip()}")
                lang_config = LANGUAGE_PROMPTS.get(detected_lang, LANGUAGE_PROMPTS['english'])
                fallback_msg = lang_config['no_answer']
                logger.info(f"Using fallback message: '{fallback_msg}'")
                yield f"\n\n{fallback_msg}"
            else:
                logger.info(f"Response preview (first 200 chars): {full_response[:200]}...")
                
            query_summary = {
                "event": "query_complete",
                "timestamp": datetime.now().isoformat(),
                "query": {
                    "text": question,
                    "length": len(question),
                    "detected_language": detected_lang
                },
                "response": {
                    "length": len(full_response),
                    "tokens": token_count,
                    "preview": full_response[:200]
                },
                "performance": {
                    "total_time_seconds": round(total_time, 3),
                    "first_token_latency_seconds": round(first_token_time - start_time, 3) if first_token_time else None,
                    "tokens_per_second": round(tokens_per_sec, 2)
                },
                "context": {
                    "rag_used": rag_used,
                    "chunks_retrieved": len(chunk_info),
                    "context_length": len(context),
                    "query_language": detected_lang,
                    "document_language": document_language,
                    "context_source": "RAG" if rag_used else "Full Document"
                }
            }
            
            logger.info(f"Query summary:\n{json.dumps(query_summary, indent=2, ensure_ascii=False)}")
            
            if chunk_info:
                logger.info(f"Chunks used in response ({len(chunk_info)} total):")
                for idx, chunk in enumerate(chunk_info, 1):
                    logger.info(f"  Chunk {idx}:")
                    logger.info(f"    - ID: {chunk['id']}")
                    logger.info(f"    - Similarity: {chunk['similarity']:.4f}")
                    logger.info(f"    - Final score: {chunk['final_score']:.4f}")
                    logger.info(f"    - Text length: {chunk['text_length']} chars")
                    logger.debug(f"    - Preview: {chunk['text_preview']}")
                
        except Exception as e:
            logger.error(f"LLM streaming error: {e}", exc_info=True)
            error_msg = f"Error: {str(e)}"
            yield error_msg
    log_data = {
                "query": question,
                "chunks_used": chunk_info,
                "payload": payload,
                "tokens": token_count,
                "rag": rag_used,
                "language": detected_lang
            }
    logger.info("\n" + json.dumps(log_data, ensure_ascii=False, default=str, indent=2))
            
    return Response(generate(), mimetype='text/plain')

@app.route('/clear', methods=['POST'])
def clear_session():
    session_id = session.get('session_id')
    logger.info(f"Session clear request: session_id={session_id}")
    
    if session_id:
        with rag_lock:
            had_vectors = session_id in vector_stores
            had_chunks = session_id in chunk_stores
            chunks_count = len(chunk_stores.get(session_id, []))
            
            vector_stores.pop(session_id, None)
            chunk_stores.pop(session_id, None)
            
        logger.info(f"Session data cleared:")
        logger.info(f"  - Had vectors: {had_vectors}")
        logger.info(f"  - Had chunks: {had_chunks}")
        logger.info(f"  - Chunks removed: {chunks_count}")
        
    session.clear()
    gc.collect()
    logger.info("Session cleared and garbage collection completed")
    return jsonify({'success': True})

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("APPLICATION STARTUP")
    logger.info("=" * 80)
    
    logger.info(f"Configuration:")
    logger.info(f"  - Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"  - Extracted folder: {EXTRACTED_FOLDER}")
    logger.info(f"  - Chunks folder: {CHUNKS_FOLDER}")
    logger.info(f"  - Metadata file: {METADATA_FILE}")
    logger.info(f"  - Chunk size: {CHUNK_SIZE}")
    logger.info(f"  - Chunk overlap: {CHUNK_OVERLAP}")
    logger.info(f"  - Top K chunks: {TOP_K}")
    logger.info(f"  - Max context length: {MAX_CONTEXT_LENGTH}")
    
    load_metadata()
    
    logger.info("Initializing embedding model...")
    embedding_model = init_embedding()
    if embedding_model:
        logger.info("Embedding model initialized successfully")
    else:
        logger.warning("Embedding model initialization failed - RAG disabled")
        
    logger.info("Starting Ollama service...")
    start_ollama()
    
    logger.info(f"Starting Flask server on 0.0.0.0:5000")
    logger.info("=" * 80)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)