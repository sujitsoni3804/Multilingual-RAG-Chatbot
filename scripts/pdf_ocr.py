import os
import sys
import hashlib
import shutil
import io
import gc
import threading
import pytesseract
import fitz
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
import numpy as np

def resource_path(relative_path):
    return os.path.join(getattr(sys, '_MEIPASS', os.path.abspath(".")), relative_path)

def find_tesseract():
    paths = [resource_path(os.path.join('Tesseract-OCR', 'tesseract.exe')), 
             r'C:\Program Files\Tesseract-OCR\tesseract.exe', 
             shutil.which('tesseract')]
    return next((p for p in paths if p and os.path.exists(p)), None)

def get_pdf_hash(path):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except:
        return None

def preprocess_image(img):
    """Enhance image quality for better OCR accuracy"""
    try:
        img = img.convert('L')
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img_array = np.array(img)
        threshold = np.mean(img_array)
        img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        img = Image.fromarray(img_array)
        return img
    except:
        return img

def is_text_relevant(text):
    """Filter out irrelevant text (too short, non-linguistic, etc.)"""
    if not text or len(text.strip()) < 10:
        return False
    
    words = text.split()
    if len(words) < 2:
        return False
    
    unique_chars = len(set(text.replace(' ', '').replace('\n', '')))
    if unique_chars < 5:
        return False
    
    alphanumeric_count = sum(c.isalnum() or c.isspace() for c in text)
    if alphanumeric_count < len(text) * 0.6:
        return False
    
    return True

def extract_native_text(page):
    """Extract native text with filtering"""
    try:
        blocks = page.get_text("dict")["blocks"]
        text_parts = []
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    
                    if is_text_relevant(line_text):
                        text_parts.append(line_text.strip())
        
        return "\n".join(text_parts)
    except:
        return ""

def is_image_page(page, threshold=0.7):
    """Determine if page is mostly images (skip OCR on such pages)"""
    try:
        blocks = page.get_text("dict")["blocks"]
        total_area = page.rect.width * page.rect.height
        image_area = 0
        
        for block in blocks:
            if block.get("type") == 1:
                bbox = block.get("bbox", [0, 0, 0, 0])
                image_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        return (image_area / total_area) > threshold if total_area > 0 else False
    except:
        return False

def extract_text_ocr_streaming(pdf_path, txt_path, tesseract_path, callback=None):
    """Extract text from PDF with intelligent filtering"""
    if not tesseract_path:
        return "TESSERACT_NOT_FOUND"
    
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            for i, page in enumerate(doc):
                native_text = extract_native_text(page)
                
                if native_text and len(native_text.strip()) > 50:
                    f.write(native_text + "\n\n")
                else:
                    if is_image_page(page, threshold=0.7):
                        if callback:
                            progress = int((i + 1) / total * 85)
                            callback(progress, f"Skipping image page {i+1}/{total}")
                        continue
                    
                    try:
                        pix = page.get_pixmap(dpi=300, alpha=False)
                        img_bytes = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        img = preprocess_image(img)
                        
                        page_text = pytesseract.image_to_string(
                            img,
                            lang='guj+hin+eng',
                            config='--oem 3 --psm 6 -c preserve_interword_spaces=1'
                        )
                        
                        if is_text_relevant(page_text):
                            f.write(page_text + "\n\n")
                        
                        img.close()
                        pix = None
                        img_bytes = None
                        page_text = None
                        
                    except Exception as ocr_error:
                        if callback:
                            progress = int((i + 1) / total * 85)
                            callback(progress, f"OCR failed on page {i+1}/{total}")
                        continue
                
                f.flush()
                gc.collect()
                
                if callback:
                    progress = int((i + 1) / total * 85)
                    callback(progress, f"Processing page {i+1}/{total}")
        
        doc.close()
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if len(content) < 100:
                    return False
        
        return True
    
    except Exception as e:
        return False

def ocr_worker(pdf_path, txt_path, task_id, tasks, tasks_lock, tesseract_path, pdf_hash=None, 
               filename=None, session_id=None, embedding_model=None, metadata_lock=None, 
               pdf_metadata=None, save_metadata_fn=None, rag_lock=None, vector_stores=None, 
               chunk_stores=None, load_chunks_fn=None, load_vector_store_fn=None, 
               create_vector_store_streaming_fn=None, detect_language_fn=None):
    
    def update(prog, msg):
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id].update({'progress': prog, 'message': msg})
    
    try:
        with tasks_lock:
            tasks[task_id] = {'status': 'processing', 'progress': 0, 'message': 'Starting...'}
        
        with metadata_lock:
            if pdf_hash and pdf_hash in pdf_metadata:
                cached_data = pdf_metadata[pdf_hash]
                cached_txt = cached_data.get('txt_path', '')
                
                if os.path.exists(cached_txt):
                    if txt_path != cached_txt:
                        shutil.copy2(cached_txt, txt_path)
                    
                    if embedding_model and session_id:
                        chunks = load_chunks_fn(pdf_hash)
                        store = load_vector_store_fn(pdf_hash)
                        if chunks and store:
                            with rag_lock:
                                vector_stores[session_id] = store
                                chunk_stores[session_id] = chunks
                    
                    pdf_metadata[pdf_hash]['access_count'] = cached_data.get('access_count', 0) + 1
                    pdf_metadata[pdf_hash]['last_accessed'] = datetime.now().isoformat()
                    save_metadata_fn()
                    
                    with open(cached_txt, 'r', encoding='utf-8') as f:
                        sample = f.read(1000)
                    
                    with tasks_lock:
                        tasks[task_id].update({
                            'status': 'complete', 
                            'progress': 100, 
                            'message': f'Loaded from cache! ({detect_language_fn(sample).title()})'
                        })
                    
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    return
        
        result = extract_text_ocr_streaming(pdf_path, txt_path, tesseract_path, update)
        
        if result == "TESSERACT_NOT_FOUND":
            raise Exception('Tesseract not found')
        if not result:
            raise Exception('Failed to extract text or insufficient content')
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            sample = f.read(1000)
        lang = detect_language_fn(sample)
        
        chunks = []
        if embedding_model and session_id:
            update(90, 'Creating vector embeddings...')
            
            def chunk_update(msg):
                update(92, msg)
            
            store, chunks = create_vector_store_streaming_fn(txt_path, pdf_hash, chunk_update)
            
            if store and chunks:
                with rag_lock:
                    vector_stores[session_id] = store
                    chunk_stores[session_id] = chunks
                update(98, 'Finalizing...')
        
        with metadata_lock:
            pdf_metadata[pdf_hash] = {
                'filename': filename,
                'txt_path': txt_path,
                'detected_language': lang,
                'cached_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 1,
                'chunk_count': len(chunks) if chunks else 0
            }
            save_metadata_fn()
        
        with tasks_lock:
            tasks[task_id].update({
                'status': 'complete', 
                'progress': 100, 
                'message': f'Ready! ({lang.title()})'
            })
    
    except Exception as e:
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id].update({'status': 'error', 'message': str(e)})
    
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        gc.collect()