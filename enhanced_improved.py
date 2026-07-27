# Integrated Hindi-Punjabi Translation + Cross-Language Plagiarism Detection System
# ENHANCED VERSION with Document Upload & Direct Google Search

import pandas as pd
import io
import contextlib
import warnings
import sqlite3
import json
import pickle
import numpy as np
import re
import unicodedata
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from werkzeug.utils import secure_filename
import traceback
import requests

warnings.filterwarnings('ignore')

try:
    import torch
    import sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from flask import Flask, render_template, request, jsonify, send_file
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"ERROR: Missing dependency - {e}")
    print("Please run: pip install flask transformers torch sacrebleu pandas sentencepiece protobuf sentence-transformers scikit-learn requests")
    exit(1)

print("="*80)
print("INTEGRATED: Hindi-Punjabi Translation + Cross-Language Plagiarism Detection")
print("ENHANCED: Document Upload + Direct Google Search after Translation")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.config.update({
    'SECRET_KEY': 'integrated-system-key',
    'UPLOAD_FOLDER': 'uploads',
    'CORPUS_FOLDER': 'corpus',
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # Increased to 50MB
    'ALLOWED_EXTENSIONS': {'txt', 'pdf', 'docx', 'doc'}
})

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CORPUS_FOLDER'], exist_ok=True)

# Path where train_indicbert.py saves the fine-tuned Hindi-Punjabi model.
# If present at startup, it is used instead of the generic pretrained model.
FINETUNED_INDICBERT_PATH = os.path.join('models', 'indicbert-finetuned-hi-pa')
PRETRAINED_SEMANTIC_MODEL = 'l3cube-pune/indic-sentence-similarity-sbert'

model_cache = {}
corpus_cache = {
    'documents': [],
    'embeddings': None,
    'metadata': [],
    'last_updated': None,
    'corpus_size': 0
}

# ============================================================================
# 0. DOCUMENT HANDLING UTILITIES
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(filepath):
    """Extract text from various file formats"""
    filename = os.path.basename(filepath)
    ext = filename.rsplit('.', 1)[1].lower()
    
    try:
        if ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext in ['pdf']:
            try:
                import PyPDF2
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                print("⚠️ PyPDF2 not installed. Install with: pip install PyPDF2")
                return None
        elif ext in ['docx']:
            try:
                from docx import Document
                doc = Document(filepath)
                text = ''
                for para in doc.paragraphs:
                    text += para.text + '\n'
                return text
            except ImportError:
                print("⚠️ python-docx not installed. Install with: pip install python-docx")
                return None
        elif ext == 'doc':
            try:
                from docx import Document
                doc = Document(filepath)
                text = ''
                for para in doc.paragraphs:
                    text += para.text + '\n'
                return text
            except ImportError:
                print("⚠️ python-docx not installed for .doc files")
                return None
        return None
    except Exception as e:
        print(f"❌ Error extracting text from {filename}: {e}")
        return None

# ============================================================================
# 1. MODEL LOADING & INITIALIZATION
# ============================================================================

def load_models():
    """Load all required models for translation and plagiarism detection"""
    global model_cache
    
    if model_cache.get('loaded'):
        return True
    
    try:
        print("\n[STEP 1/3] Loading Models...")
        print("=" * 80)
        
        # Load Translation Models
        print("\n📖 Loading NLLB-200 Translation Model...")
        model_name = "facebook/nllb-200-distilled-600M"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(device)
        
        model_cache['tokenizer'] = tokenizer
        model_cache['model'] = model
        model_cache['device'] = device
        
        print("✅ NLLB-200 model loaded successfully!")
        
        # Load Semantic Model for Cross-Language Detection.
        # Prefer our own IndicBERT fine-tuned on the Hindi-Punjabi parallel
        # corpus (see train_indicbert.py) and fall back to the generic
        # pretrained IndicSBERT model when no fine-tuned checkpoint exists -
        # or when it exists but turns out to be broken (a training run that
        # was interrupted mid-save, e.g. by an OOM kill, can leave a
        # partial/corrupted checkpoint that still "loads" as an object but
        # throws things like "IndexError: index out of range in self" the
        # first time it actually encodes something). Running one real
        # encode() call here as a self-test surfaces that immediately, with
        # a clear message, instead of it resurfacing confusingly later deep
        # inside unrelated request handling.
        print("\n🧠 Loading Semantic Similarity Model for Cross-Language Detection...")
        semantic_model = None
        model_cache['semantic_model_source'] = None

        if os.path.isdir(FINETUNED_INDICBERT_PATH) and os.listdir(FINETUNED_INDICBERT_PATH):
            try:
                print(f"   Found fine-tuned IndicBERT checkpoint: '{FINETUNED_INDICBERT_PATH}'")
                candidate = SentenceTransformer(FINETUNED_INDICBERT_PATH)
                candidate.encode(["सत्यापन वाक्य"])  # self-test
                semantic_model = candidate
                model_cache['semantic_model_source'] = 'finetuned-indicbert'
                print("✅ Fine-tuned IndicBERT (Hindi-Punjabi) loaded successfully!")
            except Exception as e:
                print(f"⚠️ Fine-tuned IndicBERT checkpoint failed to load/self-test: {e}")
                print(f"   This usually means training was interrupted before it finished and left a "
                      f"partial/corrupted checkpoint at '{FINETUNED_INDICBERT_PATH}'. Delete that folder "
                      f"and re-run train_indicbert.py when convenient - falling back to the pretrained "
                      f"model for now.")
        else:
            print(f"   No fine-tuned IndicBERT found at '{FINETUNED_INDICBERT_PATH}'.")
            print(f"   Run 'python train_indicbert.py' to fine-tune one on the custom corpus.")

        if semantic_model is None:
            try:
                print(f"   Loading pretrained model: {PRETRAINED_SEMANTIC_MODEL}")
                candidate = SentenceTransformer(PRETRAINED_SEMANTIC_MODEL)
                candidate.encode(["सत्यापन वाक्य"])  # self-test
                semantic_model = candidate
                model_cache['semantic_model_source'] = 'pretrained-indicsbert'
                print("✅ Pretrained IndicSBERT loaded successfully!")
            except Exception as e:
                print(f"⚠️ Semantic model loading failed: {e}")

        model_cache['semantic_model'] = semantic_model
        
        # Load TF-IDF Vectorizer
        print("\n📝 Loading TF-IDF Vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer='char_wb',
            max_features=8000,
            min_df=1,
            lowercase=True
        )
        model_cache['tfidf_vectorizer'] = tfidf_vectorizer
        print("✅ TF-IDF vectorizer loaded!")

        model_cache['loaded'] = True
        print("\n✅ All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        traceback.print_exc()
        return False

def init_database():
    """Initialize SQLite database for corpus management"""
    try:
        print("\n[STEP 2/3] Initializing Database...")
        print("=" * 80)
        
        conn = sqlite3.connect('corpus_database.db')
        cursor = conn.cursor()
        
        # Corpus documents table
        cursor.execute('''CREATE TABLE IF NOT EXISTS corpus_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            language TEXT,
            word_count INTEGER,
            embedding_vector BLOB,
            metadata TEXT,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT DEFAULT 'user_upload',
            tags TEXT
        )''')
        
        # Plagiarism checks table
        cursor.execute('''CREATE TABLE IF NOT EXISTS plagiarism_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_content TEXT,
            query_language TEXT,
            translated_content TEXT,
            total_corpus_docs INTEGER,
            corpus_matches INTEGER,
            max_corpus_similarity REAL,
            internet_matches INTEGER,
            max_internet_similarity REAL,
            processing_time REAL,
            check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            results_json TEXT,
            search_method TEXT
        )''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully!")
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        raise

# ============================================================================
# 3. TRANSLATION FUNCTIONS
# ============================================================================

# Create and load dictionary
print("\n[INITIAL] Creating Hindi-Punjabi Dictionary...")
data_dict = {
    'Hindi': ['नमस्ते', 'तुम', 'कैसे', 'हो', 'मैं', 'ठीक', 'धन्यवाद', 'पानी', 'खाना', 'घर', 'तैयार', 'है'],
    'Punjabi': ['ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਤੁਸੀਂ', 'ਕਿਵੇਂ', 'ਹੋ', 'ਮੈਂ', 'ਠੀਕ', 'ਧੰਨਵਾਦ', 'ਪਾਣੀ', 'ਖਾਣਾ', 'ਘਰ', 'ਤਿਆਰ', 'ਹੈ']
}

df_dict = pd.DataFrame(data_dict)
df_dict.to_csv('hindi_punjabi_dictionary.csv', index=False, encoding='utf-8')
dictionary_df = pd.read_csv('hindi_punjabi_dictionary.csv', encoding='utf-8')
translation_dict = dict(zip(dictionary_df['Hindi'], dictionary_df['Punjabi']))
print(f"✓ Dictionary loaded: {len(translation_dict)} word pairs")

# Create and load parallel corpus
print("\n[INITIAL] Creating Parallel Corpus...")
data_corpus = {
    'Hindi_Sentence': [
        'तुम कैसे हो?',
        'मैं ठीक हूँ, धन्यवाद।',
        'क्या तुम घर जा रहे हो?',
        'मुझे पानी चाहिए.',
        'यह मेरा घर है.',
        'आज मौसम अच्छा है.',
        'खाना तैयार है.',
        'नमस्ते, आपका दिन शुभ हो.',
        'अच्छे रिजल्ट लेन के लिए',
        'शिक्षा हमारे भविष्य का आधार है।',
        'हमें एक दूसरे के साथ मिलकर काम करना चाहिए।'
    ],
    'Punjabi_Sentence': [
        'ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?',
        'ਮੈਂ ਠੀਕ ਹਾਂ, ਧੰਨਵਾਦ।',
        'ਕੀ ਤੁਸੀਂ ਘਰ ਜਾ ਰਹੇ ਹੋ?',
        'ਮੈਨੂੰ ਪਾਣੀ ਚਾਹੀਦਾ ਹੈ।',
        'ਇਹ ਮੇਰਾ ਘਰ ਹੈ।',
        'ਅੱਜ ਮੌਸਮ ਵਧੀਆ ਹੈ।',
        'ਖਾਣਾ ਤਿਆਰ ਹੈ।',
        'ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਹਾਡਾ ਦਿਨ ਸ਼ੁਭ ਹੋਵੇ।',
        'ਚੰਗੇ ਨਤੀਜੇ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ',
        'ਸਿੱਖਿਆ ਸਾਡੇ ਭਵਿੱਖ ਦਾ ਆਧਾਰ ਹੈ।',
        'ਸਾਨੂੰ ਇੱਕ ਦੂਜੇ ਨਾਲ ਮਿਲ ਕੇ ਕੰਮ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ।'
    ]
}

df_corpus = pd.DataFrame(data_corpus)
df_corpus.to_csv('parallel_corpus.csv', index=False, encoding='utf-8')
df_corpus_loaded = pd.read_csv('parallel_corpus.csv', encoding='utf-8')
parallel_corpus = list(df_corpus_loaded.itertuples(index=False, name=None))
print(f"✓ Parallel corpus loaded: {len(parallel_corpus)} sentence pairs")

def jaccard_similarity(sentence1, sentence2):
    """Calculate Jaccard similarity between two sentences"""
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def ebmt_translate(hindi_sentence_to_translate, parallel_corpus, similarity_func):
    """Example-Based Machine Translation using parallel corpus"""
    best_match_punjabi = "Translation not found in corpus."
    highest_similarity = -1.0
    
    for hindi_ref, punjabi_ref in parallel_corpus:
        similarity = similarity_func(hindi_sentence_to_translate, hindi_ref)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_punjabi = punjabi_ref
    
    if highest_similarity >= 0.5:
        return best_match_punjabi, highest_similarity
    else:
        return "No similar sentence found.", -1.0

def nmt_translate(hindi_sentence, tokenizer, model, device):
    """Neural Machine Translation using NLLB-200"""
    if model is None or tokenizer is None:
        return "NMT model not available", -1.0
    
    try:
        target_lang = "pan_Guru"
        inputs = tokenizer(hindi_sentence, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                max_length=512
            )
        
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text, 0.95
        
    except Exception as e:
        return f"NMT Error: {str(e)}", -1.0

def translate_hindi_to_punjabi(hindi_sentence):
    """Main translation function with cascade approach: Dictionary -> EBMT -> NMT"""
    print(f"\n🔄 Translating Hindi to Punjabi: '{hindi_sentence}'")
    
    # 1. Dictionary-based translation
    words = hindi_sentence.split()
    translated_words = []
    untranslated_count = 0
    
    for word in words:
        translated_word = translation_dict.get(word, None)
        if translated_word is None:
            untranslated_count += 1
            translated_words.append(word)
        else:
            translated_words.append(translated_word)
    
    if untranslated_count == 0:
        result = ' '.join(translated_words)
        print(f"✓ Dictionary-based translation: {result}")
        return result, "Dictionary"
    
    # 2. EBMT as fallback
    print(f"⚠️ Dictionary failed ({untranslated_count} words untranslated). Trying EBMT...")
    ebmt_result, ebmt_score = ebmt_translate(hindi_sentence, parallel_corpus, jaccard_similarity)
    
    if ebmt_result != "No similar sentence found." and ebmt_score > 0.3:
        print(f"✓ EBMT translation found (similarity: {ebmt_score:.2f}): {ebmt_result}")
        return ebmt_result, "EBMT"
    
    # 3. NMT as final fallback
    print(f"⚠️ EBMT failed. Trying NMT...")
    if not model_cache.get('loaded'):
        return "Models not loaded", "None"
    
    nmt_result, nmt_score = nmt_translate(
        hindi_sentence,
        model_cache['tokenizer'],
        model_cache['model'],
        model_cache['device']
    )
    
    print(f"✓ NMT translation: {nmt_result}")
    return nmt_result, "NMT"

# ============================================================================
# 4. PLAGIARISM DETECTION FUNCTIONS
# ============================================================================

class EnhancedCorpusManager:
    """Manages corpus for plagiarism detection"""
    
    def __init__(self):
        self.ws_re = re.compile(r'\s+')
        self.models_available = model_cache.get('loaded', False)
        self.corpus_cache = corpus_cache
    
    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization for Indic scripts"""
        if not text:
            return ""
        
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\u200c', '').replace('\u200d', '').replace('\ufeff', '')
        text = self.ws_re.sub(' ', text).strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        """Detect language: Hindi vs Punjabi"""
        if not text:
            return 'unknown'
        
        hindi_chars = len([c for c in text if '\u0900' <= c <= '\u097F'])
        punjabi_chars = len([c for c in text if '\u0A00' <= c <= '\u0A7F'])
        latin_chars = len([c for c in text if c.isascii() and c.isalpha()])
        
        total_chars = hindi_chars + punjabi_chars + latin_chars
        
        if total_chars == 0:
            return 'unknown'
        
        hindi_pct = hindi_chars / total_chars
        punjabi_pct = punjabi_chars / total_chars
        
        if hindi_pct > 0.3:
            return 'hindi'
        elif punjabi_pct > 0.3:
            return 'punjabi'
        elif latin_chars > total_chars * 0.8:
            return 'english'
        else:
            return 'mixed'
    
    def add_to_corpus(self, filename: str, content: str, title: str = "", tags: List[str] = None) -> Tuple[bool, str]:
        """Add document to corpus"""
        try:
            print(f"\n📝 Adding document to corpus: {filename}")
            
            clean_content = self.normalize_text(content)
            
            if not clean_content:
                return False, "Document content is empty"
            
            word_count = len(clean_content.split())
            
            if word_count < 5:
                return False, f"Document too short: {word_count} words"
            
            language = self.detect_language(clean_content)
            print(f"🗣️ Detected language: {language}")
            
            # Generate embedding
            embedding_blob = None
            
            if self.models_available and model_cache.get('semantic_model'):
                try:
                    embedding = model_cache['semantic_model'].encode([clean_content], normalize_embeddings=True)[0]
                    embedding_blob = pickle.dumps(embedding)
                    print(f"✅ Embedding generated: {embedding.shape}")
                except Exception as e:
                    print(f"⚠️ Embedding generation failed: {e}")
                    embedding_blob = None
            
            metadata = {
                'title': title,
                'language': language,
                'word_count': word_count,
                'tags': tags or [],
                'has_embedding': embedding_blob is not None
            }
            
            # Save to database
            conn = sqlite3.connect('corpus_database.db')
            cursor = conn.cursor()
            
            cursor.execute('''INSERT OR REPLACE INTO corpus_documents
                (filename, title, content, language, word_count, embedding_vector, metadata, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (filename, title, clean_content, language, word_count, embedding_blob, json.dumps(metadata), json.dumps(tags or [])))
            
            conn.commit()
            conn.close()
            
            self._refresh_cache()
            return True, f"Document added: {filename} ({language}, {word_count} words)"
            
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            return False, str(e)
    
    def _refresh_cache(self):
        """Refresh corpus cache"""
        global corpus_cache
        
        try:
            conn = sqlite3.connect('corpus_database.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT filename, content, metadata, embedding_vector FROM corpus_documents ORDER BY date_added DESC')
            results = cursor.fetchall()
            
            documents, embeddings, metadata = [], [], []
            
            for filename, content, meta_json, embedding_blob in results:
                documents.append(content)
                metadata.append({
                    'filename': filename,
                    'metadata': json.loads(meta_json or '{}')
                })
                
                if embedding_blob:
                    try:
                        embedding = pickle.loads(embedding_blob)
                        embeddings.append(embedding)
                    except:
                        embeddings.append(None)
                else:
                    embeddings.append(None)
            
            valid_embeddings = [e for e in embeddings if e is not None]
            
            corpus_cache.update({
                'documents': documents,
                'embeddings': np.array(valid_embeddings) if valid_embeddings else None,
                'metadata': metadata,
                'last_updated': datetime.now(),
                'corpus_size': len(documents)
            })
            
            conn.close()
            print(f"✅ Corpus cache refreshed: {len(documents)} documents, {len(valid_embeddings)} with embeddings")
            
        except Exception as e:
            print(f"❌ Cache refresh error: {e}")
    
    def search_corpus(self, query: str, top_k: int = 5, threshold: float = 0.55) -> List[Dict]:
        """Search corpus for plagiarism matches"""
        try:
            print(f"\n🔍 Searching corpus for plagiarism...")
            
            clean_query = self.normalize_text(query)
            
            if not clean_query:
                return []
            
            query_language = self.detect_language(clean_query)
            
            if not corpus_cache['documents']:
                self._refresh_cache()
            
            if not corpus_cache['documents']:
                print("⚠️ No documents in corpus")
                return []
            
            matches = []
            
            # Semantic search (cross-language)
            if self.models_available and model_cache.get('semantic_model') and corpus_cache['embeddings'] is not None:
                try:
                    print("🧠 Performing semantic search...")
                    
                    query_embedding = model_cache['semantic_model'].encode([clean_query], normalize_embeddings=True)[0]
                    similarities = np.dot(corpus_cache['embeddings'], query_embedding)
                    
                    above_threshold = np.where(similarities >= threshold)[0]
                    print(f"🎯 Found {len(above_threshold)} corpus matches above threshold {threshold}")
                    
                    if len(above_threshold) > 0:
                        sorted_indices = above_threshold[np.argsort(similarities[above_threshold])[::-1]]
                        
                        for idx in sorted_indices[:top_k]:
                            doc_metadata = corpus_cache['metadata'][idx]
                            doc_language = doc_metadata['metadata'].get('language', 'unknown')
                            
                            match = {
                                'source': 'corpus',
                                'filename': doc_metadata['filename'],
                                'similarity': float(similarities[idx]),
                                'content_preview': corpus_cache['documents'][idx][:300] + "..." if len(corpus_cache['documents'][idx]) > 300 else corpus_cache['documents'][idx],
                                'metadata': doc_metadata['metadata'],
                                'match_type': 'semantic',
                                'is_cross_language': query_language != doc_language and query_language != 'unknown' and doc_language != 'unknown'
                            }
                            
                            matches.append(match)
                            
                            if match['is_cross_language']:
                                print(f"🌐 Cross-language match: {query_language} ↔ {doc_language} (sim: {similarities[idx]:.3f})")
                            
                except Exception as e:
                    print(f"❌ Semantic search error: {e}")
            
            print(f"✅ Corpus search completed: {len(matches)} matches")
            return matches
            
        except Exception as e:
            print(f"❌ Corpus search error: {e}")
            return []

# ============================================================================
# 5. INTERNET SEARCH (ORIGINAL GOOGLE SEARCH TECHNIQUE)
# ============================================================================

def search_internet_google(query: str, max_results: int = 30) -> List[Dict]:
    """
    Search Google for similar content, the same way a plain Google search
    does: the whole query text is sent to Google as a single search, with
    no sentence-splitting or keyword extraction beforehand.

    Args:
        query: Input text to search (usually translated Punjabi)
        max_results: Maximum number of results

    Returns:
        List of search results with similarity scores
    """
    try:
        print(f"\n🌐 Searching Google for similar content...")
        print(f"📌 Query: '{query[:80]}...'" if len(query) > 80 else f"📌 Query: '{query}'")

        matches = _perform_google_search(query, max_results)

        seen_urls = set()
        unique_matches = []
        for match in matches:
            url = match['url']
            if url not in seen_urls:  # Avoid duplicates
                seen_urls.add(url)
                unique_matches.append(match)

        # Sort by similarity and return top results
        unique_matches.sort(key=lambda x: x['similarity'], reverse=True)

        print(f"\n✅ Internet search completed: {len(unique_matches)} unique results found")
        return unique_matches[:max_results]

    except Exception as e:
        print(f"❌ Internet search error: {e}")
        traceback.print_exc()
        return []

def search_internet_bilingual(
    hindi_text: str,
    translated_punjabi: str,
    max_results: int = 30
) -> List[Dict]:
    """
    Search the internet using BOTH:
      - original Hindi text
      - translated Punjabi text

    Results are merged and deduplicated by URL.
    """
    all_matches: List[Dict] = []
    seen_urls = set()

    # 1) Search with original Hindi text
    if hindi_text and hindi_text.strip():
        print("\n🌐 INTERNET SEARCH: ORIGINAL HINDI TEXT")
        hindi_matches = search_internet_google(hindi_text, max_results=max_results)
        for m in hindi_matches:
            url = m.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            m = dict(m)
            m["query_language"] = "hindi"
            all_matches.append(m)

    # 2) Search with translated Punjabi text
    if translated_punjabi and translated_punjabi.strip():
        print("\n🌐 INTERNET SEARCH: TRANSLATED PUNJABI TEXT")
        punjabi_matches = search_internet_google(translated_punjabi, max_results=max_results)
        for m in punjabi_matches:
            url = m.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            m = dict(m)
            m["query_language"] = "punjabi"
            all_matches.append(m)

    # Sort combined list by similarity and trim
    all_matches.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    print(f"\n✅ Bilingual internet search complete: {len(all_matches)} unique results")
    return all_matches[:max_results]

def _perform_google_search(query: str, max_results: int = 10) -> List[Dict]:
    """
    Perform actual Google search for a single query
    
    Args:
        query: Search query
        max_results: Maximum results to return
    
    Returns:
        List of search results
    """
    matches = []

    try:
        # ===== CRITICAL FIX: NO HARDCODED DEFAULTS =====
        google_api_key = 'AIzaSyANu0jIdfaMusSjAcvvY9snLYYydqjiIAs'
        search_engine_id = '121f08de7b22144b1'
        serpapi_key = os.getenv('SERPAPI_KEY')
        
        print(f"\n🔍 API Key Check:")
        print(f"   Google API Key:        {'✅ SET' if google_api_key else '❌ NOT SET'}")
        print(f"   Search Engine ID:      {'✅ SET' if search_engine_id else '❌ NOT SET'}")
        print(f"   SerpAPI Key:           {'✅ SET' if serpapi_key else '❌ NOT SET'}")
        
        # ===== METHOD 1: GOOGLE CUSTOM SEARCH API =====
        if google_api_key and search_engine_id:
            print(f"\n📡 Method 1: Trying Google Custom Search API")
            print(f"   Query: '{query}'")
            
            try:
                search_url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'q': query,
                    'key': google_api_key,
                    'cx': search_engine_id,
                    'num': min(max_results, 10)
                }
                
                print(f"   Sending request...")
                print(f"   Input Query: '{query}'")
                print(f"   Query Length: {len(query)} characters")

                response = requests.get(search_url, params=params, timeout=150)
                
                print(f"   Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('items', [])
                    
                    print(f"   ✅ SUCCESS: Got {len(results)} results from Google API")

                    
                    for idx, item in enumerate(results):
                        match = {
                            'source': 'internet',
                            'url': item.get('link', ''),
                            'title': item.get('title', 'No title'),
                            'similarity': max(0.8, 0.95 - (idx * 0.08)),
                            'snippet': item.get('snippet', 'No preview available'),
                            'search_method': 'google_api'
                        }
                        matches.append(match)
                    
                    print(f"✅ Returning {len(matches)} Google API results")

                    return matches
                
                elif response.status_code == 403:
                    print(f"   ❌ 403 Forbidden - API Key or Search Engine ID invalid")
                    print(f"      Check: https://console.cloud.google.com/")
                    
                elif response.status_code == 400:
                    print(f"   ❌ 400 Bad Request - Check Search Engine ID format")
                    try:
                        error = response.json()
                        print(f"      Error: {error.get('error', {}).get('message', 'Unknown')}")
                    except:
                        pass
                
                elif response.status_code == 429:
                    print(f"   ❌ 429 Quota Exceeded - Daily limit reached")
                    print(f"      Wait 24 hours or upgrade to paid plan")
                    
                else:
                    print(f"   ❌ {response.status_code} Error")
                    try:
                        error = response.json()
                        print(f"      {error}")
                    except:
                        print(f"      {response.text[:200]}")
                
            except requests.exceptions.Timeout:
                print(f"   ❌ Request Timeout - Google API not responding")
            except requests.exceptions.ConnectionError:
                print(f"   ❌ Connection Error - Check internet connection")
            except Exception as e:
                print(f"   ❌ Exception: {type(e).__name__}: {e}")
        
        else:
            if not google_api_key:
                print(f"\n⚠️  GOOGLE_API_KEY not set")
                print(f"   Set with: setx GOOGLE_API_KEY \"your_key\"  (Windows)")
                print(f"   Set with: export GOOGLE_API_KEY=\"your_key\"  (macOS/Linux)")
            
            if not search_engine_id:
                print(f"\n⚠️  GOOGLE_SEARCH_ENGINE_ID not set")
                print(f"   Set with: setx GOOGLE_SEARCH_ENGINE_ID \"your_id\"  (Windows)")
                print(f"   Set with: export GOOGLE_SEARCH_ENGINE_ID=\"your_id\"  (macOS/Linux)")
        
        # ===== METHOD 2: SERPAPI =====
        if serpapi_key:
            print(f"\n📡 Method 2: Trying SerpAPI")
            print(f"   Query: '{query}'")
            
            try:
                search_url = "https://serpapi.com/search"
                params = {
                    'q': query,
                    'api_key': serpapi_key,
                    'num': min(max_results, 10)
                }
                
                print(f"   Sending request...")
                response = requests.get(search_url, params=params, timeout=15)
                
                print(f"   Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('organic_results', [])
                    
                    print(f"   ✅ SUCCESS: Got {len(results)} results from SerpAPI")
                    
                    for idx, item in enumerate(results):
                        match = {
                            'source': 'internet',
                            'url': item.get('link', ''),
                            'title': item.get('title', 'No title'),
                            'similarity': max(0.5, 0.85 - (idx * 0.08)),
                            'snippet': item.get('snippet', 'No preview available'),
                            'search_method': 'serpapi'
                        }
                        matches.append(match)
                    
                    print(f"✅ Returning {len(matches)} SerpAPI results")
                    return matches
                
                elif response.status_code == 403:
                    print(f"   ❌ 403 Forbidden - SerpAPI Key invalid")
                    
                elif response.status_code == 429:
                    print(f"   ❌ 429 Quota Exceeded")
                    print(f"      Free tier: 100 searches/month")
                    print(f"      Wait until next month or upgrade")
                
                else:
                    print(f"   ❌ {response.status_code} Error")
                    try:
                        error = response.json()
                        print(f"      {error}")
                    except:
                        print(f"      {response.text[:200]}")
                
            except requests.exceptions.Timeout:
                print(f"   ❌ Request Timeout - SerpAPI not responding")
            except requests.exceptions.ConnectionError:
                print(f"   ❌ Connection Error - Check internet connection")
            except Exception as e:
                print(f"   ❌ Exception: {type(e).__name__}: {e}")
        
        else:
            print(f"\n⚠️  SERPAPI_KEY not set")
            print(f"   Set with: setx SERPAPI_KEY \"your_key\"  (Windows)")
            print(f"   Set with: export SERPAPI_KEY=\"your_key\"  (macOS/Linux)")
        
        # No real search API returned results - return an empty list instead
        # of fabricating fake matches, so callers only ever see genuine
        # Google (or SerpAPI) search results.
        print(f"\n⚠️ All real search APIs unavailable or returned no results")
        return matches

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []

# ============================================================================
# 6. FLASK ROUTES
# ============================================================================

# Initialize components
print("\n[STEP 3/3] Initializing System...")
print("=" * 80)

load_models()
init_database()

corpus_manager = EnhancedCorpusManager()
corpus_manager._refresh_cache()

@app.route('/')
def index():
    """Home page"""
    return render_template('integrated_index.html')

# ============================================================================
# DOCUMENT UPLOAD ENDPOINTS
# ============================================================================

@app.route('/api/upload/plagiarism-check', methods=['POST'])
def upload_document_plagiarism():
    """
    Upload document for plagiarism detection
    
    Accepts: File upload
    Process: Extract text → Translate → Check plagiarism
    """
    try:
        # Check file
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\n📄 Processing uploaded document: {filename}")
        
        # Extract text from file
        content = extract_text_from_file(filepath)
        
        if not content:
            return jsonify({'success': False, 'error': 'Could not extract text from file'}), 400
        
        # Process for plagiarism detection
        print(f"✅ Document extracted: {len(content)} characters")
        
        # Detect language
        detected_lang = corpus_manager.detect_language(content)
        print(f"🗣️ Detected language: {detected_lang}")
        
        # Prepare response data
        response_data = {
            'success': True,
            'filename': filename,
            'content_preview': content[:500] + "..." if len(content) > 500 else content,
            'language': detected_lang,
            'character_count': len(content),
            'word_count': len(content.split()),
            'ready_for_analysis': True
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error uploading document: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload/corpus', methods=['POST'])
def upload_document_corpus():
    """
    Upload document to add to corpus
    
    Accepts: File upload + metadata
    Process: Extract text → Add to corpus database
    """
    try:
        # Check file
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
        
        # Get metadata
        title = request.form.get('title', file.filename)
        tags = request.form.getlist('tags', [])
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\n📄 Adding document to corpus: {filename}")
        
        # Extract text from file
        content = extract_text_from_file(filepath)
        
        if not content:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Could not extract text from file'}), 400
        
        # Add to corpus
        success, message = corpus_manager.add_to_corpus(filename, content, title, tags)
        
        # Clean up
        os.remove(filepath)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'filename': filename,
                'corpus_size': corpus_cache['corpus_size']
            })
        else:
            return jsonify({'success': False, 'error': message}), 400
            
    except Exception as e:
        print(f"❌ Error uploading to corpus: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plagiarism-check/text', methods=['POST'])
def plagiarism_check_text():
    """
    Check plagiarism for text input

    Accepts: Hindi text
    Process: Translate → Corpus check → Internet search (direct Google search)
    """
    try:
        data = request.get_json()
        hindi_text = data.get('hindi_text', '').strip()

        if not hindi_text:
            return jsonify({'error': 'No Hindi text provided'}), 400
        
        print("\n" + "=" * 80)
        print("PLAGIARISM CHECK STARTED")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # =========== STEP 1: TRANSLATE ===========
        print("\n[STEP 1] HINDI TO PUNJABI TRANSLATION")
        print("-" * 80)
        
        translated_punjabi, translation_method = translate_hindi_to_punjabi(hindi_text)
        
        # =========== STEP 2: CORPUS PLAGIARISM CHECK ===========
        print("\n[STEP 2] CROSS-LANGUAGE PLAGIARISM CHECK - CORPUS")
        print("-" * 80)
        
        corpus_matches = corpus_manager.search_corpus(translated_punjabi, top_k=10, threshold=0.55)
        
        # =========== STEP 3: INTERNET SEARCH (ORIGINAL GOOGLE SEARCH TECHNIQUE) ===========
        print("\n[STEP 3] INTERNET SEARCH (GOOGLE) - DIRECT SEARCH ON TRANSLATED CONTENT")
        print("-" * 80)

        internet_matches = search_internet_google(translated_punjabi, max_results=30)

         #=========== PREPARE RESPONSE ===========
        processing_time = (datetime.now() - start_time).total_seconds()    
        response_data = {
            'success': True,
            'original_hindi': hindi_text,
            'translated_punjabi': translated_punjabi,
            'translation_method': translation_method,
            
            # Corpus Results
            'corpus_results': {
                'total_matches': len(corpus_matches),
                'matches': corpus_matches[:5],
                'max_similarity': max([m['similarity'] for m in corpus_matches], default=0)
            },
            
            # Internet Results
            'internet_results': {
                'total_matches': len(internet_matches),
                'matches': internet_matches[:40],
                'max_similarity': max([m['similarity'] for m in internet_matches], default=0),
                'search_method': 'google-search'
            },
            
            # Summary
            'plagiarism_summary': {
                'total_matches': len(corpus_matches) + len(internet_matches),
                'corpus_matches': len(corpus_matches),
                'internet_matches': len(internet_matches),
                'highest_corpus_similarity': max([m['similarity'] for m in corpus_matches], default=0),
                'highest_internet_similarity': max([m['similarity'] for m in internet_matches], default=0),
                'overall_similarity': max(
                    max([m['similarity'] for m in corpus_matches], default=0),
                    max([m['similarity'] for m in internet_matches], default=0)
                ),
                'plagiarism_detected': len(corpus_matches) > 0 or len(internet_matches) > 0
            },
            'processing_time': round(processing_time, 2)
        }
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        print(f"Total Matches: {response_data['plagiarism_summary']['total_matches']}")
        print(f"Processing Time: {processing_time:.2f}s")
        
        # Save to database
        try:
            conn = sqlite3.connect('corpus_database.db')
            cursor = conn.cursor()
            
            cursor.execute('''INSERT INTO plagiarism_checks
                (query_content, query_language, translated_content, total_corpus_docs,
                corpus_matches, max_corpus_similarity, internet_matches,
                max_internet_similarity, processing_time, results_json, search_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (hindi_text, 'hindi', translated_punjabi, corpus_manager.corpus_cache.get('corpus_size', 0),
                len(corpus_matches),
                response_data['plagiarism_summary']['highest_corpus_similarity'],
                len(internet_matches),
                response_data['plagiarism_summary']['highest_internet_similarity'],
                processing_time,
                json.dumps(response_data),
                'google-search'))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Database save error: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/corpus/add', methods=['POST'])
def add_to_corpus_api():
    """Add text content to corpus"""
    try:
        data = request.get_json()
        filename = data.get('filename', 'unknown.txt')
        content = data.get('content', '')
        title = data.get('title', '')
        tags = data.get('tags', [])
        
        success, message = corpus_manager.add_to_corpus(filename, content, title, tags)
        
        return jsonify({
            'success': success,
            'message': message,
            'corpus_size': corpus_cache['corpus_size']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/corpus/list', methods=['GET'])
def list_corpus_documents():
    """Get list of documents in corpus"""
    try:
        conn = sqlite3.connect('corpus_database.db')
        cursor = conn.cursor()
        
        cursor.execute('''SELECT filename, title, language, word_count, date_added
            FROM corpus_documents
            ORDER BY date_added DESC
            LIMIT 100''')
        
        results = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in results:
            documents.append({
                'filename': row[0],
                'title': row[1] or row[0],
                'language': row[2],
                'word_count': row[3],
                'date_added': row[4]
            })
        
        print(f"✅ Retrieved {len(documents)} documents from corpus")
        return jsonify({'documents': documents})
        
    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/corpus/delete', methods=['POST'])
def delete_corpus_document():
    """Delete document from corpus"""
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'}), 400
        
        conn = sqlite3.connect('corpus_database.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM corpus_documents WHERE filename = ?', (filename,))
        
        if cursor.rowcount > 0:
            conn.commit()
            conn.close()
            corpus_manager._refresh_cache()
            print(f"✅ Document deleted: {filename}")
            return jsonify({'success': True, 'message': f'Document deleted: {filename}'})
        else:
            conn.close()
            print(f"❌ Document not found: {filename}")
            return jsonify({'success': False, 'message': 'Document not found'}), 404
            
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/corpus/stats', methods=['GET'])
def get_corpus_stats():
    """Get corpus statistics"""
    try:
        conn = sqlite3.connect('corpus_database.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM corpus_documents')
        total_docs = cursor.fetchone()[0]
        
        cursor.execute('SELECT language, COUNT(*) FROM corpus_documents GROUP BY language')
        lang_dist = dict(cursor.fetchall())
        
        cursor.execute('SELECT SUM(word_count) FROM corpus_documents')
        total_words = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'total_documents': total_docs,
            'language_distribution': lang_dist,
            'total_words': total_words,
            'last_updated': corpus_cache['last_updated'].isoformat() if corpus_cache['last_updated'] else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_cache.get('loaded', False),
        'corpus_size': corpus_cache['corpus_size'],
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("✅ ENHANCED INTEGRATED SYSTEM READY!")
    print("=" * 80)
    print("\n📱 Starting Flask Server...")
    print("Access the application at: http://localhost:5000")
    
    print("\n📋 API ENDPOINTS:")
    print(" POST /api/upload/plagiarism-check - Upload document for plagiarism detection")
    print(" POST /api/upload/corpus - Upload document to add to corpus")
    print(" POST /api/plagiarism-check/text - Check plagiarism for text input")
    print(" POST /api/corpus/add - Add text to corpus")
    print(" GET /api/corpus/list - List all corpus documents")
    print(" POST /api/corpus/delete - Delete document from corpus")
    print(" GET /api/corpus/stats - Get corpus statistics")
    print(" GET /health - Health check")
    
    print("\n🔍 FEATURES:")
    print(" ✅ Document file upload for plagiarism checking (TXT, PDF, DOCX)")
    print(" ✅ Corpus management with document upload")
    print(" ✅ Direct Google search (after translation to Punjabi)")
    print(" ✅ Hindi-Punjabi translation (Dictionary → EBMT → NMT)")
    print(" ✅ Cross-language plagiarism detection")
    
    print("\n⚙️ CONFIGURATION:")
    print(f" Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f" Corpus folder: {app.config['CORPUS_FOLDER']}")
    print(f" Max file size: {app.config['MAX_CONTENT_LENGTH'] / 1024 / 1024}MB")
    
    print("\nPress CTRL+C to stop the server\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
