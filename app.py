# Integrated Hindi-Punjabi Translation + Cross-Language Plagiarism Detection System
# ENHANCED VERSION with Document Upload, Sentence-Based Search,
# Fine-tuned IndicBERT Semantic Similarity & Dual-Mode (EBMT + NMT) Translation Output

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
import random
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from werkzeug.utils import secure_filename
import traceback
import requests
from urllib.parse import quote
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

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
    print("Please run: pip install -r requirements.txt")
    exit(1)

# Download NLTK data for phrase extraction
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("="*80)
print("INTEGRATED: Hindi-Punjabi Translation + Cross-Language Plagiarism Detection")
print("ENHANCED: Document Upload + Sentence-Based Search + Fine-tuned IndicBERT")
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

# NMT engine choices available for Hindi->Punjabi translation. NLLB-200 is
# loaded eagerly at startup (see load_models()); IndicTrans2 is much larger
# and is loaded lazily on first actual use instead, so choosing it costs
# nothing for callers who never ask for it.
NMT_MODEL_CHOICES = ('nllb', 'indictrans2')
DEFAULT_NMT_MODEL = 'nllb'
INDICTRANS2_MODEL_NAME = 'ai4bharat/indictrans2-indic-indic-1B'

def _resolve_nmt_model(value) -> str:
    """Normalize a caller-supplied NMT engine choice, falling back to the default for anything unrecognized."""
    value = (value or '').strip().lower()
    return value if value in NMT_MODEL_CHOICES else DEFAULT_NMT_MODEL

# Only surface a corpus/internet result as a plagiarism match once its
# similarity reaches this threshold - keeps the UI focused on strong,
# high-confidence matches instead of every low-similarity hit.
PLAGIARISM_SIMILARITY_THRESHOLD = 0.80

# Threshold for the translated-Punjabi leg of internet search specifically,
# which runs keyword-based (not semantic) matching - a lower bar than the
# Hindi-source leg since keyword/phrase search naturally scores lower than
# a full-sentence semantic comparison even for a real match.
PUNJABI_INTERNET_SIMILARITY_THRESHOLD = 0.65

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

def safe_upload_filename(original_filename: str) -> str:
    """
    Build a filesystem-safe filename for an uploaded file while guaranteeing
    it keeps a valid extension. werkzeug's secure_filename() strips all
    non-ASCII characters - common in Hindi/Punjabi document names - and if
    that empties the base name, it also strips the separating dot (e.g.
    'मेरा_दस्तावेज़.txt' -> 'txt'), losing the extension entirely. Re-attach
    the extension explicitly, using a generated base name if sanitizing the
    original left nothing usable.
    Caller must have already validated the original filename with allowed_file().
    """
    ext = original_filename.rsplit('.', 1)[1].lower()
    base = secure_filename(original_filename.rsplit('.', 1)[0])
    if not base:
        base = uuid.uuid4().hex[:12]
    return f"{base}.{ext}"

def extract_text_from_file(filepath):
    """Extract text from various file formats"""
    filename = os.path.basename(filepath)
    if '.' not in filename:
        print(f"❌ Cannot determine file type for '{filename}' (no extension)")
        return None
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
# 1. SENTENCE-BASED SEARCH EXTRACTOR
# ============================================================================

# Devanagari/Gurmukhi sentences typically end in "।" (danda), which nltk's
# English-trained punkt tokenizer does not recognize as a sentence boundary -
# text with no ASCII "." lands in a single giant "sentence" instead of being
# split. This lightweight regex splits on the actual Hindi/Punjabi/Latin
# sentence-enders and is used both for EBMT matching and for sentence-based
# internet search, so a multi-sentence Hindi/Punjabi document is searched
# sentence-by-sentence instead of as one long, ineffective query.
_SENTENCE_END_RE = re.compile(r'(?<=[।!?.])\s+')

def split_into_sentences(text):
    """Split text into sentences, keeping the terminating punctuation attached."""
    text = (text or '').strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]

# Common Punjabi/Hindi function words (pronouns, postpositions, auxiliary
# verbs, conjunctions) - stripped out when building keyword search queries.
# A real person searching Google types the distinctive content words from a
# sentence, not its grammatical filler; leaving stopwords in produces vague
# queries that match loosely-related pages instead of the actual topic.
_INDIC_STOPWORDS = {
    # Punjabi
    'ਹੈ', 'ਹਨ', 'ਸੀ', 'ਸਨ', 'ਹੋ', 'ਹੋਵੇ', 'ਹੋਵੇਗਾ', 'ਹੋਵੇਗੀ',
    'ਦਾ', 'ਦੀ', 'ਦੇ', 'ਨੂੰ', 'ਨੇ', 'ਤੇ', 'ਤੋਂ', 'ਵਿੱਚ', 'ਵਿਚ', 'ਨਾਲ', 'ਲਈ',
    'ਅਤੇ', 'ਜਾਂ', 'ਪਰ', 'ਕਿ', 'ਜੋ', 'ਵੀ', 'ਹੀ', 'ਨਾ',
    'ਇੱਕ', 'ਇਹ', 'ਇਸ', 'ਉਹ', 'ਉਸ', 'ਓ',
    'ਮੈਂ', 'ਤੁਸੀਂ', 'ਅਸੀਂ', 'ਉਸਦਾ', 'ਉਸਦੀ', 'ਮੇਰਾ', 'ਮੇਰੀ',
    'ਤੁਹਾਡਾ', 'ਤੁਹਾਡੀ', 'ਸਾਡਾ', 'ਸਾਡੀ',
    'ਕਰ', 'ਕਰਨ', 'ਕਰਦਾ', 'ਕਰਦੀ', 'ਹੁਣ',
    # Hindi (shared helper, so the Hindi-source leg benefits too)
    'है', 'हैं', 'था', 'थी', 'थे', 'हो', 'होगा', 'होगी',
    'का', 'की', 'के', 'को', 'ने', 'से', 'में', 'तक', 'पर',
    'और', 'या', 'पर', 'कि', 'जो', 'भी', 'ही', 'ना', 'नहीं',
    'एक', 'यह', 'इस', 'वह', 'उस',
    'मैं', 'तुम', 'हम', 'आप', 'उसका', 'उसकी', 'मेरा', 'मेरी',
    'तुम्हारा', 'तुम्हारी', 'हमारा', 'हमारी',
    'कर', 'करना', 'करता', 'करती', 'अभी',
}

class SentenceBasedSearcher:
    """Extract and create sentence-based search queries"""

    def __init__(self):
        self.min_sentence_length = 3  # Minimum words in a sentence

    def extract_sentences(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract important sentences from text

        Args:
            text: Input text
            top_n: Number of top sentences to extract

        Returns:
            List of extracted sentences
        """
        if not text:
            return []

        try:
            # NLTK's punkt tokenizer is trained on English punctuation and
            # does not recognise the Devanagari/Gurmukhi sentence-ending
            # "।" (danda), so Hindi/Punjabi text with no ASCII "." lands in
            # a single giant "sentence" - defeating sentence-based search.
            # Prefer the Indic-aware splitter (also used for EBMT matching)
            # and only fall back to sent_tokenize if it finds nothing.
            sentences = split_into_sentences(text) or sent_tokenize(text)

            # Filter sentences by minimum word count
            valid_sentences = [
                s.strip()
                for s in sentences
                if len(s.split()) >= self.min_sentence_length and len(s.strip()) > 5
            ]

            print(f"🔤 Extracted {len(valid_sentences)} sentences from text")

            # Return top N sentences
            return valid_sentences[:top_n]
        except Exception as e:
            print(f"⚠️ Sentence extraction error: {e}")
            return []

    def create_sentence_queries(self, text: str, num_queries: int = 5) -> List[str]:
        """
        Create search queries based on sentences instead of keywords

        Args:
            text: Input text (usually Punjabi translated text)
            num_queries: Number of sentence queries to generate

        Returns:
            List of sentence-based search queries
        """
        sentences = self.extract_sentences(text, top_n=num_queries * 2)

        search_queries = []
        for sentence in sentences[:num_queries]:
            # Clean sentence for search
            clean_sentence = sentence.strip()
            clean_sentence = re.sub(r'[.,!?;:\'"]+$', '', clean_sentence).strip()
            if len(clean_sentence) > 10:  # Only sentences with meaningful length
                search_queries.append(clean_sentence)

        print(f"📋 Created {len(search_queries)} sentence-based search queries:")
        for i, q in enumerate(search_queries, 1):
            print(f"   {i}. {q[:80]}..." if len(q) > 80 else f"   {i}. {q}")

        return search_queries if search_queries else [text[:150]]

    def create_keyword_queries(self, text: str, num_queries: int = 5) -> List[str]:
        """
        Create keyword search queries the way a person actually searches
        Google: drop the grammatical filler (pronouns, postpositions,
        auxiliary verbs, conjunctions) and keep the distinctive content
        words from each sentence, in their original order.

        The earlier approach took a fixed 5-word window from the middle of
        the sentence regardless of content, which often produced a phrase
        made mostly of function words (e.g. "ਵਿੱਚ ਸਥਿਤ ਇੱਕ") - vague enough
        that Google matches many loosely-related pages instead of the
        actual topic. Stripping stopwords first keeps the query anchored to
        what the sentence is actually about.

        Args:
            text: Input text
            num_queries: Number of keyword queries to generate (one per
                sentence, at most)

        Returns:
            List of natural, content-word keyword search queries
        """
        sentences = self.extract_sentences(text, top_n=num_queries)

        keyword_queries = []
        for sentence in sentences:
            words = [re.sub(r'[.,!?;:\'"।]+$', '', w).strip() for w in sentence.split()]
            content_words = [w for w in words if w and len(w) > 1 and w not in _INDIC_STOPWORDS]

            if not content_words:
                continue

            # Keep at most 6 content words - enough to stay distinctive
            # without turning back into a near-full-sentence query.
            phrase = ' '.join(content_words[:6])
            if len(phrase) > 6:
                keyword_queries.append(phrase)

        print(f"🔑 Created {len(keyword_queries)} keyword-based search queries (stopwords removed):")
        for i, q in enumerate(keyword_queries, 1):
            print(f"   {i}. {q}")

        return keyword_queries[:num_queries]

    def create_hybrid_queries(self, text: str, num_queries: int = 5) -> List[Dict]:
        """
        Create hybrid search queries (sentences + multi-word phrases)

        Args:
            text: Input text
            num_queries: Number of queries

        Returns:
            List of query dictionaries with type and content
        """
        sentences = self.extract_sentences(text, top_n=num_queries)

        hybrid_queries = []
        for i, sentence in enumerate(sentences):
            # Extract important phrases from sentence
            words = sentence.split()

            # Take 2-4 consecutive words as phrases
            phrases = []
            for j in range(len(words) - 1):
                phrase = ' '.join(words[j:min(j+4, len(words))])
                if len(phrase) > 10:
                    phrases.append(phrase)

            query_obj = {
                'type': 'sentence',
                'query': sentence.strip(),
                'phrases': phrases[:3] if phrases else [],
                'priority': 'high' if i < 3 else 'medium'
            }
            hybrid_queries.append(query_obj)

        return hybrid_queries

# ============================================================================
# 2. MODEL LOADING & INITIALIZATION
# ============================================================================

def load_models():
    """Load all required models for translation and plagiarism detection"""
    global model_cache

    if model_cache.get('loaded'):
        return True

    try:
        print("\n[STEP 1/3] Loading Models...")
        print("=" * 80)

        # Load Translation Models. NLLB-200-distilled-600M needs a few GB of
        # RAM to download and load; on a memory-constrained machine the OS
        # OOM-killer can terminate the process here with no Python
        # traceback (nothing to catch - it's a SIGKILL). Set SKIP_NMT_MODEL=1
        # to bypass this entirely and run in Dictionary+EBMT-only mode.
        if os.getenv('SKIP_NMT_MODEL', '').lower() in ('1', 'true', 'yes'):
            print("\n📖 SKIP_NMT_MODEL is set - skipping NLLB-200 load. "
                  "NMT translation will be unavailable; Dictionary and EBMT still work.")
            model_cache['tokenizer'] = None
            model_cache['model'] = None
            model_cache['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            print("\n📖 Loading NLLB-200 Translation Model...")
            model_name = "facebook/nllb-200-distilled-600M"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            try:
                # low_cpu_mem_usage avoids holding a duplicate full-precision
                # copy of the weights in RAM while loading, roughly halving
                # peak memory use during this step (requires 'accelerate').
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
            except ImportError:
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

        # Initialize Sentence-Based Searcher
        model_cache['sentence_searcher'] = SentenceBasedSearcher()
        print("✅ Sentence-based searcher initialized!")

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

# ----------------------------------------------------------------------------
# 3a. Hindi-Punjabi Dictionary
# Expanded from the original 12 word-pairs to ~160 pairs covering pronouns,
# question words, numbers, days, time, family, common nouns, adjectives and
# verbs, so the dictionary-lookup stage covers realistic thesis test corpora.
# ----------------------------------------------------------------------------
print("\n[INITIAL] Creating Hindi-Punjabi Dictionary...")

_hindi_words = [
    # Greetings & common phrases
    'नमस्ते', 'धन्यवाद', 'शुक्रिया', 'माफ़ करना', 'कृपया', 'स्वागत', 'अलविदा',
    'शुभ रात्रि', 'शुभ प्रभात',
    # Pronouns & question words
    'मैं', 'तुम', 'आप', 'हम', 'वह', 'वे', 'यह', 'ये',
    'मुझे', 'तुम्हें', 'उसे', 'हमें',
    'मेरा', 'मेरी', 'तुम्हारा', 'तुम्हारी', 'हमारा', 'हमारी', 'उसका', 'उसकी',
    'कौन', 'क्या', 'कहाँ', 'कब', 'क्यों', 'कैसे', 'कितना', 'कितने', 'कौन-सा',
    # Verb "to be" / auxiliaries
    'हो', 'है', 'हैं', 'था', 'थी', 'थे', 'होगा', 'होगी', 'ठीक', 'तैयार', 'चाहिए',
    # Numbers
    'एक', 'दो', 'तीन', 'चार', 'पांच', 'छह', 'सात', 'आठ', 'नौ', 'दस', 'सौ', 'हज़ार',
    # Days of the week
    'सोमवार', 'मंगलवार', 'बुधवार', 'गुरुवार', 'शुक्रवार', 'शनिवार', 'रविवार',
    # Time
    'समय', 'दिन', 'रात', 'सुबह', 'शाम', 'दोपहर', 'आज', 'कल', 'परसों', 'अभी',
    'साल', 'महीना', 'हफ्ता', 'सप्ताह',
    # Family
    'माता', 'मां', 'पिता', 'भाई', 'बहन', 'बेटा', 'बेटी', 'दादा', 'दादी',
    'नाना', 'नानी', 'पति', 'पत्नी', 'दोस्त', 'परिवार', 'बच्चा', 'बच्चे',
    # Common nouns
    'पानी', 'खाना', 'घर', 'किताब', 'स्कूल', 'शहर', 'गांव', 'रास्ता', 'काम',
    'पैसा', 'बाज़ार', 'दुकान', 'गाड़ी', 'सड़क', 'फूल', 'पेड़', 'आसमान', 'धरती',
    'सूरज', 'चांद', 'तारा', 'हवा', 'बारिश', 'बर्फ', 'आग', 'दूध', 'चाय', 'रोटी',
    'चावल', 'सब्ज़ी', 'फल', 'दवाई', 'डॉक्टर', 'अस्पताल', 'शिक्षक', 'विद्यार्थी',
    'किसान', 'मज़दूर', 'सरकार', 'देश', 'भाषा', 'अखबार', 'मोबाइल', 'कंप्यूटर',
    # Adjectives
    'अच्छा', 'बुरा', 'बड़ा', 'छोटा', 'नया', 'पुराना', 'सुंदर', 'गंदा', 'गर्म',
    'ठंडा', 'तेज़', 'धीमा', 'सस्ता', 'महंगा', 'खुश', 'दुखी', 'आसान', 'मुश्किल',
    # Verbs
    'जाना', 'आना', 'पीना', 'सोना', 'उठना', 'बैठना', 'चलना', 'दौड़ना', 'देखना',
    'सुनना', 'बोलना', 'पढ़ना', 'लिखना', 'समझना', 'सीखना', 'सिखाना', 'देना',
    'लेना', 'कहना', 'पूछना', 'मिलना', 'बताना', 'रहना', 'खेलना', 'हंसना', 'रोना',
]

_punjabi_words = [
    # Greetings & common phrases
    'ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਧੰਨਵਾਦ', 'ਸ਼ੁਕਰੀਆ', 'ਮਾਫ਼ ਕਰਨਾ', 'ਕਿਰਪਾ ਕਰਕੇ', 'ਸੁਆਗਤ', 'ਅਲਵਿਦਾ',
    'ਸ਼ੁਭ ਰਾਤ', 'ਸ਼ੁਭ ਸਵੇਰ',
    # Pronouns & question words
    'ਮੈਂ', 'ਤੁਸੀਂ', 'ਤੁਸੀਂ', 'ਅਸੀਂ', 'ਉਹ', 'ਉਹ', 'ਇਹ', 'ਇਹ',
    'ਮੈਨੂੰ', 'ਤੁਹਾਨੂੰ', 'ਉਸਨੂੰ', 'ਸਾਨੂੰ',
    'ਮੇਰਾ', 'ਮੇਰੀ', 'ਤੁਹਾਡਾ', 'ਤੁਹਾਡੀ', 'ਸਾਡਾ', 'ਸਾਡੀ', 'ਉਸਦਾ', 'ਉਸਦੀ',
    'ਕੌਣ', 'ਕੀ', 'ਕਿੱਥੇ', 'ਕਦੋਂ', 'ਕਿਉਂ', 'ਕਿਵੇਂ', 'ਕਿੰਨਾ', 'ਕਿੰਨੇ', 'ਕਿਹੜਾ',
    # Verb "to be" / auxiliaries
    'ਹੋ', 'ਹੈ', 'ਹਨ', 'ਸੀ', 'ਸੀ', 'ਸਨ', 'ਹੋਵੇਗਾ', 'ਹੋਵੇਗੀ', 'ਠੀਕ', 'ਤਿਆਰ', 'ਚਾਹੀਦਾ',
    # Numbers
    'ਇੱਕ', 'ਦੋ', 'ਤਿੰਨ', 'ਚਾਰ', 'ਪੰਜ', 'ਛੇ', 'ਸੱਤ', 'ਅੱਠ', 'ਨੌਂ', 'ਦਸ', 'ਸੌ', 'ਹਜ਼ਾਰ',
    # Days of the week
    'ਸੋਮਵਾਰ', 'ਮੰਗਲਵਾਰ', 'ਬੁੱਧਵਾਰ', 'ਵੀਰਵਾਰ', 'ਸ਼ੁੱਕਰਵਾਰ', 'ਸ਼ਨੀਵਾਰ', 'ਐਤਵਾਰ',
    # Time
    'ਸਮਾਂ', 'ਦਿਨ', 'ਰਾਤ', 'ਸਵੇਰ', 'ਸ਼ਾਮ', 'ਦੁਪਹਿਰ', 'ਅੱਜ', 'ਕੱਲ੍ਹ', 'ਪਰਸੋਂ', 'ਹੁਣੇ',
    'ਸਾਲ', 'ਮਹੀਨਾ', 'ਹਫ਼ਤਾ', 'ਹਫ਼ਤਾ',
    # Family
    'ਮਾਤਾ', 'ਮਾਂ', 'ਪਿਤਾ', 'ਭਰਾ', 'ਭੈਣ', 'ਪੁੱਤਰ', 'ਧੀ', 'ਦਾਦਾ', 'ਦਾਦੀ',
    'ਨਾਨਾ', 'ਨਾਨੀ', 'ਪਤੀ', 'ਪਤਨੀ', 'ਦੋਸਤ', 'ਪਰਿਵਾਰ', 'ਬੱਚਾ', 'ਬੱਚੇ',
    # Common nouns
    'ਪਾਣੀ', 'ਖਾਣਾ', 'ਘਰ', 'ਕਿਤਾਬ', 'ਸਕੂਲ', 'ਸ਼ਹਿਰ', 'ਪਿੰਡ', 'ਰਸਤਾ', 'ਕੰਮ',
    'ਪੈਸਾ', 'ਬਜ਼ਾਰ', 'ਦੁਕਾਨ', 'ਗੱਡੀ', 'ਸੜਕ', 'ਫੁੱਲ', 'ਰੁੱਖ', 'ਅਸਮਾਨ', 'ਧਰਤੀ',
    'ਸੂਰਜ', 'ਚੰਦ', 'ਤਾਰਾ', 'ਹਵਾ', 'ਮੀਂਹ', 'ਬਰਫ਼', 'ਅੱਗ', 'ਦੁੱਧ', 'ਚਾਹ', 'ਰੋਟੀ',
    'ਚੌਲ', 'ਸਬਜ਼ੀ', 'ਫਲ', 'ਦਵਾਈ', 'ਡਾਕਟਰ', 'ਹਸਪਤਾਲ', 'ਅਧਿਆਪਕ', 'ਵਿਦਿਆਰਥੀ',
    'ਕਿਸਾਨ', 'ਮਜ਼ਦੂਰ', 'ਸਰਕਾਰ', 'ਦੇਸ਼', 'ਭਾਸ਼ਾ', 'ਅਖ਼ਬਾਰ', 'ਮੋਬਾਈਲ', 'ਕੰਪਿਊਟਰ',
    # Adjectives
    'ਚੰਗਾ', 'ਮਾੜਾ', 'ਵੱਡਾ', 'ਛੋਟਾ', 'ਨਵਾਂ', 'ਪੁਰਾਣਾ', 'ਸੋਹਣਾ', 'ਗੰਦਾ', 'ਗਰਮ',
    'ਠੰਡਾ', 'ਤੇਜ਼', 'ਹੌਲੀ', 'ਸਸਤਾ', 'ਮਹਿੰਗਾ', 'ਖੁਸ਼', 'ਦੁਖੀ', 'ਸੌਖਾ', 'ਔਖਾ',
    # Verbs
    'ਜਾਣਾ', 'ਆਉਣਾ', 'ਪੀਣਾ', 'ਸੌਣਾ', 'ਉੱਠਣਾ', 'ਬੈਠਣਾ', 'ਤੁਰਨਾ', 'ਦੌੜਨਾ', 'ਵੇਖਣਾ',
    'ਸੁਣਨਾ', 'ਬੋਲਣਾ', 'ਪੜ੍ਹਨਾ', 'ਲਿਖਣਾ', 'ਸਮਝਣਾ', 'ਸਿੱਖਣਾ', 'ਸਿਖਾਉਣਾ', 'ਦੇਣਾ',
    'ਲੈਣਾ', 'ਕਹਿਣਾ', 'ਪੁੱਛਣਾ', 'ਮਿਲਣਾ', 'ਦੱਸਣਾ', 'ਰਹਿਣਾ', 'ਖੇਡਣਾ', 'ਹੱਸਣਾ', 'ਰੋਣਾ',
]

assert len(_hindi_words) == len(_punjabi_words), "Dictionary word-list lengths must match"

data_dict = {'Hindi': _hindi_words, 'Punjabi': _punjabi_words}

df_dict = pd.DataFrame(data_dict)
df_dict.to_csv('hindi_punjabi_dictionary.csv', index=False, encoding='utf-8')
dictionary_df = pd.read_csv('hindi_punjabi_dictionary.csv', encoding='utf-8')
translation_dict = dict(zip(dictionary_df['Hindi'], dictionary_df['Punjabi']))
print(f"✓ Dictionary loaded: {len(translation_dict)} word pairs")

# ----------------------------------------------------------------------------
# 3b. Parallel Corpus (EBMT examples)
# Expanded from 11 to 70 Hindi-Punjabi sentence pairs spanning greetings,
# daily life, family, education, weather, work, travel, shopping, health,
# technology and environment, for better EBMT coverage and to double as the
# custom dataset used to fine-tune IndicBERT (see train_indicbert.py).
# ----------------------------------------------------------------------------
print("\n[INITIAL] Creating Parallel Corpus...")

_hindi_sentences = [
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
    'हमें एक दूसरे के साथ मिलकर काम करना चाहिए।',
    'सुप्रभात, आपका दिन अच्छा रहे.',
    'मुझे भूख लगी है.',
    'मुझे नींद आ रही है.',
    'कृपया दरवाज़ा बंद कर दो.',
    'यह किताब बहुत अच्छी है.',
    'मैं स्कूल जा रहा हूँ.',
    'वह हर रोज़ पढ़ाई करता है.',
    'मेरा नाम राम है.',
    'तुम्हारा नाम क्या है?',
    'आज मेरा जन्मदिन है.',
    'कल बारिश होगी.',
    'सर्दियों में बर्फ पड़ती है.',
    'गर्मियों में बहुत गर्मी होती है.',
    'मुझे पंजाबी सीखना है.',
    'वह हिंदी बहुत अच्छी बोलता है.',
    'हमें मेहनत करनी चाहिए.',
    'सफलता मेहनत से मिलती है.',
    'समय बहुत कीमती है.',
    'मेरे पिताजी एक डॉक्टर हैं.',
    'मेरी माँ स्कूल में पढ़ाती हैं.',
    'मेरा भाई क्रिकेट खेलता है.',
    'मेरी बहन गाना गाती है.',
    'हम बाज़ार जा रहे हैं.',
    'मुझे यह कपड़ा पसंद है.',
    'यह कितने का है?',
    'कृपया थोड़ा सस्ता कर दो.',
    'मुझे डॉक्टर के पास जाना है.',
    'मुझे बुखार है.',
    'यह दवाई दिन में दो बार लो.',
    'जल्दी ठीक हो जाओ.',
    'ट्रेन कितने बजे आएगी?',
    'स्टेशन यहाँ से कितनी दूर है?',
    'मुझे अमृतसर जाना है.',
    'सीधे जाओ फिर दाएँ मुड़ जाओ.',
    'यह रास्ता बंद है.',
    'मोबाइल की बैटरी खत्म हो गई.',
    'कंप्यूटर काम नहीं कर रहा है.',
    'इंटरनेट बहुत धीमा है.',
    'यह ऐप बहुत उपयोगी है.',
    'पर्यावरण को बचाना ज़रूरी है.',
    'पेड़ लगाओ, धरती बचाओ.',
    'पानी बचाना हम सबकी ज़िम्मेदारी है.',
    'प्रदूषण दिन-ब-दिन बढ़ रहा है.',
    'किसान खेत में काम कर रहा है.',
    'फसल अच्छी हुई है.',
    'गेहूं की कटाई शुरू हो गई है.',
    'आज बहुत काम है.',
    'मुझे छुट्टी चाहिए.',
    'मीटिंग कल सुबह होगी.',
    'रिपोर्ट समय पर जमा करो.',
    'यह काम बहुत मुश्किल है.',
    'चिंता मत करो, सब ठीक हो जाएगा.',
    'मुझे तुम्हारी मदद चाहिए.',
    'धन्यवाद, आपकी बहुत मेहरबानी.',
    'फिर मिलेंगे.',
    'अपना ख्याल रखना.',
    'मुझे तुमसे प्यार है.',
    'यह बहुत खुशी की बात है.',
    'सत्य की हमेशा जीत होती है.',
]

_punjabi_sentences = [
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
    'ਸਾਨੂੰ ਇੱਕ ਦੂਜੇ ਨਾਲ ਮਿਲ ਕੇ ਕੰਮ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ।',
    'ਸ਼ੁਭ ਸਵੇਰ, ਤੁਹਾਡਾ ਦਿਨ ਚੰਗਾ ਰਹੇ।',
    'ਮੈਨੂੰ ਭੁੱਖ ਲੱਗੀ ਹੈ।',
    'ਮੈਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ।',
    'ਕਿਰਪਾ ਕਰਕੇ ਦਰਵਾਜ਼ਾ ਬੰਦ ਕਰ ਦਿਓ।',
    'ਇਹ ਕਿਤਾਬ ਬਹੁਤ ਚੰਗੀ ਹੈ।',
    'ਮੈਂ ਸਕੂਲ ਜਾ ਰਿਹਾ ਹਾਂ।',
    'ਉਹ ਹਰ ਰੋਜ਼ ਪੜ੍ਹਾਈ ਕਰਦਾ ਹੈ।',
    'ਮੇਰਾ ਨਾਮ ਰਾਮ ਹੈ।',
    'ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?',
    'ਅੱਜ ਮੇਰਾ ਜਨਮਦਿਨ ਹੈ।',
    'ਕੱਲ੍ਹ ਮੀਂਹ ਪਵੇਗਾ।',
    'ਸਰਦੀਆਂ ਵਿੱਚ ਬਰਫ਼ ਪੈਂਦੀ ਹੈ।',
    'ਗਰਮੀਆਂ ਵਿੱਚ ਬਹੁਤ ਗਰਮੀ ਹੁੰਦੀ ਹੈ।',
    'ਮੈਨੂੰ ਪੰਜਾਬੀ ਸਿੱਖਣੀ ਹੈ।',
    'ਉਹ ਹਿੰਦੀ ਬਹੁਤ ਵਧੀਆ ਬੋਲਦਾ ਹੈ।',
    'ਸਾਨੂੰ ਮਿਹਨਤ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ।',
    'ਸਫਲਤਾ ਮਿਹਨਤ ਨਾਲ ਮਿਲਦੀ ਹੈ।',
    'ਸਮਾਂ ਬਹੁਤ ਕੀਮਤੀ ਹੈ।',
    'ਮੇਰੇ ਪਿਤਾ ਜੀ ਇੱਕ ਡਾਕਟਰ ਹਨ।',
    'ਮੇਰੀ ਮਾਂ ਸਕੂਲ ਵਿੱਚ ਪੜ੍ਹਾਉਂਦੀ ਹੈ।',
    'ਮੇਰਾ ਭਰਾ ਕ੍ਰਿਕਟ ਖੇਡਦਾ ਹੈ।',
    'ਮੇਰੀ ਭੈਣ ਗਾਣਾ ਗਾਉਂਦੀ ਹੈ।',
    'ਅਸੀਂ ਬਜ਼ਾਰ ਜਾ ਰਹੇ ਹਾਂ।',
    'ਮੈਨੂੰ ਇਹ ਕੱਪੜਾ ਪਸੰਦ ਹੈ।',
    'ਇਹ ਕਿੰਨੇ ਦਾ ਹੈ?',
    'ਕਿਰਪਾ ਕਰਕੇ ਥੋੜ੍ਹਾ ਸਸਤਾ ਕਰ ਦਿਓ।',
    'ਮੈਨੂੰ ਡਾਕਟਰ ਕੋਲ ਜਾਣਾ ਹੈ।',
    'ਮੈਨੂੰ ਬੁਖ਼ਾਰ ਹੈ।',
    'ਇਹ ਦਵਾਈ ਦਿਨ ਵਿੱਚ ਦੋ ਵਾਰ ਲਓ।',
    'ਜਲਦੀ ਠੀਕ ਹੋ ਜਾਓ।',
    'ਟ੍ਰੇਨ ਕਿੰਨੇ ਵਜੇ ਆਵੇਗੀ?',
    'ਸਟੇਸ਼ਨ ਇੱਥੋਂ ਕਿੰਨੀ ਦੂਰ ਹੈ?',
    'ਮੈਨੂੰ ਅੰਮ੍ਰਿਤਸਰ ਜਾਣਾ ਹੈ।',
    'ਸਿੱਧੇ ਜਾਓ ਫਿਰ ਸੱਜੇ ਮੁੜ ਜਾਓ।',
    'ਇਹ ਰਸਤਾ ਬੰਦ ਹੈ।',
    'ਮੋਬਾਈਲ ਦੀ ਬੈਟਰੀ ਖ਼ਤਮ ਹੋ ਗਈ।',
    'ਕੰਪਿਊਟਰ ਕੰਮ ਨਹੀਂ ਕਰ ਰਿਹਾ ਹੈ।',
    'ਇੰਟਰਨੈੱਟ ਬਹੁਤ ਹੌਲੀ ਹੈ।',
    'ਇਹ ਐਪ ਬਹੁਤ ਲਾਭਦਾਇਕ ਹੈ।',
    'ਵਾਤਾਵਰਣ ਨੂੰ ਬਚਾਉਣਾ ਜ਼ਰੂਰੀ ਹੈ।',
    'ਰੁੱਖ ਲਗਾਓ, ਧਰਤੀ ਬਚਾਓ।',
    'ਪਾਣੀ ਬਚਾਉਣਾ ਸਾਡੀ ਸਭ ਦੀ ਜ਼ਿੰਮੇਵਾਰੀ ਹੈ।',
    'ਪ੍ਰਦੂਸ਼ਣ ਦਿਨੋ-ਦਿਨ ਵਧ ਰਿਹਾ ਹੈ।',
    'ਕਿਸਾਨ ਖੇਤ ਵਿੱਚ ਕੰਮ ਕਰ ਰਿਹਾ ਹੈ।',
    'ਫ਼ਸਲ ਚੰਗੀ ਹੋਈ ਹੈ।',
    'ਕਣਕ ਦੀ ਵਾਢੀ ਸ਼ੁਰੂ ਹੋ ਗਈ ਹੈ।',
    'ਅੱਜ ਬਹੁਤ ਕੰਮ ਹੈ।',
    'ਮੈਨੂੰ ਛੁੱਟੀ ਚਾਹੀਦੀ ਹੈ।',
    'ਮੀਟਿੰਗ ਕੱਲ੍ਹ ਸਵੇਰੇ ਹੋਵੇਗੀ।',
    'ਰਿਪੋਰਟ ਸਮੇਂ ਸਿਰ ਜਮ੍ਹਾਂ ਕਰੋ।',
    'ਇਹ ਕੰਮ ਬਹੁਤ ਔਖਾ ਹੈ।',
    'ਚਿੰਤਾ ਨਾ ਕਰੋ, ਸਭ ਠੀਕ ਹੋ ਜਾਵੇਗਾ।',
    'ਮੈਨੂੰ ਤੁਹਾਡੀ ਮਦਦ ਚਾਹੀਦੀ ਹੈ।',
    'ਧੰਨਵਾਦ, ਤੁਹਾਡੀ ਬਹੁਤ ਮਿਹਰਬਾਨੀ।',
    'ਫਿਰ ਮਿਲਾਂਗੇ।',
    'ਆਪਣਾ ਧਿਆਨ ਰੱਖਣਾ।',
    'ਮੈਨੂੰ ਤੁਹਾਡੇ ਨਾਲ ਪਿਆਰ ਹੈ।',
    'ਇਹ ਬਹੁਤ ਖੁਸ਼ੀ ਦੀ ਗੱਲ ਹੈ।',
    'ਸੱਚ ਦੀ ਹਮੇਸ਼ਾ ਜਿੱਤ ਹੁੰਦੀ ਹੈ।',
]

assert len(_hindi_sentences) == len(_punjabi_sentences), "Parallel corpus lengths must match"

data_corpus = {'Hindi_Sentence': _hindi_sentences, 'Punjabi_Sentence': _punjabi_sentences}

df_corpus = pd.DataFrame(data_corpus)
df_corpus.to_csv('parallel_corpus.csv', index=False, encoding='utf-8')
df_corpus_loaded = pd.read_csv('parallel_corpus.csv', encoding='utf-8')
parallel_corpus = list(df_corpus_loaded.itertuples(index=False, name=None))
print(f"✓ Parallel corpus loaded: {len(parallel_corpus)} sentence pairs")

# Also persist the same parallel corpus as the "custom dataset" consumed by
# train_indicbert.py, so the EBMT examples and the IndicBERT fine-tuning
# data stay in sync for the thesis write-up.
os.makedirs('data', exist_ok=True)
indicbert_dataset_path = os.path.join('data', 'indicbert_training_data.csv')
if not os.path.exists(indicbert_dataset_path):
    pd.DataFrame({'hindi': _hindi_sentences, 'punjabi': _punjabi_sentences}).to_csv(
        indicbert_dataset_path, index=False, encoding='utf-8'
    )

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

        # A flat max_length=512 with plain greedy decoding and no repetition
        # controls lets the model degenerate into looping the same phrase
        # ("X, X, X, X...") once it exhausts real content for a short or
        # ambiguous input - a well-known seq2seq failure mode. Beam search
        # plus repetition controls fixes this, and capping the output
        # length relative to the input (instead of a flat 512) means even
        # a short sentence can't run away into hundreds of tokens.
        input_len = inputs['input_ids'].shape[1]
        max_new_tokens = min(200, max(20, input_len * 4))

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                early_stopping=True
            )

        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text, 0.95

    except Exception as e:
        return f"NMT Error: {str(e)}", -1.0

def load_indictrans2_model():
    """
    Lazily load AI4Bharat's IndicTrans2 (indic-indic) model on first actual
    use, and cache it afterward. It is a larger model than
    NLLB-200-distilled-600M, so loading it eagerly at startup alongside
    NLLB would only add to the OOM risk already flagged for NLLB alone;
    since it is purely an alternate NMT engine a caller opts into, there is
    no reason to pay that memory cost unless someone actually selects it.

    Requires the 'IndicTransToolkit' package (for text pre/post-processing)
    in addition to transformers - returns None with a clear log message if
    it can't be imported, rather than raising, so the app keeps running
    with NLLB (and Dictionary/EBMT) available.
    """
    if 'indictrans2' in model_cache:
        return model_cache['indictrans2']

    print("\n📖 Loading IndicTrans2 (indic-indic) Translation Model...")
    print(f"   Model: {INDICTRANS2_MODEL_NAME}")

    # IndicTransToolkit 1.1.1's collator.py does
    # "from transformers.tokenization_utils import PreTrainedTokenizerBase",
    # a module path that no longer exposes that name on transformers>=5
    # (confirmed: transformers 5.x turned transformers.tokenization_utils
    # into a lazy/virtual module - PreTrainedTokenizerBase now lives at the
    # transformers top level instead). That breaks IndicTransToolkit's
    # import chain with "ImportError: cannot import name
    # 'PreTrainedTokenizerBase' from 'transformers.tokenization_utils'"
    # even though both packages are installed correctly. Patch the name
    # back onto that module before importing IndicTransToolkit, rather
    # than requiring the whole app to pin an older transformers just for
    # this one optional NMT engine.
    try:
        import transformers.tokenization_utils as _tok_utils
        if not hasattr(_tok_utils, 'PreTrainedTokenizerBase'):
            from transformers import PreTrainedTokenizerBase as _PTB
            _tok_utils.PreTrainedTokenizerBase = _PTB
    except Exception as e:
        print(f"⚠️ Could not apply the transformers>=5 compatibility shim for IndicTransToolkit: {e}")

    # IndicTransToolkit/__init__.py eagerly imports its evaluator (which
    # needs indic-nlp-library and sacrebleu) and collator (which needs
    # transformers) submodules, so importing IndicProcessor at all - via
    # either the documented top-level path or the submodule path - runs
    # that same __init__.py and can fail on any of those transitive
    # dependencies, not just on IndicTransToolkit itself being absent.
    # Catch broadly and always print the REAL underlying error/type instead
    # of a blanket "not installed": if the package is actually installed
    # but one of its own dependencies has a version mismatch, "not
    # installed" is simply wrong and sends someone re-installing a package
    # that was never the problem.
    try:
        from IndicTransToolkit import IndicProcessor
    except Exception as e:
        print(f"⚠️ Could not import IndicProcessor from IndicTransToolkit: {type(e).__name__}: {e}")
        print(f"   If IndicTransToolkit is already installed, the error above names the actual "
              f"failing import (commonly indic-nlp-library, sacrebleu, or a transformers version "
              f"mismatch pulled in by IndicTransToolkit's own __init__.py) rather than "
              f"IndicTransToolkit itself being missing.")
        model_cache['indictrans2'] = None
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(INDICTRANS2_MODEL_NAME, trust_remote_code=True)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                INDICTRANS2_MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=True
            )
        except ImportError:
            model = AutoModelForSeq2SeqLM.from_pretrained(INDICTRANS2_MODEL_NAME, trust_remote_code=True)
        model = model.to(device)

        entry = {
            'tokenizer': tokenizer,
            'model': model,
            'device': device,
            'processor': IndicProcessor(inference=True),
        }
        model_cache['indictrans2'] = entry
        print("✅ IndicTrans2 model loaded successfully!")
        return entry

    except Exception as e:
        print(f"⚠️ IndicTrans2 model loading failed: {e}")
        model_cache['indictrans2'] = None
        return None

def indictrans2_translate(hindi_sentence):
    """Neural Machine Translation using AI4Bharat's IndicTrans2 (indic-indic)."""
    entry = load_indictrans2_model()
    if not entry:
        return "IndicTrans2 model not available", -1.0

    try:
        tokenizer = entry['tokenizer']
        model = entry['model']
        device = entry['device']
        ip = entry['processor']

        src_lang, tgt_lang = "hin_Deva", "pan_Guru"
        batch = ip.preprocess_batch([hindi_sentence], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_len = inputs['input_ids'].shape[1]
        max_new_tokens = min(200, max(20, input_len * 4))

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                max_new_tokens=max_new_tokens,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations = ip.postprocess_batch(decoded, lang=tgt_lang)
        return translations[0], 0.95

    except Exception as e:
        return f"IndicTrans2 Error: {str(e)}", -1.0

def run_nmt(hindi_sentence: str, model_choice: str = DEFAULT_NMT_MODEL):
    """
    Dispatch Hindi->Punjabi neural translation to the selected NMT engine.

    model_choice: 'nllb' (default, loaded eagerly at startup - see
        load_models()) or 'indictrans2' (loaded lazily on first use here).
    """
    if model_choice == 'indictrans2':
        return indictrans2_translate(hindi_sentence)
    return nmt_translate(
        hindi_sentence, model_cache.get('tokenizer'), model_cache.get('model'), model_cache.get('device')
    )

def translate_hindi_to_punjabi(hindi_sentence, model_choice: str = DEFAULT_NMT_MODEL):
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
    print(f"⚠️ EBMT failed. Trying NMT ({model_choice})...")
    if model_choice != 'indictrans2' and not model_cache.get('loaded'):
        return "Models not loaded", "None"

    nmt_result, nmt_score = run_nmt(hindi_sentence, model_choice=model_choice)

    print(f"✓ NMT translation: {nmt_result}")
    return nmt_result, "NMT"

def translate_both_modes(hindi_sentence: str, model_choice: str = DEFAULT_NMT_MODEL) -> Dict:
    """
    Run BOTH translation modes (EBMT and NMT) independently and return both
    outputs side-by-side, together with agreement metrics (BLEU and semantic
    similarity between the two outputs). This is used for the thesis
    evaluation of example-based vs. neural machine translation, and for the
    "final output" of the system so a caller can see both modes rather than
    only the single cascade-selected translation.

    model_choice selects the NMT engine used for the 'nmt' mode: 'nllb'
    (default) or 'indictrans2'.
    """
    result = {
        'input': hindi_sentence,
        'dictionary': {'covered_fully': False, 'translation': None, 'untranslated_words': None},
        'ebmt': {'translation': None, 'similarity': None},
        'nmt': {'translation': None, 'confidence': None, 'engine': model_choice},
        'agreement': {},
        'recommended': None,
    }

    # Dictionary coverage (informational; used to pick the recommended output)
    words = hindi_sentence.split()
    translated_words = []
    untranslated = 0
    for word in words:
        tw = translation_dict.get(word)
        if tw is None:
            untranslated += 1
            translated_words.append(word)
        else:
            translated_words.append(tw)

    result['dictionary']['covered_fully'] = (untranslated == 0)
    result['dictionary']['translation'] = ' '.join(translated_words)
    result['dictionary']['untranslated_words'] = untranslated

    # Mode 1: EBMT
    ebmt_text, ebmt_score = ebmt_translate(hindi_sentence, parallel_corpus, jaccard_similarity)
    result['ebmt']['translation'] = ebmt_text
    result['ebmt']['similarity'] = round(float(ebmt_score), 4) if ebmt_score is not None else None

    # Mode 2: NMT (selected engine)
    if model_choice == 'indictrans2' or model_cache.get('loaded'):
        nmt_text, nmt_conf = run_nmt(hindi_sentence, model_choice=model_choice)
    else:
        nmt_text, nmt_conf = "Models not loaded", -1.0
    result['nmt']['translation'] = nmt_text
    result['nmt']['confidence'] = round(float(nmt_conf), 4) if nmt_conf is not None else None

    ebmt_valid = ebmt_text not in ("Translation not found in corpus.", "No similar sentence found.")
    nmt_valid = nmt_text and 'Error' not in nmt_text and nmt_text != "Models not loaded" and nmt_text != "NMT model not available"

    # Agreement between the two modes: BLEU (using NMT output as reference)
    # and cosine similarity of their embeddings, when both outputs exist.
    try:
        if ebmt_valid and nmt_valid:
            bleu = sacrebleu.sentence_bleu(ebmt_text, [nmt_text])
            result['agreement']['bleu_ebmt_vs_nmt'] = round(bleu.score, 2)
        else:
            result['agreement']['bleu_ebmt_vs_nmt'] = None
    except Exception as e:
        result['agreement']['bleu_ebmt_vs_nmt'] = None
        print(f"⚠️ BLEU comparison failed: {e}")

    try:
        sem_model = model_cache.get('semantic_model')
        if sem_model and ebmt_valid and nmt_valid:
            embs = sem_model.encode([ebmt_text, nmt_text], normalize_embeddings=True)
            sim = float(np.dot(embs[0], embs[1]))
            result['agreement']['semantic_similarity_ebmt_vs_nmt'] = round(sim, 4)
        else:
            result['agreement']['semantic_similarity_ebmt_vs_nmt'] = None
    except Exception as e:
        result['agreement']['semantic_similarity_ebmt_vs_nmt'] = None
        print(f"⚠️ Semantic comparison failed: {e}")

    # Recommended translation, using the same cascade priority as
    # translate_hindi_to_punjabi(): Dictionary -> EBMT (confident) -> NMT.
    if result['dictionary']['covered_fully']:
        result['recommended'] = {'translation': result['dictionary']['translation'], 'method': 'Dictionary'}
    elif ebmt_valid and ebmt_score is not None and ebmt_score > 0.3:
        result['recommended'] = {'translation': ebmt_text, 'method': 'EBMT'}
    else:
        result['recommended'] = {'translation': nmt_text, 'method': 'NMT', 'engine': model_choice}

    return result

# ============================================================================
# 4. PLAGIARISM DETECTION FUNCTIONS
# ============================================================================

class EnhancedCorpusManager:
    """Manages corpus for plagiarism detection"""

    def __init__(self):
        self.ws_re = re.compile(r'\s+')
        self.models_available = model_cache.get('loaded', False)
        self.corpus_cache = corpus_cache
        self.sentence_searcher = model_cache.get('sentence_searcher')

    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization for Indic scripts"""
        if not text:
            return ""

        text = unicodedata.normalize('NFC', text)
        text = text.replace('‌', '').replace('‍', '').replace('﻿', '')
        text = self.ws_re.sub(' ', text).strip()

        return text

    def detect_language(self, text: str) -> str:
        """Detect language: Hindi vs Punjabi"""
        if not text:
            return 'unknown'

        hindi_chars = len([c for c in text if 'ऀ' <= c <= 'ॿ'])
        punjabi_chars = len([c for c in text if '਀' <= c <= '੿'])
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

    def search_corpus(self, query: str, top_k: int = 5, threshold: float = PLAGIARISM_SIMILARITY_THRESHOLD) -> List[Dict]:
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

    def search_corpus_bilingual(
        self,
        hindi_text: str,
        translated_punjabi: str,
        top_k: int = 10,
        threshold: float = PLAGIARISM_SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """
        Search the corpus using BOTH:
          - the original Hindi source document
          - its translated Punjabi version

        A corpus document is matched if it is similar to either query, so a
        plagiarized source is caught whether the corpus copy was kept in
        Hindi or already translated to Punjabi. Results are merged and
        deduplicated by filename, keeping the higher-similarity hit and
        tagging which query language produced it.
        """
        merged: Dict[str, Dict] = {}

        if hindi_text and hindi_text.strip():
            print("\n📚 CORPUS SEARCH: ORIGINAL HINDI SOURCE DOCUMENT")
            for m in self.search_corpus(hindi_text, top_k=top_k, threshold=threshold):
                m = dict(m)
                m['query_language'] = 'hindi'
                existing = merged.get(m['filename'])
                if existing is None or m['similarity'] > existing['similarity']:
                    merged[m['filename']] = m

        if translated_punjabi and translated_punjabi.strip():
            print("\n📚 CORPUS SEARCH: TRANSLATED PUNJABI VERSION")
            for m in self.search_corpus(translated_punjabi, top_k=top_k, threshold=threshold):
                m = dict(m)
                m['query_language'] = 'punjabi'
                existing = merged.get(m['filename'])
                if existing is None or m['similarity'] > existing['similarity']:
                    merged[m['filename']] = m

        results = sorted(merged.values(), key=lambda x: x['similarity'], reverse=True)
        print(f"✅ Bilingual corpus search complete: {len(results)} unique document matches")
        return results[:top_k]

# ============================================================================
# 5. ENHANCED INTERNET SEARCH WITH SENTENCE-BASED APPROACH
# ============================================================================

def _semantic_similarity(text_a: str, text_b: str) -> Optional[float]:
    """
    Real cross-lingual cosine similarity between two texts, using the same
    fine-tuned IndicBERT / IndicSBERT model used for corpus matching -
    instead of a fake, rank-based heuristic (e.g. "result #2 gets 0.87").

    Returns None (rather than a made-up number) when the semantic model
    isn't loaded or either text is empty, so callers can fall back to a
    heuristic score explicitly instead of silently trusting a meaningless
    default.
    """
    semantic_model = model_cache.get('semantic_model')
    if not semantic_model or not text_a or not text_b:
        return None
    try:
        embeddings = semantic_model.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))
    except Exception as e:
        print(f"⚠️ Semantic similarity computation failed: {e}")
        return None

def _keyword_overlap_similarity(query: str, text: str) -> float:
    """
    Fraction of the query's (non-stopword) words that actually appear in
    the given text - a genuine keyword-relevance score for non-semantic
    search modes, used instead of Google's rank position.

    A rank-based score treats every result Google returns as progressively
    "similar" regardless of content, so a loosely-related page that merely
    ranked #1 for a vague query still scored ~0.85-0.95. Counting how many
    of the actual searched-for words show up in the result's title/snippet
    is how a natural keyword search judges relevance, and correctly scores
    an off-topic result low even if Google ranked it first.
    """
    query_words = [
        re.sub(r'[.,!?;:\'"।]+$', '', w).strip().lower()
        for w in query.split()
    ]
    query_words = [w for w in query_words if len(w) > 1 and w not in _INDIC_STOPWORDS]
    if not query_words:
        return 0.0
    text_lower = (text or '').lower()
    hits = sum(1 for w in query_words if w in text_lower)
    return hits / len(query_words)

def _search_by_queries(
    search_queries: List[str],
    max_results: int = 30,
    label: str = "queries",
    use_semantic: bool = True
) -> List[Dict]:
    """
    Run _perform_google_search for each query string and merge the results.

    Each query is searched independently, so the same source URL can turn up
    under several different queries. Instead of listing that source once per
    matching query, combine those hits into a single entry: keep the highest
    similarity score seen for that URL, and record how many queries matched
    it.

    use_semantic=False forces plain rank-based scoring (no cross-lingual
    semantic model call) for keyword-only search modes.
    """
    if not search_queries:
        return []

    best_by_url: Dict[str, Dict] = {}
    match_counts: Dict[str, int] = {}

    for i, search_query in enumerate(search_queries, 1):
        print(f"\n📌 Searching with {label} {i}/{len(search_queries)}: '{search_query[:80]}...'")
        matches = _perform_google_search(
            search_query, max_results // len(search_queries) + 2, use_semantic=use_semantic
        )

        for match in matches:
            url = match.get('url')
            if not url:
                continue
            match_counts[url] = match_counts.get(url, 0) + 1
            existing = best_by_url.get(url)
            if existing is None or match['similarity'] > existing['similarity']:
                best_by_url[url] = match

    all_matches = []
    for url, match in best_by_url.items():
        match = dict(match)
        match['matched_sentence_count'] = match_counts[url]
        all_matches.append(match)

    all_matches.sort(key=lambda x: x['similarity'], reverse=True)
    return all_matches[:max_results]

def search_internet_google(query: str, max_results: int = 30, use_sentence_search: bool = True) -> List[Dict]:
    """
    Search Google for similar content using sentence-based approach

    Args:
        query: Input text to search (usually translated Punjabi)
        max_results: Maximum number of results
        use_sentence_search: If True, use sentence-based search instead of keyword-level

    Returns:
        List of search results with similarity scores
    """
    try:
        print(f"\n🌐 Searching Google for similar content...")
        print(f"📝 Search strategy: {'SENTENCE-BASED' if use_sentence_search else 'KEYWORD-BASED'}")

        # Extract sentences if enabled
        search_queries = [query]  # Default: use whole query

        if use_sentence_search and model_cache.get('sentence_searcher'):
            sentence_searcher = model_cache['sentence_searcher']
            search_queries = sentence_searcher.create_sentence_queries(query, num_queries=5)

        all_matches = _search_by_queries(search_queries, max_results=max_results, label="Query")

        print(f"\n✅ Internet search completed: {len(all_matches)} unique sources found "
              f"(from {len(search_queries)} sentence queries)")
        return all_matches

    except Exception as e:
        print(f"❌ Internet search error: {e}")
        traceback.print_exc()
        return []

def search_internet_google_keywords(query: str, max_results: int = 30, use_semantic: bool = True) -> List[Dict]:
    """
    Keyword-based internet search: short phrase queries extracted from the
    text, as a complement to whole-sentence search. A full (often
    machine-translated) sentence frequently fails to match anything
    verbatim on Google, while a short distinctive phrase pulled from it is
    far more likely to hit a real result.

    use_semantic=False scores results by rank alone rather than calling the
    cross-lingual semantic model.
    """
    try:
        print(f"\n🌐 Searching Google with KEYWORD-BASED queries...")

        search_queries = [query]
        if model_cache.get('sentence_searcher'):
            search_queries = model_cache['sentence_searcher'].create_keyword_queries(query, num_queries=5)

        all_matches = _search_by_queries(
            search_queries, max_results=max_results, label="Keyword query", use_semantic=use_semantic
        )

        print(f"\n✅ Keyword-based search completed: {len(all_matches)} unique sources found "
              f"(from {len(search_queries)} keyword queries)")
        return all_matches

    except Exception as e:
        print(f"❌ Keyword-based internet search error: {e}")
        traceback.print_exc()
        return []

def search_internet_bilingual(
    hindi_text: str,
    translated_punjabi: str,
    max_results: int = 30,
    use_sentence_search: bool = True,
    hindi_similarity_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Search the internet using BOTH:
      - original Hindi text
      - translated Punjabi text

    The Hindi leg is searched sentence-by-sentence (when use_sentence_search
    is on), scored semantically, and filtered to hindi_similarity_threshold,
    when given, so only strong matches for the original source surface.

    The Punjabi (translated) leg is searched keyword-based ONLY (not
    sentence-based): a full machine-translated sentence often fails to
    match anything verbatim, while short distinctive phrases from it are
    far more likely to hit a real result. It is scored by rank alone
    (no semantic model call) and filtered to
    PUNJABI_INTERNET_SIMILARITY_THRESHOLD.

    Results are merged and deduplicated by URL, keeping the higher-similarity
    hit (rather than whichever language happened to find it first) so the
    same source is never listed twice.
    """
    best_by_url: Dict[str, Dict] = {}

    # 1) Search with original Hindi text - sentence-based, semantic scoring,
    #    filtered to the similarity threshold.
    if hindi_text and hindi_text.strip():
        print("\n🌐 INTERNET SEARCH: ORIGINAL HINDI TEXT")
        hindi_matches = search_internet_google(
            hindi_text,
            max_results=max_results,
            use_sentence_search=use_sentence_search
        )
        if hindi_similarity_threshold is not None:
            hindi_matches = [m for m in hindi_matches if m.get('similarity', 0.0) >= hindi_similarity_threshold]
        for m in hindi_matches:
            url = m.get("url")
            if not url:
                continue
            m = dict(m)
            m["query_language"] = "hindi"
            existing = best_by_url.get(url)
            if existing is None or m.get("similarity", 0.0) > existing.get("similarity", 0.0):
                best_by_url[url] = m

    # 2) Search with translated Punjabi text - keyword-based only, rank-based
    #    scoring (no semantic model call), filtered to
    #    PUNJABI_INTERNET_SIMILARITY_THRESHOLD.
    if translated_punjabi and translated_punjabi.strip():
        print("\n🌐 INTERNET SEARCH: TRANSLATED PUNJABI TEXT (keyword-based only, "
              f"threshold {PUNJABI_INTERNET_SIMILARITY_THRESHOLD})")
        punjabi_matches = search_internet_google_keywords(
            translated_punjabi,
            max_results=max_results,
            use_semantic=False
        )
        punjabi_matches = [
            m for m in punjabi_matches if m.get('similarity', 0.0) >= PUNJABI_INTERNET_SIMILARITY_THRESHOLD
        ]
        for m in punjabi_matches:
            url = m.get("url")
            if not url:
                continue
            m = dict(m)
            m["query_language"] = "punjabi"
            existing = best_by_url.get(url)
            if existing is None or m.get("similarity", 0.0) > existing.get("similarity", 0.0):
                best_by_url[url] = m

    # Sort combined list by (maximum) similarity and trim
    all_matches = sorted(best_by_url.values(), key=lambda x: x.get("similarity", 0.0), reverse=True)
    print(f"\n✅ Bilingual internet search complete: {len(all_matches)} unique sources")
    return all_matches[:max_results]

def _perform_google_search(query: str, max_results: int = 10, use_semantic: bool = True) -> List[Dict]:
    """
    Perform actual Google search for a single query

    Args:
        query: Search query
        max_results: Maximum results to return
        use_semantic: If False, score results by rank alone instead of
            calling the cross-lingual semantic model (used for keyword-only
            search modes).

    Returns:
        List of search results
    """
    matches = []
    # Set once the Google Custom Search API actually returns a valid (200)
    # response, whether or not it contained any items - "the API works"
    # means the request itself succeeded, not that it happened to find a
    # match. Used below to skip the simulated/fake fallback: simulated data
    # is only appropriate when no real search actually ran.
    google_api_worked = False

    try:
        # API credentials are read from the environment - never hardcode
        # secrets in source control.
        google_api_key = os.getenv('GOOGLE_API_KEY')
        search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
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

                response = requests.get(search_url, params=params, timeout=15)

                print(f"   Response Status: {response.status_code}")

                if response.status_code == 200:
                    google_api_worked = True
                    data = response.json()
                    results = data.get('items', [])

                    print(f"   ✅ SUCCESS: Got {len(results)} results from Google API")

                    for idx, item in enumerate(results):
                        title = item.get('title', 'No title')
                        snippet = item.get('snippet', 'No preview available')
                        combined_text = f"{title} {snippet}"
                        if use_semantic:
                            # Real semantic similarity between the query and
                            # this result's title+snippet, using the
                            # cross-lingual IndicBERT/IndicSBERT model -
                            # falls back to a rank-based estimate only when
                            # the model isn't loaded.
                            sim = _semantic_similarity(query, combined_text)
                            similarity_method = 'semantic' if sim is not None else 'rank_estimate'
                            if sim is None:
                                sim = max(0.8, 0.95 - (idx * 0.08))
                        else:
                            # Keyword-only mode: score by how many of the
                            # actual query words appear in the result,
                            # instead of trusting Google's rank position -
                            # a loosely-related top result should not score
                            # as if it were a strong match.
                            sim = _keyword_overlap_similarity(query, combined_text)
                            similarity_method = 'keyword_overlap'
                        match = {
                            'source': 'internet',
                            'url': item.get('link', ''),
                            'title': title,
                            'similarity': sim,
                            'similarity_method': similarity_method,
                            'snippet': snippet,
                            'search_method': 'google_api'
                        }
                        matches.append(match)

                    if matches:
                        print(f"✅ Returning {len(matches)} Google API results")
                        return matches

                    # A 200 status with zero items is a "successful" call that
                    # found nothing - do NOT stop here. The most common cause
                    # is that the Custom Search Engine (cx) is restricted to
                    # a handful of sites instead of the whole web: a brand
                    # new CSE defaults to that unless "Search the entire web"
                    # is explicitly enabled, so it can return 0 results for a
                    # query that finds real pages on google.com itself. Fall
                    # through to SerpAPI / simulated results below instead of
                    # silently returning an empty list.
                    print(f"   ⚠️ Google API returned 0 results for this query, even though the "
                          f"request succeeded. If a manual Google search finds real matches, check "
                          f"that Custom Search Engine cx={search_engine_id} has 'Search the entire "
                          f"web' enabled at https://programmablesearchengine.google.com/ - a new CSE "
                          f"is restricted to specific sites by default.")

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
                        title = item.get('title', 'No title')
                        snippet = item.get('snippet', 'No preview available')
                        combined_text = f"{title} {snippet}"
                        if use_semantic:
                            sim = _semantic_similarity(query, combined_text)
                            similarity_method = 'semantic' if sim is not None else 'rank_estimate'
                            if sim is None:
                                sim = max(0.5, 0.85 - (idx * 0.08))
                        else:
                            sim = _keyword_overlap_similarity(query, combined_text)
                            similarity_method = 'keyword_overlap'
                        match = {
                            'source': 'internet',
                            'url': item.get('link', ''),
                            'title': title,
                            'similarity': sim,
                            'similarity_method': similarity_method,
                            'snippet': snippet,
                            'search_method': 'serpapi'
                        }
                        matches.append(match)

                    if matches:
                        print(f"✅ Returning {len(matches)} SerpAPI results")
                        return matches

                    print(f"   ⚠️ SerpAPI returned 0 results for this query - falling through to "
                          f"simulated results.")

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

        # ===== METHOD 3: FALLBACK TO SIMULATION =====
        # Only used when NO real search actually ran (no API configured, or
        # every configured API failed outright - bad key, quota, network
        # error). If the Google API responded successfully at all - even
        # with zero items for this particular query - that's a real,
        # truthful "no matches found" and must not be replaced with fake
        # data that could be mistaken for a genuine match.
        if google_api_worked:
            print(f"\n📝 Google API responded successfully but found no results for this query - "
                  f"returning no matches instead of simulated/fake data.")
            return matches

        print(f"\n📝 All real APIs unavailable - using simulated results")
        print(f"   (This is FAKE data for testing)")

        simulated_results = [
            {
                'url': f'https://example-site-1.com/search?q={quote(query)}&hl=en',
                'title': f'Search Results for: {query}',
                'snippet': f'Find information and resources about {query}.',
                'similarity': 0.82,
                'search_method': 'simulated'
            },
            {
                'url': f'https://wiki-example.com/{quote(query)}',
                'title': f'{query} - Reference',
                'snippet': f'Comprehensive information about {query}.',
                'similarity': 0.76,
                'search_method': 'simulated'
            },
            {
                'url': f'https://www.quora.com/search?q={quote(query)}',
                'title': f'Q&A: {query}',
                'snippet': f'Questions and answers related to {query}.',
                'similarity': 0.68,
                'search_method': 'simulated'
            },
            {
                'url': f'https://blog-example.com/{quote(query)}',
                'title': f'Blog: {query}',
                'snippet': f'In-depth analysis about {query}.',
                'similarity': 0.61,
                'search_method': 'simulated'
            },
            {
                'url': f'https://news-example.com/topic/{quote(query)}',
                'title': f'News: {query}',
                'snippet': f'Latest news about {query}.',
                'similarity': 0.55,
                'search_method': 'simulated'
            }
        ]

        for result in simulated_results[:max_results]:
            match = {
                'source': 'internet',
                'url': result['url'],
                'title': result['title'],
                'similarity': result['similarity'],
                'snippet': result['snippet'],
                'search_method': result['search_method']
            }
            matches.append(match)

        print(f"✅ Generated {len(matches)} SIMULATED results (not real)")
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

    Accepts: File upload (+ optional use_sentence_search form field)
    Process: Extract text → Translate (both modes) → Corpus check → Internet search
    Runs the exact same pipeline as /api/plagiarism-check/text, so uploaded
    documents get full corpus and internet matching, not just a preview.
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

        use_sentence_search = request.form.get('use_sentence_search', 'true').lower() != 'false'
        nmt_model = _resolve_nmt_model(request.form.get('nmt_model'))

        # Save uploaded file
        filename = safe_upload_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"\n📄 Processing uploaded document: {filename}")

        # Extract text from file
        content = extract_text_from_file(filepath)

        # Clean up the uploaded file immediately - we only need the extracted text
        os.remove(filepath)

        if not content or not content.strip():
            return jsonify({'success': False, 'error': 'Could not extract text from file'}), 400

        print(f"✅ Document extracted: {len(content)} characters")

        response_data = run_plagiarism_pipeline(content, use_sentence_search, nmt_model=nmt_model)
        response_data['filename'] = filename
        response_data['character_count'] = len(content)
        response_data['word_count'] = len(content.split())

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
        tags = request.form.getlist('tags')

        # Save uploaded file temporarily
        filename = safe_upload_filename(file.filename)
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

def run_plagiarism_pipeline(
    hindi_text: str,
    use_sentence_search: bool = True,
    nmt_model: str = DEFAULT_NMT_MODEL
) -> Dict:
    """
    Core plagiarism-check pipeline shared by the text-input and
    document-upload endpoints: translate (both EBMT and NMT modes) → corpus
    matching → sentence-based internet search → summary + DB logging.

    nmt_model selects the NMT engine ('nllb' or 'indictrans2') used for the
    NMT translation mode.
    """
    print("\n" + "=" * 80)
    print("PLAGIARISM CHECK STARTED")
    print("=" * 80)

    start_time = datetime.now()

    # =========== STEP 1: TRANSLATE (BOTH MODES) ===========
    print(f"\n[STEP 1] HINDI TO PUNJABI TRANSLATION (DICTIONARY + EBMT + NMT/{nmt_model})")
    print("-" * 80)

    both_modes = translate_both_modes(hindi_text, model_choice=nmt_model)
    translated_punjabi = both_modes['recommended']['translation']
    translation_method = both_modes['recommended']['method']

    # =========== STEP 2: CORPUS PLAGIARISM CHECK (BILINGUAL) ===========
    # Compares the original Hindi source document AND its translated
    # Punjabi version against the corpus, so a match is caught whether the
    # corpus copy of the source is still in Hindi or already in Punjabi.
    print("\n[STEP 2] CROSS-LANGUAGE PLAGIARISM CHECK - CORPUS (HINDI + PUNJABI)")
    print("-" * 80)

    corpus_matches = corpus_manager.search_corpus_bilingual(
        hindi_text, translated_punjabi, top_k=10, threshold=PLAGIARISM_SIMILARITY_THRESHOLD
    )
    corpus_matches_hindi = [m for m in corpus_matches if m.get('query_language') == 'hindi']
    corpus_matches_punjabi = [m for m in corpus_matches if m.get('query_language') == 'punjabi']

    # =========== STEP 3: INTERNET SEARCH (GOOGLE) - ORIGINAL HINDI + TRANSLATED PUNJABI ===========
    # Searches the internet with BOTH the original Hindi source document and
    # its Punjabi translation, each carrying its own similarity %, instead
    # of only searching the translated text. The Hindi leg is sentence-based
    # with semantic scoring, filtered to PLAGIARISM_SIMILARITY_THRESHOLD; the
    # Punjabi (translated) leg is keyword-based only, with rank-based (non-
    # semantic) scoring, filtered to PUNJABI_INTERNET_SIMILARITY_THRESHOLD.
    print("\n[STEP 3] INTERNET SEARCH (GOOGLE) - HINDI SOURCE (sentence-based, semantic, thresholded) "
          "+ PUNJABI TRANSLATION (keyword-based, rank-based, thresholded)")
    print("-" * 80)

    internet_matches = search_internet_bilingual(
        hindi_text, translated_punjabi, max_results=30, use_sentence_search=use_sentence_search,
        hindi_similarity_threshold=PLAGIARISM_SIMILARITY_THRESHOLD
    )
    internet_matches_hindi = [m for m in internet_matches if m.get('query_language') == 'hindi']
    internet_matches_punjabi = [m for m in internet_matches if m.get('query_language') == 'punjabi']

    #=========== PREPARE RESPONSE ===========
    processing_time = (datetime.now() - start_time).total_seconds()
    response_data = {
        'success': True,
        'original_hindi': hindi_text,
        'translated_punjabi': translated_punjabi,
        'translation_method': translation_method,

        # Both translation modes (EBMT + NMT), shown side-by-side so the
        # final output is not limited to a single cascade winner.
        'translation_modes': both_modes,

        # Corpus Results - searched with both the original Hindi source
        # document and its translated Punjabi version.
        'corpus_results': {
            'total_matches': len(corpus_matches),
            'matches': corpus_matches[:5],
            'max_similarity': max([m['similarity'] for m in corpus_matches], default=0),
            'hindi_matches': len(corpus_matches_hindi),
            'punjabi_matches': len(corpus_matches_punjabi),
            'max_similarity_hindi': max([m['similarity'] for m in corpus_matches_hindi], default=0),
            'max_similarity_punjabi': max([m['similarity'] for m in corpus_matches_punjabi], default=0),
            'compared_against': ['original_hindi_source', 'translated_punjabi']
        },

        # Internet Results - searched with both the original Hindi source
        # document and its translated Punjabi version, each with its own
        # similarity %.
        'internet_results': {
            'total_matches': len(internet_matches),
            'matches': internet_matches[:40],
            'max_similarity': max([m['similarity'] for m in internet_matches], default=0),
            'hindi_matches': len(internet_matches_hindi),
            'punjabi_matches': len(internet_matches_punjabi),
            'max_similarity_hindi': max([m['similarity'] for m in internet_matches_hindi], default=0),
            'max_similarity_punjabi': max([m['similarity'] for m in internet_matches_punjabi], default=0),
            'search_method': 'sentence-based' if use_sentence_search else 'keyword-based'
        },

        # Summary
        'plagiarism_summary': {
            'total_matches': len(corpus_matches) + len(internet_matches),
            'corpus_matches': len(corpus_matches),
            'internet_matches': len(internet_matches),
            'highest_corpus_similarity': max([m['similarity'] for m in corpus_matches], default=0),
            'highest_internet_similarity': max([m['similarity'] for m in internet_matches], default=0),
            'highest_internet_similarity_hindi': max([m['similarity'] for m in internet_matches_hindi], default=0),
            'highest_internet_similarity_punjabi': max([m['similarity'] for m in internet_matches_punjabi], default=0),
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
            'sentence-based' if use_sentence_search else 'keyword-based'))

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"⚠️ Database save error: {e}")

    return response_data

@app.route('/api/plagiarism-check/text', methods=['POST'])
def plagiarism_check_text():
    """
    Check plagiarism for text input

    Accepts: Hindi text
    Process: Translate (BOTH EBMT and NMT modes) → Corpus check → Internet search (SENTENCE-BASED)
    """
    try:
        data = request.get_json()
        hindi_text = data.get('hindi_text', '').strip()
        use_sentence_search = data.get('use_sentence_search', True)
        nmt_model = _resolve_nmt_model(data.get('nmt_model'))

        if not hindi_text:
            return jsonify({'error': 'No Hindi text provided'}), 400

        return jsonify(run_plagiarism_pipeline(hindi_text, use_sentence_search, nmt_model=nmt_model))

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/translate/compare', methods=['POST'])
def translate_compare():
    """
    Translate Hindi text and return BOTH translation modes (EBMT and NMT)
    side-by-side, along with their agreement metrics (BLEU, semantic
    similarity). Dedicated endpoint for the thesis translation-quality
    evaluation chapter.
    """
    try:
        data = request.get_json()
        hindi_text = data.get('hindi_text', '').strip()
        nmt_model = _resolve_nmt_model(data.get('nmt_model'))

        if not hindi_text:
            return jsonify({'error': 'No Hindi text provided'}), 400

        result = translate_both_modes(hindi_text, model_choice=nmt_model)
        return jsonify({'success': True, **result})

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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

@app.route('/api/history/list', methods=['GET'])
def list_history():
    """Get recent plagiarism-check history (text-input and document-upload checks)"""
    try:
        conn = sqlite3.connect('corpus_database.db')
        cursor = conn.cursor()

        cursor.execute('''SELECT id, query_content, query_language, corpus_matches,
            max_corpus_similarity, internet_matches, max_internet_similarity,
            processing_time, check_timestamp, search_method
            FROM plagiarism_checks
            ORDER BY check_timestamp DESC
            LIMIT 100''')

        results = cursor.fetchall()
        conn.close()

        history = []
        for row in results:
            history.append({
                'id': row[0],
                'query_preview': (row[1] or '')[:150],
                'query_language': row[2],
                'corpus_matches': row[3],
                'max_corpus_similarity': row[4],
                'internet_matches': row[5],
                'max_internet_similarity': row[6],
                'processing_time': row[7],
                'check_timestamp': row[8],
                'search_method': row[9]
            })

        print(f"✅ Retrieved {len(history)} plagiarism-check history entries")
        return jsonify({'history': history})

    except Exception as e:
        print(f"❌ Error listing history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:check_id>', methods=['GET'])
def get_history_detail(check_id):
    """Get the full stored result for one past plagiarism check, for re-viewing."""
    try:
        conn = sqlite3.connect('corpus_database.db')
        cursor = conn.cursor()

        cursor.execute('SELECT results_json FROM plagiarism_checks WHERE id = ?', (check_id,))
        row = cursor.fetchone()
        conn.close()

        if not row or not row[0]:
            return jsonify({'error': 'History entry not found'}), 404

        return jsonify(json.loads(row[0]))

    except Exception as e:
        print(f"❌ Error fetching history detail: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/delete', methods=['POST'])
def delete_history_entry():
    """Delete a plagiarism-check history entry"""
    try:
        data = request.get_json()
        check_id = data.get('id')

        if not check_id:
            return jsonify({'success': False, 'error': 'No id provided'}), 400

        conn = sqlite3.connect('corpus_database.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM plagiarism_checks WHERE id = ?', (check_id,))

        if cursor.rowcount > 0:
            conn.commit()
            conn.close()
            print(f"✅ History entry deleted: {check_id}")
            return jsonify({'success': True, 'message': f'History entry {check_id} deleted'})
        else:
            conn.close()
            print(f"❌ History entry not found: {check_id}")
            return jsonify({'success': False, 'message': 'History entry not found'}), 404

    except Exception as e:
        print(f"❌ Error deleting history entry: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_cache.get('loaded', False),
        'semantic_model_source': model_cache.get('semantic_model_source'),
        'dictionary_size': len(translation_dict),
        'ebmt_corpus_size': len(parallel_corpus),
        'corpus_size': corpus_cache['corpus_size'],
        'nmt_models': {
            'nllb': model_cache.get('model') is not None,
            # IndicTrans2 loads lazily on first use, so 'indictrans2' not
            # yet being a key in model_cache means "not attempted yet"
            # (still selectable) rather than "unavailable".
            'indictrans2': model_cache.get('indictrans2') is not None if 'indictrans2' in model_cache else None,
        },
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
    print(" POST /api/plagiarism-check/text - Check plagiarism for text input (both translation modes)")
    print(" POST /api/translate/compare - Compare EBMT vs NMT translation modes")
    print(" POST /api/corpus/add - Add text to corpus")
    print(" GET /api/corpus/list - List all corpus documents")
    print(" POST /api/corpus/delete - Delete document from corpus")
    print(" GET /api/corpus/stats - Get corpus statistics")
    print(" GET /health - Health check")

    print("\n🔍 FEATURES:")
    print(" ✅ Document file upload for plagiarism checking (TXT, PDF, DOCX)")
    print(" ✅ Corpus management with document upload")
    print(" ✅ SENTENCE-BASED Google search (after translation to Punjabi)")
    print(" ✅ Hindi-Punjabi translation - BOTH EBMT and NMT modes in final output")
    print(" ✅ Cross-language plagiarism detection using fine-tuned IndicBERT (or IndicSBERT fallback)")
    print(" ✅ Expanded dictionary (160+ pairs) and EBMT parallel corpus (70 pairs)")

    print("\n⚙️ CONFIGURATION:")
    print(f" Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f" Corpus folder: {app.config['CORPUS_FOLDER']}")
    print(f" Max file size: {app.config['MAX_CONTENT_LENGTH'] / 1024 / 1024}MB")
    print(f" Semantic model source: {model_cache.get('semantic_model_source')}")

    print("\nPress CTRL+C to stop the server\n")

    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
