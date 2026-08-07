# Build a large, real Hindi-Punjabi parallel corpus for app.py's EBMT module,
# plus a much larger Hindi-Punjabi dictionary derived from that same corpus
# via statistical word alignment - both from the published AI4Bharat
# Samanantar dataset, the same source evaluate_samanantar.py already uses to
# benchmark this project's cross-language matcher.
#
# Samanantar ships English-Indic parallel pairs, not direct Hindi-Punjabi
# pairs (unless a direct hi-pa config has since been published), so this
# script derives Hindi-Punjabi pairs by pivoting through the English
# sentences shared between the 'hi' and 'pa' configs - identical approach to
# evaluate_samanantar.py's load_samanantar_pairs().
#
# WHY THIS IS A SEPARATE SCRIPT YOU RUN YOURSELF:
# This was written and tested in a sandboxed environment whose network policy
# blocks huggingface.co entirely (no exceptions), so the Hugging Face
# download itself could not be exercised here. Run this script on a machine
# with normal internet access; app.py and train_indicbert.py will
# automatically pick up its output files if present (see
# EXTENDED_CORPUS_PATH / EXTENDED_DICTIONARY_PATH in app.py) - nothing else
# needs to change.
#
# What it produces (all under data/, none of them touch the small hand-
# curated built-in lists inside app.py - they only extend them at runtime):
#   data/parallel_corpus_extended.csv          Hindi_Sentence,Punjabi_Sentence
#   data/indicbert_training_data_extended.csv  hindi,punjabi (for train_indicbert.py --data)
#   data/hindi_punjabi_dictionary_extended.csv Hindi,Punjabi,confidence
#
# Usage:
#   pip install datasets nltk
#   python build_parallel_corpus.py --target_pairs 10000
#
# If fewer than --target_pairs pairs survive cleaning after scanning
# --pivot_scan_limit rows of each config, re-run with a larger
# --pivot_scan_limit (Samanantar's hi/pa configs each have several million
# rows, but the *overlap* of shared English pivot sentences between them is
# what actually bounds how many Hindi-Punjabi pairs can be derived).

import argparse
import os
import random
import re
import unicodedata
from collections import defaultdict

import pandas as pd


# ============================================================================
# Text cleaning / script detection
# (mirrors app.py's EnhancedCorpusManager.normalize_text / detect_language,
# kept standalone here so this script has no dependency on app.py itself -
# importing app.py would trigger loading the full translation/semantic model
# stack, which this script has no use for.)
# ============================================================================

_WS_RE = re.compile(r'\s+')
_ZERO_WIDTH = ('‌', '‍', '﻿')

DEVANAGARI_RANGE = ('ऀ', 'ॿ')
GURMUKHI_RANGE = ('਀', '੿')

# Word/token splitter: splits on whitespace and common punctuation, including
# the Devanagari/Gurmukhi danda and double-danda sentence terminators.
_TOKEN_RE = re.compile(r"[^\s.,!?;:\"'()\[\]{}।॥]+")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    for ch in _ZERO_WIDTH:
        text = text.replace(ch, '')
    return _WS_RE.sub(' ', text).strip()


def script_ratio(text: str, lo: str, hi: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    in_range = sum(1 for c in chars if lo <= c <= hi)
    return in_range / len(chars)


def tokenize(text: str):
    return _TOKEN_RE.findall(text)


def clean_pair(hindi: str, punjabi: str, min_words: int, max_words: int):
    """Return (clean_hindi, clean_punjabi) or None if the pair should be dropped."""
    hindi = normalize_text(hindi)
    punjabi = normalize_text(punjabi)
    if not hindi or not punjabi:
        return None

    hindi_words = hindi.split()
    punjabi_words = punjabi.split()
    if not (min_words <= len(hindi_words) <= max_words):
        return None
    if not punjabi_words:
        return None

    # Length-ratio sanity filter (standard parallel-corpus cleaning heuristic,
    # e.g. Moses' clean-corpus-n.perl) - catches misaligned/garbage rows
    # where one side is wildly longer/shorter than the other.
    ratio = len(punjabi_words) / len(hindi_words)
    if not (0.4 <= ratio <= 2.5):
        return None

    # Script sanity: Hindi side should be predominantly Devanagari, Punjabi
    # side predominantly Gurmukhi - drops rows where the pivot mapping picked
    # up mismatched/mislabeled languages.
    if script_ratio(hindi, *DEVANAGARI_RANGE) < 0.5:
        return None
    if script_ratio(punjabi, *GURMUKHI_RANGE) < 0.5:
        return None

    return hindi, punjabi


# ============================================================================
# Samanantar loading (adapted from evaluate_samanantar.py)
# ============================================================================

def _pick_col(row, candidates):
    for c in candidates:
        if c in row:
            return c
    return None


def _load_dataset_streaming(config_name):
    from datasets import load_dataset
    try:
        return load_dataset("ai4bharat/samanantar", config_name, split="train", streaming=True)
    except Exception as e:
        if 'trust_remote_code' in str(e):
            return load_dataset("ai4bharat/samanantar", config_name, split="train", streaming=True,
                                 trust_remote_code=True)
        raise


def _stream_en_to_indic_map(config_name, scan_limit, src_col=None, tgt_col=None):
    print(f"  Streaming Samanantar config '{config_name}' (scanning up to {scan_limit:,} rows)...")
    ds = _load_dataset_streaming(config_name)
    mapping = {}
    detected_src, detected_tgt = src_col, tgt_col
    for i, row in enumerate(ds):
        if detected_src is None:
            detected_src = _pick_col(row, ['src', 'source', 'en', 'english'])
            detected_tgt = _pick_col(row, ['tgt', 'target', 'translation', config_name])
            if detected_src is None or detected_tgt is None:
                raise RuntimeError(
                    f"Could not auto-detect source/target columns for Samanantar config '{config_name}'. "
                    f"Row keys were: {list(row.keys())}. Re-run with --src_col/--tgt_col to override."
                )
            print(f"    Using columns: src='{detected_src}', tgt='{detected_tgt}'")
        en = (row.get(detected_src) or '').strip()
        txt = (row.get(detected_tgt) or '').strip()
        if en and txt and en not in mapping:
            mapping[en] = txt
        if (i + 1) % 200_000 == 0:
            print(f"    ...scanned {i + 1:,} rows, {len(mapping):,} unique English sentences so far")
        if i + 1 >= scan_limit:
            break
    print(f"  Collected {len(mapping):,} unique English-keyed sentences from '{config_name}'")
    return mapping


def load_raw_samanantar_pairs(pivot_scan_limit, seed, src_col=None, tgt_col=None):
    """Return ALL available (hindi, punjabi) sentence pairs from Samanantar (uncleaned)."""
    for direct_config in ("hi-pa", "hi_pa", "pa-hi"):
        try:
            print(f"Trying direct Samanantar config '{direct_config}' ...")
            ds = _load_dataset_streaming(direct_config)
            pairs = []
            for i, row in enumerate(ds):
                h = row.get('hi') or row.get('src')
                p = row.get('pa') or row.get('tgt')
                if h and p:
                    pairs.append((h.strip(), p.strip()))
                if (i + 1) % 200_000 == 0:
                    print(f"  ...scanned {i + 1:,} rows, {len(pairs):,} pairs so far")
                if i + 1 >= pivot_scan_limit:
                    break
            if pairs:
                print(f"[OK] Loaded {len(pairs)} direct Hindi-Punjabi pairs from config '{direct_config}'")
                random.Random(seed).shuffle(pairs)
                return pairs
        except Exception:
            continue

    print("No direct hi-pa config found - deriving pairs via English pivot (hi & pa configs)...")
    hi_map = _stream_en_to_indic_map("hi", pivot_scan_limit, src_col, tgt_col)
    pa_map = _stream_en_to_indic_map("pa", pivot_scan_limit, src_col, tgt_col)

    shared_en = list(set(hi_map.keys()) & set(pa_map.keys()))
    print(f"  {len(shared_en):,} English sentences appear in both configs")
    if not shared_en:
        raise RuntimeError(
            "No overlapping English pivot sentences found between the 'hi' and 'pa' configs. "
            "Try increasing --pivot_scan_limit, or pass --src_col/--tgt_col if column "
            "auto-detection picked the wrong fields."
        )

    rng = random.Random(seed)
    rng.shuffle(shared_en)
    pairs = [(hi_map[en], pa_map[en]) for en in shared_en]
    print(f"[OK] Derived {len(pairs)} Hindi-Punjabi pairs via English pivot")
    return pairs


def collect_clean_pairs(target_pairs, pivot_scan_limit, seed, min_words, max_words, src_col, tgt_col):
    raw_pairs = load_raw_samanantar_pairs(pivot_scan_limit, seed, src_col, tgt_col)

    cleaned = []
    seen_hindi = set()
    for hindi, punjabi in raw_pairs:
        result = clean_pair(hindi, punjabi, min_words, max_words)
        if result is None:
            continue
        clean_hindi, clean_punjabi = result
        if clean_hindi in seen_hindi:
            continue
        seen_hindi.add(clean_hindi)
        cleaned.append((clean_hindi, clean_punjabi))
        if len(cleaned) >= target_pairs:
            break

    print(f"\nCleaned {len(cleaned):,} / requested {target_pairs:,} pairs "
          f"(from {len(raw_pairs):,} raw candidate pairs)")
    if len(cleaned) < target_pairs:
        print(f"  [!] Did not reach --target_pairs. Re-run with a larger --pivot_scan_limit "
              f"to scan more of Samanantar (current: {pivot_scan_limit:,}).")

    return cleaned


# ============================================================================
# Dictionary extraction via IBM Model 1 word alignment
# ============================================================================

def build_dictionary(pairs, align_sample_size, iterations, min_prob, max_pairs, seed):
    """
    Derive a Hindi -> Punjabi word-level dictionary from a sample of the
    cleaned sentence pairs, using IBM Model 1 statistical word alignment
    (the classic MT lexical-translation-probability model, EM-trained here
    via NLTK). This is the standard way to mine a bilingual lexicon out of a
    sentence-aligned parallel corpus without a pre-existing dictionary.

    For each Hindi word, the Punjabi word with the highest learned
    translation probability P(punjabi_word | hindi_word) is kept, provided
    it clears --dict_min_prob. Entries are inherently approximate (EM often
    latches onto frequent collocates rather than true translations for rare
    words) - app.py always prefers its hand-verified built-in dictionary on
    conflicts, this extended one only fills gaps.
    """
    try:
        from nltk.translate.api import AlignedSent
        from nltk.translate.ibm1 import IBMModel1
    except ImportError:
        raise SystemExit("Missing dependency. Run: pip install nltk")

    rng = random.Random(seed)
    sample = pairs[:]
    rng.shuffle(sample)
    sample = sample[:align_sample_size]

    print(f"\nTraining IBM Model 1 word alignment on {len(sample):,} sentence pairs "
          f"({iterations} EM iterations)...")

    bitext = []
    for hindi, punjabi in sample:
        hi_tokens = tokenize(hindi)
        pa_tokens = tokenize(punjabi)
        if hi_tokens and pa_tokens:
            # words=target=Punjabi, mots=source=Hindi, so translation_table
            # ends up indexed as translation_table[punjabi_word][hindi_word]
            # = P(punjabi_word | hindi_word) - the direction we want.
            bitext.append(AlignedSent(pa_tokens, hi_tokens))

    model = IBMModel1(bitext, iterations)

    # translation_table is indexed [target=punjabi_word][source=hindi_word].
    # Invert to hindi_word -> {punjabi_word: prob} by walking only the
    # (target, source) pairs the EM training actually populated, which is
    # far sparser than the full vocab cross-product.
    hindi_to_candidates = defaultdict(dict)
    for punjabi_word, hindi_probs in model.translation_table.items():
        if punjabi_word is None:
            continue
        for hindi_word, prob in hindi_probs.items():
            if hindi_word is None:  # NULL-alignment token, not a real word
                continue
            hindi_to_candidates[hindi_word][punjabi_word] = prob

    entries = []
    for hindi_word, candidates in hindi_to_candidates.items():
        if script_ratio(hindi_word, *DEVANAGARI_RANGE) < 0.5:
            continue
        best_punjabi_word, best_prob = max(candidates.items(), key=lambda kv: kv[1])
        if best_prob < min_prob:
            continue
        if script_ratio(best_punjabi_word, *GURMUKHI_RANGE) < 0.5:
            continue
        entries.append((hindi_word, best_punjabi_word, round(float(best_prob), 4)))

    entries.sort(key=lambda e: e[2], reverse=True)
    entries = entries[:max_pairs]

    print(f"Derived {len(entries):,} Hindi-Punjabi word pairs above min_prob={min_prob}")
    return entries


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build a large real Hindi-Punjabi parallel corpus (EBMT) and dictionary "
                    "from AI4Bharat's Samanantar dataset."
    )
    parser.add_argument("--target_pairs", type=int, default=10000,
                         help="Desired number of cleaned Hindi-Punjabi sentence pairs")
    parser.add_argument("--pivot_scan_limit", type=int, default=3_000_000,
                         help="Rows to stream from each of the hi/pa Samanantar configs "
                              "(only used if no direct hi-pa config is found)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_words", type=int, default=3, help="Minimum Hindi-side word count to keep a pair")
    parser.add_argument("--max_words", type=int, default=40, help="Maximum Hindi-side word count to keep a pair")
    parser.add_argument("--src_col", default=None, help="Override auto-detected English column name")
    parser.add_argument("--tgt_col", default=None, help="Override auto-detected Indic-language column name")
    parser.add_argument("--output_corpus", default=os.path.join("data", "parallel_corpus_extended.csv"))
    parser.add_argument("--output_indicbert_data",
                         default=os.path.join("data", "indicbert_training_data_extended.csv"))
    parser.add_argument("--skip_dictionary", action="store_true",
                         help="Skip the word-alignment dictionary-extraction step")
    parser.add_argument("--output_dictionary", default=os.path.join("data", "hindi_punjabi_dictionary_extended.csv"))
    parser.add_argument("--dict_align_sample_size", type=int, default=8000,
                         help="Number of cleaned sentence pairs used to train the word-alignment model "
                              "(bounds runtime - IBM Model 1 in NLTK is pure Python)")
    parser.add_argument("--dict_iterations", type=int, default=5, help="IBM Model 1 EM iterations")
    parser.add_argument("--dict_min_prob", type=float, default=0.4,
                         help="Minimum learned translation probability to accept a word pair")
    parser.add_argument("--dict_max_pairs", type=int, default=20000)
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit("Missing dependency. Run: pip install datasets")

    print("=" * 80)
    print("Building Hindi-Punjabi parallel corpus + dictionary from Samanantar")
    print("=" * 80)

    print(f"\n[1/3] Collecting up to {args.target_pairs:,} cleaned Hindi-Punjabi sentence pairs...")
    pairs = collect_clean_pairs(
        args.target_pairs, args.pivot_scan_limit, args.seed,
        args.min_words, args.max_words, args.src_col, args.tgt_col
    )
    if not pairs:
        raise SystemExit("No usable pairs collected - see warnings above.")

    print(f"\n[2/3] Writing corpus files...")
    os.makedirs(os.path.dirname(args.output_corpus) or '.', exist_ok=True)
    corpus_df = pd.DataFrame(pairs, columns=['Hindi_Sentence', 'Punjabi_Sentence'])
    corpus_df.to_csv(args.output_corpus, index=False, encoding='utf-8')
    print(f"  Wrote {len(corpus_df):,} pairs to {args.output_corpus}  (used by app.py's EBMT module)")

    indicbert_df = pd.DataFrame(pairs, columns=['hindi', 'punjabi'])
    indicbert_df.to_csv(args.output_indicbert_data, index=False, encoding='utf-8')
    print(f"  Wrote {len(indicbert_df):,} pairs to {args.output_indicbert_data}  "
          f"(pass to train_indicbert.py via --data)")

    if not args.skip_dictionary:
        print(f"\n[3/3] Deriving dictionary via word alignment...")
        dict_entries = build_dictionary(
            pairs, args.dict_align_sample_size, args.dict_iterations,
            args.dict_min_prob, args.dict_max_pairs, args.seed
        )
        os.makedirs(os.path.dirname(args.output_dictionary) or '.', exist_ok=True)
        dict_df = pd.DataFrame(dict_entries, columns=['Hindi', 'Punjabi', 'confidence'])
        dict_df.to_csv(args.output_dictionary, index=False, encoding='utf-8')
        print(f"  Wrote {len(dict_df):,} word pairs to {args.output_dictionary}  "
              f"(used by app.py's dictionary-lookup translation stage)")
    else:
        print("\n[3/3] Skipped dictionary extraction (--skip_dictionary)")

    print("\nDone. Restart app.py (or the Flask process) to pick up the new files -")
    print("it merges them into the built-in dictionary/corpus automatically at startup.")


if __name__ == "__main__":
    main()
