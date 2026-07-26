# Evaluate the cross-language Hindi-Punjabi matcher (the core of app.py's
# plagiarism-detection module) against the Samanantar corpus (AI4Bharat),
# reporting Precision / Recall / F1 as a binary "is this the correct
# translation match" classification task.
#
# Samanantar ships English-Indic parallel pairs, not direct Hindi-Punjabi
# pairs, so this script derives Hindi-Punjabi pairs by pivoting through the
# English sentences shared between the 'hi' (English-Hindi) and 'pa'
# (English-Punjabi) configs, unless a direct hi-pa config happens to exist.
#
# Methodology
# -----------
# For each of N sampled Hindi-Punjabi reference pairs (h_i, p_i):
#   1. Translate h_i through app.translate_both_modes() - the exact same
#      call app.py's /api/plagiarism-check/text makes - giving three
#      candidate translations: the cascade-recommended one, the EBMT-only
#      one, and the NMT-only one.
#   2. Score each candidate translation's similarity against ALL N
#      reference Punjabi sentences {p_1..p_N} (the same corpus-matching
#      approach as EnhancedCorpusManager.search_corpus in app.py), calling
#      anything >= threshold a "match". This runs twice when a semantic
#      model is loaded: once with the app's semantic embeddings
#      (fine-tuned IndicBERT if trained, else pretrained IndicSBERT), and
#      once with plain Jaccard word-overlap (the same function EBMT uses).
#   3. p_i is the single ground-truth positive for query i; every other
#      p_j is a ground-truth negative.
# TP / FP / FN are accumulated across all N queries to get micro-averaged
# Precision, Recall and F1 for each (translation variant, matcher) pair.
#
# Usage:
#   pip install datasets
#   python evaluate_samanantar.py --sample_size 100
#
# Note: this script was written and syntax-checked without live access to
# huggingface.co (network policy in the authoring sandbox blocks it), so
# the Samanantar column-name auto-detection is defensive but unverified
# against the live dataset. If it can't find the right columns it will
# print what it found and tell you to pass --src_col/--tgt_col explicitly.

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np


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
        if i + 1 >= scan_limit:
            break
    print(f"  Collected {len(mapping):,} unique English-keyed sentences from '{config_name}'")
    return mapping


def load_samanantar_pairs(sample_size, pivot_scan_limit, seed=42, src_col=None, tgt_col=None):
    """Return up to sample_size (hindi, punjabi) sentence pairs from Samanantar."""
    # First, try a direct Hindi-Punjabi config in case one is published.
    for direct_config in ("hi-pa", "hi_pa", "pa-hi"):
        try:
            print(f"Trying direct Samanantar config '{direct_config}' ...")
            ds = _load_dataset_streaming(direct_config)
            pairs = []
            for row in ds:
                h = row.get('hi') or row.get('src')
                p = row.get('pa') or row.get('tgt')
                if h and p:
                    pairs.append((h.strip(), p.strip()))
                if len(pairs) >= sample_size:
                    break
            if pairs:
                print(f"[OK] Loaded {len(pairs)} direct Hindi-Punjabi pairs from config '{direct_config}'")
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
    chosen = shared_en[:sample_size]
    pairs = [(hi_map[en], pa_map[en]) for en in chosen]
    print(f"[OK] Derived {len(pairs)} Hindi-Punjabi pairs via English pivot")
    return pairs


# ---------------------------------------------------------------------------
# Similarity scorers - mirror the two matching strategies app.py supports.
# ---------------------------------------------------------------------------

def semantic_score_matrix(semantic_model):
    """N-query x N-corpus cosine similarity matrix using the app's semantic model."""
    def _fn(translations, corpus_pa):
        trans_emb = semantic_model.encode(translations, normalize_embeddings=True)
        corpus_emb = semantic_model.encode(corpus_pa, normalize_embeddings=True)
        return np.array(trans_emb) @ np.array(corpus_emb).T
    return _fn


def jaccard_score_matrix(jaccard_fn):
    """N-query x N-corpus Jaccard word-overlap matrix (same function EBMT uses)."""
    def _fn(translations, corpus_pa):
        n, m = len(translations), len(corpus_pa)
        mat = np.zeros((n, m))
        for i, t in enumerate(translations):
            for j, c in enumerate(corpus_pa):
                mat[i, j] = jaccard_fn(t, c)
        return mat
    return _fn


# ---------------------------------------------------------------------------
# Classification-style evaluation
# ---------------------------------------------------------------------------

def evaluate_matcher(translations, corpus_pa, score_matrix_fn, threshold, label):
    """
    translations[i] is the system's translation for query i; corpus_pa[i] is
    its true reference Punjabi translation and also entry i of the corpus
    pool. sims[i][j] >= threshold means the matcher flagged corpus item j as
    a match for query i. j == i is the only true positive per query.
    """
    n = len(translations)
    sims = score_matrix_fn(translations, corpus_pa)
    tp = fp = fn = tn = 0
    for i in range(n):
        matched = sims[i] >= threshold
        for j in range(n):
            is_true = (i == j)
            is_matched = bool(matched[j])
            if is_true and is_matched:
                tp += 1
            elif is_true and not is_matched:
                fn += 1
            elif not is_true and is_matched:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        'label': label, 'threshold': threshold, 'n_queries': n,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
    }


def print_metrics(m):
    print(f"  [{m['label']}]  threshold={m['threshold']}  N={m['n_queries']}")
    print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"    Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  Accuracy={m['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate app.py's cross-language Hindi-Punjabi matcher on Samanantar (Precision/Recall/F1)"
    )
    parser.add_argument("--sample_size", type=int, default=100,
                         help="Number of Hindi-Punjabi pairs to sample (also the size of the distractor pool)")
    parser.add_argument("--pivot_scan_limit", type=int, default=300_000,
                         help="Rows to stream from each of the hi/pa configs while building the English pivot map")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic_threshold", type=float, default=0.55,
                         help="Matches app.py's EnhancedCorpusManager.search_corpus default threshold")
    parser.add_argument("--jaccard_threshold", type=float, default=0.3)
    parser.add_argument("--src_col", default=None, help="Override auto-detected English column name")
    parser.add_argument("--tgt_col", default=None, help="Override auto-detected Indic-language column name")
    parser.add_argument("--output", default=os.path.join("results", "samanantar_eval.json"))
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit("Missing dependency. Run: pip install datasets")

    print("=" * 80)
    print("Samanantar Cross-Language Matching Evaluation")
    print("=" * 80)

    print("\n[1/4] Loading app.py (this also loads the translation/semantic models)...")
    import app  # local import: executes app.py's module-level model loading

    print(f"\n[2/4] Loading {args.sample_size} Hindi-Punjabi pairs from Samanantar...")
    pairs = load_samanantar_pairs(args.sample_size, args.pivot_scan_limit, args.seed,
                                   args.src_col, args.tgt_col)
    hindi_sentences = [h for h, _ in pairs]
    reference_pa = [p for _, p in pairs]

    print(f"\n[3/4] Translating {len(pairs)} Hindi sentences "
          f"(Dictionary -> EBMT -> NMT cascade, plus EBMT-only and NMT-only)...")
    variant_translations = {'recommended': [], 'ebmt': [], 'nmt': []}
    method_counts = {}
    for i, h in enumerate(hindi_sentences, 1):
        result = app.translate_both_modes(h)
        variant_translations['recommended'].append(result['recommended']['translation'])
        variant_translations['ebmt'].append(result['ebmt']['translation'])
        variant_translations['nmt'].append(result['nmt']['translation'])
        method = result['recommended']['method']
        method_counts[method] = method_counts.get(method, 0) + 1
        if i % 25 == 0 or i == len(hindi_sentences):
            print(f"  Translated {i}/{len(hindi_sentences)}")

    print(f"\nCascade method breakdown (which mode 'won' per query): {method_counts}")
    if not app.model_cache.get('loaded'):
        print("[!] NMT model not loaded - 'nmt' variant translations are all "
              "'Models not loaded'; those rows below reflect a non-functional translator, not the real NMT mode.")
    if method_counts.get('EBMT', 0) == 0 and method_counts.get('Dictionary', 0) < len(pairs):
        print("[!] EBMT rarely/never matched Samanantar sentences against the app's small "
              "70-pair internal corpus - expected, since EBMT only generalizes to sentences "
              "similar to its own training examples.")

    print("\n[4/4] Scoring matches and computing Precision / Recall / F1...")
    scorers = []
    semantic_model = app.model_cache.get('semantic_model')
    if semantic_model:
        scorers.append((
            f"semantic ({app.model_cache.get('semantic_model_source')})",
            semantic_score_matrix(semantic_model),
            args.semantic_threshold,
        ))
    else:
        print("[!] No semantic model loaded - skipping semantic-matcher evaluation "
              "(train_indicbert.py or check internet access to Hugging Face).")
    scorers.append(("jaccard (lexical/EBMT-style)", jaccard_score_matrix(app.jaccard_similarity),
                     args.jaccard_threshold))

    all_results = []
    for variant_name, translations in variant_translations.items():
        for scorer_label, score_fn, threshold in scorers:
            label = f"{variant_name} translation + {scorer_label} matcher"
            print(f"\nEvaluating: {label} (threshold={threshold})")
            metrics = evaluate_matcher(translations, reference_pa, score_fn, threshold, label)
            print_metrics(metrics)
            all_results.append(metrics)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    output_payload = {
        'dataset': 'ai4bharat/samanantar (Hindi-Punjabi, English-pivoted unless a direct config existed)',
        'sample_size': len(pairs),
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'method_counts': method_counts,
        'results': all_results,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()
