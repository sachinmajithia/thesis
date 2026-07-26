# Fine-tune IndicBERT (ai4bharat/indic-bert) into a Hindi-Punjabi cross-lingual
# sentence-similarity encoder, using the custom parallel dataset built for
# this thesis (data/indicbert_training_data.csv). The resulting model is
# consumed by app.py's cross-language plagiarism detection module in place
# of the generic pretrained IndicSBERT model.
#
# Usage:
#   python train_indicbert.py \
#       --data data/indicbert_training_data.csv \
#       --output models/indicbert-finetuned-hi-pa \
#       --epochs 4 --batch_size 16

import argparse
import csv
import math
import os
import random

try:
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, InputExample, losses, models
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
except ImportError as e:
    print(f"ERROR: Missing dependency - {e}")
    print("Please run: pip install -r requirements.txt")
    exit(1)

BASE_MODEL = "ai4bharat/indic-bert"
POSITIVE_LABEL = 1.0
NEGATIVE_LABEL = 0.05


def build_base_sentence_model(base_model_name: str = BASE_MODEL) -> "SentenceTransformer":
    """
    IndicBERT is a plain transformers encoder (ALBERT-based), not a
    sentence-transformers model, so it has no built-in pooling head. Wrap it
    with mean-pooling over token embeddings to turn it into a sentence
    encoder that can be fine-tuned with a similarity objective.
    """
    word_embedding_model = models.Transformer(base_model_name, max_seq_length=128)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def load_positive_pairs(csv_path: str):
    """Load the custom Hindi-Punjabi parallel dataset (columns: hindi, punjabi)."""
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hindi = (row.get("hindi") or "").strip()
            punjabi = (row.get("punjabi") or "").strip()
            if hindi and punjabi:
                pairs.append((hindi, punjabi))
    return pairs


def build_training_examples(pairs, negatives_per_positive: int = 1, seed: int = 42):
    """
    Build a labelled InputExample set for CosineSimilarityLoss:
      - every true Hindi-Punjabi translation pair gets label 1.0
      - each Hindi sentence is additionally paired with a random
        non-corresponding Punjabi sentence (hard negative) at a low label,
        so the model learns to separate matching from non-matching content
        rather than mapping everything to the same embedding.
    """
    rng = random.Random(seed)
    examples = [InputExample(texts=[h, p], label=POSITIVE_LABEL) for h, p in pairs]

    punjabi_pool = [p for _, p in pairs]
    for hindi, punjabi in pairs:
        candidates = [c for c in punjabi_pool if c != punjabi]
        if not candidates:
            continue
        for _ in range(negatives_per_positive):
            negative = rng.choice(candidates)
            examples.append(InputExample(texts=[hindi, negative], label=NEGATIVE_LABEL))

    rng.shuffle(examples)
    return examples


def split_train_dev(examples, dev_ratio: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_ratio))
    return shuffled[n_dev:], shuffled[:n_dev]


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune IndicBERT for Hindi-Punjabi cross-lingual sentence similarity"
    )
    parser.add_argument("--data", default=os.path.join("data", "indicbert_training_data.csv"),
                         help="CSV with columns: hindi,punjabi")
    parser.add_argument("--output", default=os.path.join("models", "indicbert-finetuned-hi-pa"))
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--negatives_per_positive", type=int, default=1)
    args = parser.parse_args()

    print("=" * 80)
    print("IndicBERT Fine-tuning: Hindi-Punjabi Cross-Lingual Sentence Similarity")
    print("=" * 80)

    print(f"\n[1/4] Loading custom dataset from '{args.data}' ...")
    pairs = load_positive_pairs(args.data)
    if not pairs:
        raise SystemExit(f"No usable rows found in {args.data}. Expected columns: hindi,punjabi")
    print(f"  Loaded {len(pairs)} Hindi-Punjabi sentence pairs")

    examples = build_training_examples(pairs, negatives_per_positive=args.negatives_per_positive)
    train_examples, dev_examples = split_train_dev(examples)
    print(f"  Total examples (positive + synthesized negatives): {len(examples)}")
    print(f"  Train: {len(train_examples)}  |  Dev: {len(dev_examples)}")

    print(f"\n[2/4] Building base model from '{args.base_model}' (mean pooling) ...")
    model = build_base_sentence_model(args.base_model)

    print("\n[3/4] Training ...")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = None
    if dev_examples:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name="hi-pa-dev")

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * args.warmup_ratio)
    os.makedirs(args.output, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output,
        show_progress_bar=True,
    )

    print(f"\n[4/4] Fine-tuned model saved to: {args.output}")
    print("\nRestart app.py (or call load_models() again) — it automatically picks up a")
    print(f"fine-tuned checkpoint at '{args.output}' and uses it for cross-language")
    print("plagiarism detection instead of the pretrained IndicSBERT model.")


if __name__ == "__main__":
    main()
