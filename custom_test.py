import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer

from data.LEVIR_CC.custom_dataset import ChangeDetectionCaptionDataset
from model.model_encoder import Encoder, AttentiveEncoder
from custom_train import LSTMCaptionGeneratorModel


def load_model(checkpoint_path, vocab_size, encoder_dim, device):
    """Load the trained LSTM model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = LSTMCaptionGeneratorModel(vocab_size=vocab_size, encoder_dim=encoder_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(
        f"Model loaded! Val accuracy from checkpoint: {checkpoint.get('val_acc', 'N/A'):.4f}\n"
    )
    return model


def load_encoder(
    checkpoint_path,
    network="resnet101",
    encoder_dim=2048,
    feat_size=16,
    n_layers=3,
    n_heads=8,
    hidden_dim=512,
    attention_dim=2048,
    dropout=0.1,
    device="cuda",
):
    """Load frozen encoder."""
    print(f"Loading encoder from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = Encoder(network)
    encoder_trans = AttentiveEncoder(
        n_layers=n_layers,
        feature_size=[feat_size, feat_size, encoder_dim],
        heads=n_heads,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        dropout=dropout,
    )

    encoder.load_state_dict(checkpoint["encoder_dict"])
    encoder_trans.load_state_dict(checkpoint["encoder_trans_dict"])

    encoder.eval()
    encoder_trans.eval()
    encoder = encoder.to(device)
    encoder_trans = encoder_trans.to(device)

    print("Encoder loaded!\n")
    return encoder, encoder_trans


def generate_caption(
    model, encoder, encoder_trans, imgA, imgB, word_vocab, max_length=50, device="cuda"
):
    """Generate caption using greedy decoding."""
    model.eval()

    with torch.no_grad():
        # Extract features
        feat1, feat2 = encoder(imgA, imgB)
        feat1, feat2 = encoder_trans(feat1, feat2)

        # Start with <START> token
        start_token = word_vocab["<START>"]
        end_token = word_vocab["<END>"]

        # Initialize sequence with start token
        generated = [start_token]
        input_seq = torch.tensor([[start_token]], dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Forward pass
            logits = model(feat1, feat2, input_seq)

            # Get prediction for last position
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token)

            # Stop if end token is generated
            if next_token == end_token:
                break

            # Append to input sequence
            input_seq = torch.cat(
                [
                    input_seq,
                    torch.tensor([[next_token]], dtype=torch.long, device=device),
                ],
                dim=1,
            )

    return generated


def evaluate_metrics(references, hypotheses, id_to_word):
    """
    Compute metrics using pure Python implementations.

    Args:
        references: list of lists of token IDs (references)
        hypotheses: list of token IDs (predictions)
        id_to_word: dict mapping token IDs to words
    """
    smoothing = SmoothingFunction().method4

    # Convert token IDs to word lists for BLEU
    refs_words = [
        [[id_to_word.get(tok, "<UNK>") for tok in ref] for ref in refs]
        for refs in references
    ]
    hyps_words = [[id_to_word.get(tok, "<UNK>") for tok in hyp] for hyp in hypotheses]

    # BLEU scores (corpus-level)
    bleu1 = corpus_bleu(
        refs_words, hyps_words, weights=(1, 0, 0, 0), smoothing_function=smoothing
    )
    bleu2 = corpus_bleu(
        refs_words, hyps_words, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing
    )
    bleu3 = corpus_bleu(
        refs_words,
        hyps_words,
        weights=(0.33, 0.33, 0.33, 0),
        smoothing_function=smoothing,
    )
    bleu4 = corpus_bleu(
        refs_words,
        hyps_words,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    # METEOR scores (sentence-level averaged)
    meteor_scores = []
    for refs, hyp in zip(refs_words, hyps_words):
        refs_str = [" ".join(ref) for ref in refs]
        hyp_str = " ".join(hyp)
        try:
            score = nltk_meteor(refs_str, hyp_str)
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)
    meteor = np.mean(meteor_scores)

    # ROUGE-L scores
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for refs, hyp in zip(refs_words, hyps_words):
        hyp_str = " ".join(hyp)
        scores = [
            scorer.score(" ".join(ref), hyp_str)["rougeL"].fmeasure for ref in refs
        ]
        rouge_scores.append(max(scores) if scores else 0.0)
    rouge_l = np.mean(rouge_scores)

    return {
        "Bleu_1": bleu1,
        "Bleu_2": bleu2,
        "Bleu_3": bleu3,
        "Bleu_4": bleu4,
        "METEOR": meteor,
        "ROUGE_L": rouge_l,
    }


def test(model, encoder, encoder_trans, dataloader, word_vocab, device, verbose=True):
    """Test the model and compute metrics."""
    model.eval()
    encoder.eval()
    encoder_trans.eval()

    references = []
    hypotheses = []

    id_to_word = {v: k for k, v in word_vocab.items()}
    special_tokens = {word_vocab["<START>"], word_vocab["<END>"], word_vocab["<NULL>"]}

    print("Generating captions...")
    with torch.no_grad():
        for idx, (imgA, imgB, input_seqs, output_seqs, lengths) in enumerate(
            dataloader
        ):
            imgA = imgA.to(device)
            imgB = imgB.to(device)

            batch_size = imgA.shape[0]

            for i in range(batch_size):
                # Generate caption
                generated = generate_caption(
                    model,
                    encoder,
                    encoder_trans,
                    imgA[i : i + 1],
                    imgB[i : i + 1],
                    word_vocab,
                    device=device,
                )

                # Remove special tokens
                pred_tokens = [t for t in generated if t not in special_tokens]
                hypotheses.append(pred_tokens)

                # Get reference (ground truth)
                ref_tokens = input_seqs[i].cpu().tolist()
                ref_tokens = [t for t in ref_tokens if t not in special_tokens]
                references.append([ref_tokens])  # Wrap in list for BLEU evaluation

                # Print samples
                if verbose and idx < 5 and i == 0:
                    pred_caption = " ".join(
                        [id_to_word.get(t, "<UNK>") for t in pred_tokens]
                    )
                    ref_caption = " ".join(
                        [id_to_word.get(t, "<UNK>") for t in ref_tokens]
                    )
                    print(f"\nSample {idx + 1}:")
                    print(f"  Reference: {ref_caption}")
                    print(f"  Generated: {pred_caption}")

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(dataloader)} batches...")

    print(f"\nComputing evaluation metrics on {len(references)} samples...\n")

    # Compute metrics using pure Python
    metrics = evaluate_metrics(references, hypotheses, id_to_word)

    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"BLEU-1:  {metrics['Bleu_1']:.4f}")
    print(f"BLEU-2:  {metrics['Bleu_2']:.4f}")
    print(f"BLEU-3:  {metrics['Bleu_3']:.4f}")
    print(f"BLEU-4:  {metrics['Bleu_4']:.4f}")
    print(f"METEOR:  {metrics['METEOR']:.4f}")
    print(f"ROUGE-L: {metrics['ROUGE_L']:.4f}")
    print("=" * 70)

    return metrics


def main():
    # Device
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    print(f"Using device: {DEVICE}\n")

    # Paths
    IMAGE_DIR = "/Users/ayushbhauwala/Documents/Columbia/Sem 1/DL for CV/project/experimenting/levir_cc/Levir-CC-dataset/images"
    CAPTIONS_FILE = "/Users/ayushbhauwala/Documents/Columbia/Sem 1/DL for CV/project/experimenting/levir_cc/Levir-CC-dataset/LevirCCcaptions.json"
    VOCAB_FILE = "./data/LEVIR_CC/vocab.json"
    ENCODER_CHECKPOINT = "Pretrained_models/LEVIR_CC_batchsize_32_resnet101.pth"

    # Find the best model checkpoint
    CHECKPOINT_DIR = "./models_checkpoint/"
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")

    # Sort by accuracy in filename
    checkpoints.sort(
        key=lambda x: float(x.split("_")[-1].replace(".pth", "")[3:]), reverse=True
    )
    best_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[0])
    print(f"Using checkpoint: {best_checkpoint}\n")

    # Hyperparameters
    BATCH_SIZE = 8
    MAX_SEQ_LEN = 50
    ENCODER_DIM = 2048

    # Load vocabulary
    with open(VOCAB_FILE, "r") as f:
        word_vocab = json.load(f)
    vocab_size = len(word_vocab)
    print(f"Vocabulary size: {vocab_size}\n")

    # Create test dataset
    test_dataset = ChangeDetectionCaptionDataset(
        image_dir=IMAGE_DIR,
        captions_file=CAPTIONS_FILE,
        vocab_file=VOCAB_FILE,
        split="test",
        max_seq_len=MAX_SEQ_LEN,
        sample_one_caption=True,
    )

    print(f"Test samples: {len(test_dataset)}\n")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load models
    encoder, encoder_trans = load_encoder(ENCODER_CHECKPOINT, device=DEVICE)
    model = load_model(best_checkpoint, vocab_size, ENCODER_DIM, DEVICE)

    # Test
    metrics = test(model, encoder, encoder_trans, test_loader, word_vocab, DEVICE)

    print("\nTesting complete!")


if __name__ == "__main__":
    main()
