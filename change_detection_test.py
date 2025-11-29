import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from data.LEVIR_CC.custom_dataset import ChangeDetectionCaptionDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.change_detector import SiameseChangeHead

# No-change captions list (from original test.py)
levir_nochange_list = [
    "the scene is the same as before ",
    "there is no difference ",
    "the two scenes seem identical ",
    "no change has occurred ",
    "almost nothing has changed ",
]


def load_change_detector_model(checkpoint_path, encoder_dim, device):
    """Load the trained change detection model from checkpoint."""
    print(f"Loading change detector from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SiameseChangeHead(in_channels=encoder_dim, reduced_dim=512)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(
        f"Model loaded! Val accuracy from checkpoint: {checkpoint.get('val_acc', 'N/A'):.4f}\n"
    )
    return model


def load_encoder(checkpoint_path, device):
    """Load frozen encoder and encoder_trans."""
    print(f"Loading encoder from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Encoder hyperparameters
    network = "resnet101"
    encoder_dim = 2048
    feat_size = 16
    n_layers = 3
    n_heads = 8
    hidden_dim = 512
    attention_dim = 2048
    dropout = 0.1

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


def is_change_caption(caption_tokens, word2idx, nochange_list):
    """Determine if a caption indicates change or no-change."""
    # Convert tokens back to string
    id_to_word = {v: k for k, v in word2idx.items()}
    caption_str = " ".join([id_to_word.get(tok, "<UNK>") for tok in caption_tokens])

    # Remove special tokens for comparison
    caption_str = (
        caption_str.replace("<START>", "")
        .replace("<END>", "")
        .replace("<NULL>", "")
        .strip()
    )
    # print(f"Caption: '{caption_str}'")

    # Check if it's in no-change list
    # return 0 if caption_str in nochange_list else 1
    # print(f"Checking against no-change list...")
    for phrase in nochange_list:
        if phrase.lower().strip() == caption_str.lower().strip():
            # print(f"Matched no-change phrase: '{phrase}'")
            return 0

    # print("No match found in no-change list.")
    # print("\n")
    return 1


def test_change_detection(
    model, encoder, encoder_trans, dataloader, word2idx, nochange_list, device
):
    """Test the change detection model and compute metrics."""
    model.eval()
    encoder.eval()
    encoder_trans.eval()

    all_predictions = []
    all_targets = []

    print("Testing change detection...")
    with torch.no_grad():
        for idx, (imgA, imgB, input_seqs, _, _) in enumerate(dataloader):
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            input_seqs = input_seqs.to(device)

            # Extract features
            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)

            # Forward pass
            logits = model(feat1, feat2).squeeze(1)
            predictions = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

            # Determine targets
            targets = []
            for seq in input_seqs:
                seq_list = seq.cpu().tolist()
                target = is_change_caption(seq_list, word2idx, nochange_list)
                targets.append(target)
            targets = np.array(targets)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(dataloader)} batches...")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Compute metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nComputing evaluation metrics on {len(all_targets)} samples...\n")

    print("=" * 70)
    print("CHANGE DETECTION TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print()

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Change", "Change"])
    ax.set_yticklabels(["No Change", "Change"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va="center", ha="center", color="black")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")
    print()

    print("Detailed counts:")
    print(f"  True Positives (TP):  {tp} - Correctly predicted change")
    print(f"  True Negatives (TN):  {tn} - Correctly predicted no change")
    print(f"  False Positives (FP): {fp} - Incorrectly predicted change")
    print(f"  False Negatives (FN): {fn} - Incorrectly predicted no change")
    print("=" * 70)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


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

    # Find the best change detector checkpoint
    CHECKPOINT_DIR = "./models_checkpoint/"
    change_detector_checkpoints = [
        f
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("change_detector") and f.endswith(".pth")
    ]
    if not change_detector_checkpoints:
        raise FileNotFoundError(
            f"No change detector checkpoints found in {CHECKPOINT_DIR}"
        )

    # Sort by accuracy in filename
    change_detector_checkpoints.sort(
        key=lambda x: float(x.split("_")[-1].replace(".pth", "")), reverse=True
    )
    best_checkpoint = os.path.join(CHECKPOINT_DIR, change_detector_checkpoints[0])
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
    model = load_change_detector_model(best_checkpoint, ENCODER_DIM, DEVICE)

    # Test
    metrics = test_change_detection(
        model,
        encoder,
        encoder_trans,
        test_loader,
        test_dataset.word2idx,
        levir_nochange_list,
        DEVICE,
    )

    print("\nTesting complete!")


if __name__ == "__main__":
    main()
