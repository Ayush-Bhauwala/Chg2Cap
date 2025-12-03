import os
import argparse
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

from data.LEVIR_CC.vqa_dataset import LevirCCVQADataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.vqa_classifier import VQAWithAttention
from sentence_transformers import SentenceTransformer


def load_vqa_model(checkpoint_path, encoder_dim, text_embed_dim, device):
    """Load the trained VQA model from checkpoint."""
    print(f"Loading VQA model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = VQAWithAttention(
        in_channels=encoder_dim, text_embed_dim=text_embed_dim, hidden_dim=512
    )
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


def test_vqa(model, encoder, encoder_trans, dataloader, text_encoder, device):
    """Test the VQA model and compute metrics."""
    model.eval()
    encoder.eval()
    encoder_trans.eval()

    all_predictions = []
    all_targets = []

    print("Testing VQA model...")
    with torch.no_grad():
        for idx, (imgA, imgB, question, ans) in enumerate(dataloader):
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            ans = ans.to(device)

            # Extract features
            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)

            # Encode question
            question_embedding = text_encoder.encode(
                question, convert_to_tensor=True
            ).to(device)

            # Forward pass
            logits = model(feat1, feat2, question_embedding)
            predictions = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            targets = ans.cpu().numpy()

            all_predictions.extend(predictions.flatten())
            all_targets.extend(targets.flatten())

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
    print("VQA TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print()

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - VQA")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va="center", ha="center", color="black")
    plt.savefig("vqa_confusion_matrix.png")
    print("Confusion matrix saved as 'vqa_confusion_matrix.png'")
    print()

    print("Detailed counts:")
    print(f"  True Positives (TP):  {tp} - Correctly predicted Yes")
    print(f"  True Negatives (TN):  {tn} - Correctly predicted No")
    print(f"  False Positives (FP): {fp} - Incorrectly predicted Yes")
    print(f"  False Negatives (FN): {fn} - Incorrectly predicted No")
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


def main(use_unseen_questions=False):
    # Device
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    print(f"Using device: {DEVICE}\n")

    # Paths
    IMAGE_DIR = "/home/ab6106/Levir-CC-dataset/images"
    ENCODER_CHECKPOINT = "Pretrained_models/LEVIR_CC_batchsize_32_resnet101.pth"
    SEMANTIC_LABELS_FILE = "./data/LEVIR_CC/semantic_labels.json"

    # Find the best VQA checkpoint
    CHECKPOINT_DIR = "./models_checkpoint/"
    vqa_checkpoints = [
        f
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("vqa") and f.endswith(".pth")
    ]
    if not vqa_checkpoints:
        raise FileNotFoundError(f"No VQA checkpoints found in {CHECKPOINT_DIR}")

    # Sort by accuracy in filename
    vqa_checkpoints.sort(
        key=lambda x: float(x.split("_")[-1].replace(".pth", "")), reverse=True
    )
    best_checkpoint = os.path.join(CHECKPOINT_DIR, vqa_checkpoints[0])
    print(f"Using checkpoint: {best_checkpoint}\n")

    # Hyperparameters
    BATCH_SIZE = 16
    ENCODER_DIM = 2048
    TEXT_EMBED_DIM = 384

    # Create test dataset
    test_dataset = LevirCCVQADataset(
        image_dir=IMAGE_DIR,
        semantic_labels_file=SEMANTIC_LABELS_FILE,
        split="test",
        use_unseen_questions=use_unseen_questions,
    )

    print(f"Test samples: {len(test_dataset)}\n")
    print(
        "Using unseen question variants for test set: " f"{bool(use_unseen_questions)}"
    )

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load models
    encoder, encoder_trans = load_encoder(ENCODER_CHECKPOINT, device=DEVICE)

    print("Loading Text Encoder...")
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    text_encoder.to(DEVICE)
    print("Text encoder loaded!\n")

    model = load_vqa_model(best_checkpoint, ENCODER_DIM, TEXT_EMBED_DIM, DEVICE)

    # Test
    metrics = test_vqa(
        model,
        encoder,
        encoder_trans,
        test_loader,
        text_encoder,
        DEVICE,
    )

    print("\nTesting complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA testing script")
    parser.add_argument(
        "--use-unseen-questions",
        action="store_true",
        help="Include unseen paraphrased question variants in the test set",
    )
    args = parser.parse_args()
    main(use_unseen_questions=args.use_unseen_questions)
