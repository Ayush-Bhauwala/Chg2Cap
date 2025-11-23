import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

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

    print(f"Model loaded! Val accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}\n")
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


def denormalize_image(img_tensor, mean, std):
    """Denormalize image tensor back to [0, 255] range."""
    img = img_tensor.clone()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


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


def visualize_samples(
    model,
    encoder,
    encoder_trans,
    dataloader,
    word_vocab,
    device,
    num_samples=10,
    save_path="sample_results.png",
):
    """Generate and visualize sample predictions."""
    model.eval()
    encoder.eval()
    encoder_trans.eval()

    id_to_word = {v: k for k, v in word_vocab.items()}
    special_tokens = {word_vocab["<START>"], word_vocab["<END>"], word_vocab["<NULL>"]}

    # Image normalization parameters (from LEVIR_CC dataset)
    mean = [100.6790, 99.5023, 84.9932]
    std = [50.9820, 48.4838, 44.7057]

    samples = []

    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        for idx, (imgA, imgB, input_seqs, output_seqs, lengths) in enumerate(
            dataloader
        ):
            if len(samples) >= num_samples:
                break

            imgA = imgA.to(device)
            imgB = imgB.to(device)

            # Process first image in batch
            generated = generate_caption(
                model,
                encoder,
                encoder_trans,
                imgA[0:1],
                imgB[0:1],
                word_vocab,
                device=device,
            )

            # Convert tokens to words
            pred_tokens = [t for t in generated if t not in special_tokens]
            pred_caption = " ".join([id_to_word.get(t, "<UNK>") for t in pred_tokens])

            ref_tokens = input_seqs[0].cpu().tolist()
            ref_tokens = [t for t in ref_tokens if t not in special_tokens]
            ref_caption = " ".join([id_to_word.get(t, "<UNK>") for t in ref_tokens])

            # Denormalize images
            imgA_vis = denormalize_image(imgA[0], mean, std)
            imgB_vis = denormalize_image(imgB[0], mean, std)

            samples.append(
                {
                    "imgA": imgA_vis,
                    "imgB": imgB_vis,
                    "reference": ref_caption,
                    "predicted": pred_caption,
                }
            )

            print(f"Sample {len(samples)}/{num_samples} collected")

    # Create visualization
    fig = plt.figure(figsize=(20, 4 * num_samples))

    for i, sample in enumerate(samples):
        # Image A (Before)
        ax1 = plt.subplot(num_samples, 3, i * 3 + 1)
        ax1.imshow(sample["imgA"])
        ax1.set_title(f"Sample {i+1}: Image A (Before)", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Image B (After)
        ax2 = plt.subplot(num_samples, 3, i * 3 + 2)
        ax2.imshow(sample["imgB"])
        ax2.set_title(f"Image B (After)", fontsize=10, fontweight="bold")
        ax2.axis("off")

        # Captions
        ax3 = plt.subplot(num_samples, 3, i * 3 + 3)
        ax3.axis("off")

        caption_text = (
            f"Reference:\n{sample['reference']}\n\n"
            f"Generated:\n{sample['predicted']}"
        )

        ax3.text(
            0.1,
            0.5,
            caption_text,
            verticalalignment="center",
            fontsize=9,
            wrap=True,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.close()


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
    BATCH_SIZE = 1  # Process one at a time for visualization
    MAX_SEQ_LEN = 50
    ENCODER_DIM = 2048
    NUM_SAMPLES = 10

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

    print(f"Test samples available: {len(test_dataset)}\n")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle to get random samples each run
        num_workers=0,
        pin_memory=False,
    )

    # Load models
    encoder, encoder_trans = load_encoder(ENCODER_CHECKPOINT, device=DEVICE)
    model = load_model(best_checkpoint, vocab_size, ENCODER_DIM, DEVICE)

    # Generate visualizations
    visualize_samples(
        model,
        encoder,
        encoder_trans,
        test_loader,
        word_vocab,
        DEVICE,
        num_samples=NUM_SAMPLES,
        save_path="test_sample_results.png",
    )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
