import os
import argparse
import torch
import numpy as np
from imageio import imread
from sentence_transformers import SentenceTransformer

from model.model_encoder import Encoder, AttentiveEncoder
from model.vqa_classifier import VQAWithAttention


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
    return encoder, encoder_trans


def preprocess_images(image_path, image_dir):
    """Load and preprocess image pair."""
    mean = [100.6790, 99.5023, 84.9932]
    std = [50.9820, 48.4838, 44.7057]

    # Construct full paths
    # Assuming image_path is relative to the 'images' directory structure

    if os.path.isabs(image_path):
        base_dir = os.path.dirname(
            os.path.dirname(image_path)
        ) 
        filename = os.path.basename(image_path)
        # Check if path contains 'A' or 'B'
        if "/A/" in image_path:
            img_fileA = image_path
            img_fileB = image_path.replace("/A/", "/B/")
        elif "/B/" in image_path:
            img_fileB = image_path
            img_fileA = image_path.replace("/B/", "/A/")
        else:
            img_fileA = image_path
            img_fileB = os.path.join(
                os.path.dirname(os.path.dirname(image_path)), "B", filename
            )
    else:
        if "/" in image_path:
            # Assume format like "test/A/test_000001.png"
            img_fileA = os.path.join(image_dir, image_path)
            if "/A/" in image_path:
                img_fileB = os.path.join(image_dir, image_path.replace("/A/", "/B/"))
            else:
                # If user pointed to B, swap to A
                img_fileB = os.path.join(image_dir, image_path)
                img_fileA = os.path.join(image_dir, image_path.replace("/B/", "/A/"))
        else:
            img_fileA = os.path.join(image_dir, "test", "A", image_path)
            img_fileB = os.path.join(image_dir, "test", "B", image_path)

    print(f"Image A: {img_fileA}")
    print(f"Image B: {img_fileB}")

    if not os.path.exists(img_fileA) or not os.path.exists(img_fileB):
        raise FileNotFoundError("Could not find one or both images.")

    imgA = imread(img_fileA).astype(np.float32)
    imgB = imread(img_fileB).astype(np.float32)

    imgA = np.moveaxis(imgA, -1, 0)
    imgB = np.moveaxis(imgB, -1, 0)

    for i in range(len(mean)):
        imgA[i, :, :] -= mean[i]
        imgA[i, :, :] /= std[i]
        imgB[i, :, :] -= mean[i]
        imgB[i, :, :] /= std[i]

    # Add batch dimension
    imgA = torch.from_numpy(imgA).unsqueeze(0)
    imgB = torch.from_numpy(imgB).unsqueeze(0)

    return imgA, imgB


def predict(model, encoder, encoder_trans, text_encoder, imgA, imgB, question, device):
    """Run inference."""
    imgA = imgA.to(device)
    imgB = imgB.to(device)

    with torch.no_grad():
        # Extract features
        feat1, feat2 = encoder(imgA, imgB)
        feat1, feat2 = encoder_trans(feat1, feat2)

        # Encode question
        question_embedding = text_encoder.encode(question, convert_to_tensor=True).to(
            device
        )

        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.unsqueeze(0)

        # Forward pass
        logits = model(feat1, feat2, question_embedding)
        probability = torch.sigmoid(logits).item()

        prediction = "Yes" if probability > 0.5 else "No"

    return prediction, probability


def main():
    parser = argparse.ArgumentParser(description="VQA Inference on Single Image Pair")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Relative path to image (e.g., 'test/A/test_000001.png' or just 'test_000001.png')",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the image pair",
    )
    args = parser.parse_args()

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    print(f"Using device: {DEVICE}\n")

    IMAGE_DIR = "/home/ab6106/Levir-CC-dataset/images"
    ENCODER_CHECKPOINT = "Pretrained_models/LEVIR_CC_batchsize_32_resnet101.pth"
    CHECKPOINT_DIR = "./models_checkpoint/"

    vqa_checkpoints = [
        f
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("vqa") and f.endswith(".pth")
    ]
    if not vqa_checkpoints:
        raise FileNotFoundError(f"No VQA checkpoints found in {CHECKPOINT_DIR}")

    vqa_checkpoints.sort(
        key=lambda x: float(x.split("_")[-1].replace(".pth", "")), reverse=True
    )
    best_checkpoint = os.path.join(CHECKPOINT_DIR, vqa_checkpoints[0])

    ENCODER_DIM = 2048
    TEXT_EMBED_DIM = 384

    encoder, encoder_trans = load_encoder(ENCODER_CHECKPOINT, device=DEVICE)

    print("Loading Text Encoder...")
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    text_encoder.to(DEVICE)

    model = load_vqa_model(best_checkpoint, ENCODER_DIM, TEXT_EMBED_DIM, DEVICE)

    try:
        imgA, imgB = preprocess_images(args.image_path, IMAGE_DIR)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    print(f"\nQuestion: {args.question}")
    prediction, probability = predict(
        model, encoder, encoder_trans, text_encoder, imgA, imgB, args.question, DEVICE
    )

    print(f"Answer: {prediction}")
    print(f"Confidence (Yes probability): {probability:.4f}")


if __name__ == "__main__":
    main()
