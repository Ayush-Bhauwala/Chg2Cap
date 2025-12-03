import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from data.LEVIR_CC.vqa_dataset import LevirCCVQADataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.vqa_classifier import VQAWithAttention
from sentence_transformers import SentenceTransformer


def load_and_freeze_encoder(
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
    """Load pretrained encoder and freeze parameters."""
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

    # Freeze all parameters
    for param in encoder.parameters():
        param.requires_grad = False
    for param in encoder_trans.parameters():
        param.requires_grad = False

    encoder.eval()
    encoder_trans.eval()

    encoder = encoder.to(device)
    encoder_trans = encoder_trans.to(device)

    print("Encoder loaded and frozen!")
    return encoder, encoder_trans


def train_epoch(
    model,
    encoder,
    encoder_trans,
    dataloader,
    loss_function,
    optimizer,
    device,
    epoch,
    text_encoder,
):
    """Train for one epoch."""
    model.train()
    encoder.eval()
    encoder_trans.eval()

    tr_loss = 0
    nb_tr_steps = 0
    total_correct, total_predictions = 0, 0

    for idx, batch in enumerate(dataloader):
        # Unpack batch (imgA, imgB, input_seqs, output_seqs, lengths)
        imgA, imgB, question, ans = batch

        # Move to device
        imgA = imgA.to(device)
        imgB = imgB.to(device)
        ans = ans.to(device)

        # Extract features with frozen encoder
        with torch.no_grad():
            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)
            question_embedding = text_encoder.encode(
                question, convert_to_tensor=True
            ).to(device)
            question_embedding = question_embedding.clone()

        # Forward pass through change detection head
        logits = model(feat1, feat2, question_embedding)  # Shape: (batch_size,1)

        # Compute loss
        loss = loss_function(logits, ans)
        tr_loss += loss.item()
        nb_tr_steps += 1

        # Calculate accuracy
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct = torch.sum(predictions == ans)
        total_correct += correct.item()
        total_predictions += ans.size(0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            curr_avg_loss = tr_loss / nb_tr_steps
            curr_avg_acc = (
                total_correct / total_predictions if total_predictions > 0 else 0
            )
            print(
                f"Epoch {epoch} | Batch {idx}/{len(dataloader)} | "
                f"Loss: {curr_avg_loss:.4f} | Acc: {curr_avg_acc:.4f}"
            )

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accuracy = total_correct / total_predictions if total_predictions > 0 else 0

    print(f"\nEpoch {epoch} Complete:")
    print(f"  Training Loss: {epoch_loss:.4f}")
    print(f"  Training Accuracy: {epoch_accuracy:.4f}\n")

    return epoch_loss, epoch_accuracy


def validate(model, encoder, encoder_trans, dataloader, device, text_encoder):
    """Validate the model."""
    model.eval()
    encoder.eval()
    encoder_trans.eval()

    total_correct, total_predictions = 0, 0
    val_loss = 0
    nb_val_steps = 0

    loss_function = BCEWithLogitsLoss()

    with torch.no_grad():
        for imgA, imgB, question, ans in dataloader:
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            ans = ans.to(device)

            # Extract features
            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)
            question_embedding = text_encoder.encode(
                question, convert_to_tensor=True
            ).to(device)

            # Forward pass
            logits = model(feat1, feat2, question_embedding)

            # Loss
            loss = loss_function(logits, ans)
            val_loss += loss.item()
            nb_val_steps += 1

            # Accuracy
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct = torch.sum(predictions == ans)
            total_correct += correct.item()
            total_predictions += ans.size(0)

    avg_loss = val_loss / nb_val_steps
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0

    print(f"Validation - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}\n")
    return accuracy, avg_loss


def main():
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
    SAVE_DIR = "./models_checkpoint/"
    SEMANTIC_LABELS_FILE = "./data/LEVIR_CC/semantic_labels.json"

    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    ENCODER_DIM = 2048
    PATIENCE = 3

    # Create datasets
    train_dataset = LevirCCVQADataset(
        image_dir=IMAGE_DIR,
        semantic_labels_file=SEMANTIC_LABELS_FILE,
        split="train",
    )

    val_dataset = LevirCCVQADataset(
        image_dir=IMAGE_DIR,
        semantic_labels_file=SEMANTIC_LABELS_FILE,
        split="val",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load and freeze encoder
    encoder, encoder_trans = load_and_freeze_encoder(ENCODER_CHECKPOINT, device=DEVICE)

    print("Loading Text Encoder...")
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    text_encoder.to(DEVICE)
    TEXT_EMBED_DIM = 384

    # Initialize change detection head
    model = VQAWithAttention(
        in_channels=ENCODER_DIM, text_embed_dim=TEXT_EMBED_DIM, hidden_dim=512
    ).to(DEVICE)

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
    )

    # Loss and optimizer
    loss_function = BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"{'='*70}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*70}")

        # Train
        train_loss, train_acc = train_epoch(
            model,
            encoder,
            encoder_trans,
            train_loader,
            loss_function,
            optimizer,
            DEVICE,
            epoch + 1,
            text_encoder,
        )

        # Validate
        val_acc, val_loss = validate(
            model, encoder, encoder_trans, val_loader, DEVICE, text_encoder
        )
        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            save_path = os.path.join(SAVE_DIR, f"vqa_loss_{val_loss:.4f}.pth")
            torch.save(checkpoint, save_path)
            print(f"✓ Saved best model (val loss): {save_path}\n")
        else:
            patience_counter += 1
            print(
                f"No improvement in val loss for {patience_counter} epoch(s). "
                f"Patience: {PATIENCE}"
            )
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.\n")
                break

    print(
        f"Training complete! Best validation loss: {best_val_loss:.4f} | "
        f"Best validation accuracy: {best_val_acc:.4f}"
    )


if __name__ == "__main__":
    main()
