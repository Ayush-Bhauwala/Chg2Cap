import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import json

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


def train_epoch(
    model,
    encoder,
    encoder_trans,
    dataloader,
    loss_function,
    optimizer,
    device,
    epoch,
    word2idx,
    nochange_list,
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
        imgA, imgB, input_seqs, _, _ = batch

        # Move to device
        imgA = imgA.to(device)
        imgB = imgB.to(device)
        input_seqs = input_seqs.to(device)

        # Extract features with frozen encoder
        with torch.no_grad():
            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)

        # Determine targets based on captions
        targets = []
        for seq in input_seqs:
            # Convert tensor to list and determine if change
            seq_list = seq.cpu().tolist()
            target = is_change_caption(seq_list, word2idx, nochange_list)
            targets.append(target)
        targets = torch.tensor(targets, dtype=torch.float32, device=device)

        # Forward pass through change detection head
        logits = model(feat1, feat2).squeeze(1)  # Shape: (batch_size,)

        # Compute loss
        loss = loss_function(logits, targets)
        tr_loss += loss.item()
        nb_tr_steps += 1

        # Calculate accuracy
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct = torch.sum(predictions == targets)
        total_correct += correct.item()
        total_predictions += targets.size(0)

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


def validate(
    model, encoder, encoder_trans, dataloader, device, word2idx, nochange_list
):
    """Validate the model."""
    model.eval()
    encoder.eval()
    encoder_trans.eval()

    total_correct, total_predictions = 0, 0
    val_loss = 0
    nb_val_steps = 0

    loss_function = BCEWithLogitsLoss()

    with torch.no_grad():
        for imgA, imgB, input_seqs, _, _ in dataloader:
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            input_seqs = input_seqs.to(device)

            # Extract features
            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)

            # Determine targets
            targets = []
            for seq in input_seqs:
                seq_list = seq.cpu().tolist()
                target = is_change_caption(seq_list, word2idx, nochange_list)
                targets.append(target)
            targets = torch.tensor(targets, dtype=torch.float32, device=device)

            # Forward pass
            logits = model(feat1, feat2).squeeze(1)

            # Loss
            loss = loss_function(logits, targets)
            val_loss += loss.item()
            nb_val_steps += 1

            # Accuracy
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct = torch.sum(predictions == targets)
            total_correct += correct.item()
            total_predictions += targets.size(0)

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
    CAPTIONS_FILE = "/home/ab6106/Levir-CC-dataset/LevirCCcaptions.json"
    VOCAB_FILE = "./data/LEVIR_CC/vocab.json"
    ENCODER_CHECKPOINT = "Pretrained_models/LEVIR_CC_batchsize_32_resnet101.pth"
    SAVE_DIR = "./models_checkpoint/"

    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    MAX_SEQ_LEN = 50  # Fixed sequence length
    ENCODER_DIM = 2048
    PATIENCE = 3

    # Load vocabulary
    with open(VOCAB_FILE, "r") as f:
        word_vocab = json.load(f)
    vocab_size = len(word_vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Create datasets
    train_dataset = ChangeDetectionCaptionDataset(
        image_dir=IMAGE_DIR,
        captions_file=CAPTIONS_FILE,
        vocab_file=VOCAB_FILE,
        split="train",
        max_seq_len=MAX_SEQ_LEN,
        sample_one_caption=True,
    )

    val_dataset = ChangeDetectionCaptionDataset(
        image_dir=IMAGE_DIR,
        captions_file=CAPTIONS_FILE,
        vocab_file=VOCAB_FILE,
        split="val",
        max_seq_len=MAX_SEQ_LEN,
        sample_one_caption=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load and freeze encoder
    encoder, encoder_trans = load_and_freeze_encoder(ENCODER_CHECKPOINT, device=DEVICE)

    # Initialize change detection head
    model = SiameseChangeHead(in_channels=ENCODER_DIM, reduced_dim=512).to(DEVICE)

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
            train_dataset.word2idx,
            levir_nochange_list,
        )

        # Validate
        val_acc, val_loss = validate(
            model,
            encoder,
            encoder_trans,
            val_loader,
            DEVICE,
            val_dataset.word2idx,
            levir_nochange_list,
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
            save_path = os.path.join(
                SAVE_DIR, f"change_detector_loss_{val_loss:.4f}.pth"
            )
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
