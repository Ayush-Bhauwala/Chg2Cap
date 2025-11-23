import torch
import torch.nn as nn
import numpy as np
from imageio import imread
import json
import argparse
from pathlib import Path

from model.model_encoder import Encoder, AttentiveEncoder
from model.lstm_decoder import LSTMCaptionGeneratorModel

def load_and_preprocess_image(img_path, mean, std):
    """
    Load and preprocess a single image.
    
    Args:
        img_path: Path to image file
        mean: Normalization mean values
        std: Normalization std values
    
    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    img = imread(img_path)
    img = np.asarray(img, dtype=np.float32)
    
    # Convert HWC to CHW
    img = np.moveaxis(img, -1, 0)
    
    # Normalize
    for i in range(3):
        img[i] = (img[i] - mean[i]) / std[i]
    
    # Add batch dimension
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor


def greedy_decode(model, encoder, encoder_trans, imgA, imgB, 
                  start_idx, end_idx, max_len, device):
    """
    Generate caption using greedy decoding (always pick highest probability word).
    
    Args:
        model: LSTM decoder model
        encoder: CNN encoder
        encoder_trans: Transformer encoder
        imgA, imgB: Input images
        start_idx: <START> token index
        end_idx: <END> token index
        max_len: Maximum caption length
        device: torch device
    
    Returns:
        List of token indices
    """
    model.eval()
    encoder.eval()
    encoder_trans.eval()
    
    with torch.no_grad():
        # Extract image features
        feat1, feat2 = encoder(imgA, imgB)
        feat1, feat2 = encoder_trans(feat1, feat2)
        
        # Start with <START> token
        generated = [start_idx]
        
        # Generate word by word
        for _ in range(max_len):
            # Create input sequence from generated tokens so far
            input_seq = torch.tensor([generated], dtype=torch.long).to(device)  # (1, seq_len)
            
            # Get predictions
            logits = model(feat1, feat2, input_seq)  # (1, seq_len, vocab_size)
            
            # Get prediction for last position
            last_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Pick token with highest probability
            next_token = torch.argmax(last_logits).item()
            
            # Add to generated sequence
            generated.append(next_token)
            
            # Stop if we generate <END> token
            if next_token == end_idx:
                break
        
        return generated


def beam_search_decode(model, encoder, encoder_trans, imgA, imgB,
                       start_idx, end_idx, max_len, beam_size, device):
    """
    Generate caption using beam search decoding.
    
    Args:
        model: LSTM decoder model
        encoder: CNN encoder
        encoder_trans: Transformer encoder
        imgA, imgB: Input images
        start_idx: <START> token index
        end_idx: <END> token index
        max_len: Maximum caption length
        beam_size: Number of beams to keep
        device: torch device
    
    Returns:
        Best sequence of token indices
    """
    model.eval()
    encoder.eval()
    encoder_trans.eval()
    
    with torch.no_grad():
        # Extract image features
        feat1, feat2 = encoder(imgA, imgB)
        feat1, feat2 = encoder_trans(feat1, feat2)
        
        # Initialize beams: (sequence, score)
        beams = [([start_idx], 0.0)]
        completed = []
        
        for step in range(max_len):
            candidates = []
            
            for seq, score in beams:
                # Stop if sequence already ended
                if seq[-1] == end_idx:
                    completed.append((seq, score))
                    continue
                
                # Prepare input
                input_seq = torch.tensor([seq], dtype=torch.long).to(device)
                
                # Get predictions
                logits = model(feat1, feat2, input_seq)
                last_logits = logits[0, -1, :]  # (vocab_size,)
                
                # Get log probabilities
                log_probs = torch.log_softmax(last_logits, dim=0)
                
                # Get top k tokens
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                
                # Create new candidates
                for log_prob, idx in zip(top_log_probs, top_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))
            
            # Select top beam_size candidates
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Stop if all beams ended
            if all(seq[-1] == end_idx for seq, _ in beams):
                break
        
        # Add remaining beams to completed
        completed.extend(beams)
        
        # Return best sequence
        best_seq, _ = max(completed, key=lambda x: x[1])
        return best_seq


def tokens_to_caption(token_indices, idx2word, start_idx, end_idx, pad_idx):
    """
    Convert token indices to readable caption string.
    
    Args:
        token_indices: List of token indices
        idx2word: Dictionary mapping indices to words
        start_idx, end_idx, pad_idx: Special token indices
    
    Returns:
        Caption string
    """
    words = []
    for idx in token_indices:
        # Skip special tokens
        if idx in [start_idx, end_idx, pad_idx]:
            continue
        words.append(idx2word.get(idx, '<UNK>'))
    
    return ' '.join(words)


def main():
    parser = argparse.ArgumentParser(description='Generate caption from two images')
    parser.add_argument('--imageA', type=str, required=True, help='Path to first image (before)')
    parser.add_argument('--imageB', type=str, required=True, help='Path to second image (after)')
    parser.add_argument('--vocab', type=str, default='./vocab.json', help='Path to vocab.json')
    parser.add_argument('--encoder_checkpoint', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--decoder_checkpoint', type=str, required=True, help='Path to decoder checkpoint')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size (1 for greedy)')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum caption length')
    parser.add_argument('--encoder_dim', type=int, default=2048, help='Encoder dimension')
    parser.add_argument('--network', type=str, default='resnet101', help='Encoder network')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = "cpu"
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}\n")
    
    # Load vocabulary
    print("Loading vocabulary...")
    with open(args.vocab, 'r') as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx)
    
    # Special tokens
    start_idx = word2idx['<START>']
    end_idx = word2idx['<END>']
    pad_idx = word2idx['<NULL>']
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens - START: {start_idx}, END: {end_idx}, PAD: {pad_idx}\n")
    
    # Load encoder
    print("Loading encoder...")
    encoder_checkpoint = torch.load(args.encoder_checkpoint, map_location=device)
    
    encoder = Encoder(args.network)
    encoder_trans = AttentiveEncoder(
        n_layers=3,
        feature_size=[16, 16, args.encoder_dim],
        heads=8,
        hidden_dim=512,
        attention_dim=2048,
        dropout=0.1,
    )
    
    encoder.load_state_dict(encoder_checkpoint['encoder_dict'])
    encoder_trans.load_state_dict(encoder_checkpoint['encoder_trans_dict'])
    
    encoder.eval()
    encoder_trans.eval()
    encoder.to(device)
    encoder_trans.to(device)
    print("Encoder loaded!\n")
    
    # Load decoder
    print("Loading decoder...")
    decoder_checkpoint = torch.load(args.decoder_checkpoint, map_location=device)
    
    model = LSTMCaptionGeneratorModel(
        vocab_size=vocab_size,
        encoder_dim=args.encoder_dim
    )
    model.load_state_dict(decoder_checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print("Decoder loaded!\n")
    
    # Load and preprocess images
    print(f"Loading images...")
    print(f"  Image A: {args.imageA}")
    print(f"  Image B: {args.imageB}")
    
    mean = np.array([100.6790, 99.5023, 84.9932])
    std = np.array([50.9820, 48.4838, 44.7057])
    
    imgA = load_and_preprocess_image(args.imageA, mean, std).to(device)
    imgB = load_and_preprocess_image(args.imageB, mean, std).to(device)
    print("Images loaded and preprocessed!\n")
    
    # Generate caption
    print("Generating caption...")
    
    if args.beam_size == 1:
        print("Using greedy decoding...")
        generated = greedy_decode(
            model, encoder, encoder_trans, imgA, imgB,
            start_idx, end_idx, args.max_len, device
        )
    else:
        print(f"Using beam search with beam size {args.beam_size}...")
        generated = beam_search_decode(
            model, encoder, encoder_trans, imgA, imgB,
            start_idx, end_idx, args.max_len, args.beam_size, device
        )
    
    # Convert to caption
    caption = tokens_to_caption(generated, idx2word, start_idx, end_idx, pad_idx)
    
    print("\n" + "="*70)
    print("GENERATED CAPTION:")
    print("="*70)
    print(caption)
    print("="*70)
    
    # Print token indices for debugging
    print(f"\nToken indices: {generated}")


if __name__ == "__main__":
    main()