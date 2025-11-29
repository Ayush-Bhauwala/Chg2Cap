import os
import torch
import numpy as np
from imageio import imread
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


import os
import torch
import numpy as np
from imageio import imread
import json
from torch.utils.data import Dataset, DataLoader


class ChangeDetectionCaptionDataset(Dataset):
    """
    Dataset for change detection image pairs with multiple captions.
    Sequences are padded to fixed max_seq_len in __getitem__.
    """
    
    def __init__(self, image_dir, captions_file, vocab_file, split='train', 
                 max_seq_len=50, sample_one_caption=True):
        """
        Args:
            image_dir: Root directory containing images (with A/ and B/ subdirs)
            captions_file: Path to JSON file with captions
            vocab_file: Path to vocab.json file
            split: 'train', 'val', or 'test'
            max_seq_len: Fixed maximum sequence length (sequences padded to this)
            sample_one_caption: If True, randomly sample one caption per image pair
        """
        self.image_dir = os.path.join(image_dir, split)
        self.split = split
        self.max_seq_len = max_seq_len
        self.sample_one_caption = sample_one_caption
        
        # Load vocabulary from JSON file
        with open(vocab_file, 'r') as f:
            self.word2idx = json.load(f)
        
        # Create reverse mapping
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        # Special token indices
        self.pad_idx = self.word2idx['<NULL>']    # 0
        self.unk_idx = self.word2idx['<UNK>']     # 1
        self.start_idx = self.word2idx['<START>'] # 2
        self.end_idx = self.word2idx['<END>']     # 3
        
        # Image normalization parameters
        self.mean = np.array([100.6790, 99.5023, 84.9932])
        self.std = np.array([50.9820, 48.4838, 44.7057])
        
        # Load captions
        with open(captions_file, 'r') as f:
            data = json.load(f)
        
        # Filter images by split
        self.samples = []
        for img_info in data['images']:
            if img_info['split'] == split:
                if sample_one_caption:
                    self.samples.append({
                        'filename': img_info['filename'],
                        'imgid': img_info['imgid'],
                        'sentences': img_info['sentences']
                    })
                else:
                    for sent in img_info['sentences']:
                        self.samples.append({
                            'filename': img_info['filename'],
                            'imgid': img_info['imgid'],
                            'sentence': sent
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image."""
        img = imread(img_path)
        img = np.asarray(img, dtype=np.float32)
        
        # Convert HWC to CHW
        img = np.moveaxis(img, -1, 0)
        
        # Normalize
        for i in range(3):
            img[i] = (img[i] - self.mean[i]) / self.std[i]
        
        return torch.from_numpy(img)
    
    def pad_sequence(self, sequence, max_len, pad_value=0):
        """
        Pad sequence to fixed length.
        
        Args:
            sequence: List of token indices
            max_len: Maximum length to pad to
            pad_value: Value to use for padding (default: 0 for <NULL>)
        
        Returns:
            Padded tensor of shape (max_len,)
        """
        seq_len = len(sequence)
        
        if seq_len >= max_len:
            # Truncate if too long
            return torch.tensor(sequence[:max_len], dtype=torch.long)
        else:
            # Pad if too short
            padded = sequence + [pad_value] * (max_len - seq_len)
            return torch.tensor(padded, dtype=torch.long)
    
    def tokenize_caption(self, tokens):
        """
        Convert tokens to indices and create input/output sequences with padding.
        Both sequences are padded to max_seq_len.
        """
        # Convert tokens to indices
        indices = [self.word2idx.get(token, self.unk_idx) for token in tokens]
        
        # Add special tokens: <START> at beginning, <END> at end
        sequence = [self.start_idx] + indices + [self.end_idx]
        
        # Input sequence: <START> + tokens (for decoder input)
        input_seq = sequence[:-1]  # Remove <END>
        
        # Output sequence: tokens + <END> (target for prediction)
        output_seq = sequence[1:]  # Remove <START>
        
        # Pad both sequences to max_seq_len
        input_seq_padded = self.pad_sequence(input_seq, self.max_seq_len, self.pad_idx)
        output_seq_padded = self.pad_sequence(output_seq, self.max_seq_len, self.pad_idx)
        
        # Calculate actual length (before padding) for masking
        actual_length = min(len(output_seq), self.max_seq_len)
        
        return input_seq_padded, output_seq_padded, actual_length
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        img_filename = sample['filename']
        img_path_A = os.path.join(self.image_dir, 'A', img_filename)
        img_path_B = os.path.join(self.image_dir, 'B', img_filename)
        
        imgA = self.load_and_preprocess_image(img_path_A)
        imgB = self.load_and_preprocess_image(img_path_B)
        
        # Select caption
        if self.sample_one_caption:
            sentence = np.random.choice(sample['sentences'])
        else:
            sentence = sample['sentence']
        
        # Tokenize and pad caption
        tokens = sentence['tokens']
        input_seq, output_seq, length = self.tokenize_caption(tokens)
        
        return imgA, imgB, input_seq, output_seq, length


# def collate_fn(batch):
#     """
#     Custom collate function to handle variable-length sequences.
#     Pads sequences to the maximum length in the batch.
#     """
#     imgA_list, imgB_list, input_seq_list, output_seq_list = zip(*batch)
    
#     # Stack images
#     imgA_batch = torch.stack(imgA_list, dim=0)
#     imgB_batch = torch.stack(imgB_list, dim=0)
    
#     # Get sequence lengths
#     lengths = torch.tensor([len(seq) for seq in input_seq_list])
    
#     # Pad sequences with <NULL> (index 0)
#     input_seqs = pad_sequence(input_seq_list, batch_first=True, padding_value=0)
#     output_seqs = pad_sequence(output_seq_list, batch_first=True, padding_value=0)
    
#     return imgA_batch, imgB_batch, input_seqs, output_seqs, lengths


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = ChangeDetectionCaptionDataset(
        image_dir='/Users/ayushbhauwala/Documents/Columbia/Sem 1/DL for CV/project/experimenting/levir_cc/Levir-CC-dataset/images',
        captions_file='/Users/ayushbhauwala/Documents/Columbia/Sem 1/DL for CV/project/experimenting/levir_cc/Levir-CC-dataset/LevirCCcaptions.json',
        vocab_file='./data/LEVIR_CC/vocab.json',
        split='train',
        max_seq_len=50,
        sample_one_caption=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.word2idx)}")
    print(f"Special tokens - PAD: {dataset.pad_idx}, UNK: {dataset.unk_idx}, START: {dataset.start_idx}, END: {dataset.end_idx}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        # collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Test iteration
    for imgA, imgB, input_seqs, output_seqs, lengths in dataloader:
        print(f"\nBatch shapes:")
        print(f"  Image A: {imgA.shape}")
        print(f"  Image B: {imgB.shape}")
        print(f"  Input sequences: {input_seqs.shape}")
        print(f"  Output sequences: {output_seqs.shape}")
        print(f"  Lengths: {lengths}")
        print(f"\nExample sequence:")
        print(f"  Input: {input_seqs[0]}")
        print(f"  Output: {output_seqs[0]}")
        break
