import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
from imageio import imread
import numpy as np

UNIVERSAL_QUESTIONS = {
    "building_added": [
        "Have new building appeared?",
        "Have buildings been added?",
        "Have buildings been constructed?",
        "Has some construction taken place?",
        "Were new buildings built?",
    ],
    "building_removed": [
        "Have buildings been demolished?",
        "Have buildings been removed?",
        "Have some buildings disappeared?",
        "Have some building been torn down?",
        "Were any buildings destroyed?",
    ],
    "road_added": [
        "Has a new road been constructed?",
        "Has a road been added?",
        "Is there a new lane in the image?",
        "Has any road been built?",
        "Was a lane constructed?",
    ],
    "vegetation_added": [
        "Have trees been grown?",
        "Have plants been added?",
        "Is there new vegetation in the image?",
        "Has any greenery appeared?",
        "Were new plants grown?",
    ],
    "vegetation_removed": [
        "Has vegetation been removed?",
        "Have trees been cut down?",
        "Has any greenery disappeared?",
        "Were plants removed?",
        "Has some vegetation been cleared?",
    ],
    "no_change": [
        "Is there no significant change?",
        "Has the scene remained the same?",
        "Are there no noticeable differences?",
        "Is the area unchanged?",
        "Has nothing changed?",
    ],
}


class LevirCCVQADataset(Dataset):
    def __init__(self, image_dir, semantic_labels_file, split, transform=None):
        self.items = []
        self.image_dir = image_dir
        self.split = split
        self.mean = [100.6790, 99.5023, 84.9932]
        self.std = [50.9820, 48.4838, 44.7057]
        with open(semantic_labels_file, "r") as f:
            semantic_labels = json.load(f)

        for file_name, labels in semantic_labels.items():
            image_path = os.path.join(image_dir, split, "A", file_name)
            if not os.path.exists(image_path):
                continue
            yes_questions = []
            no_questions = []
            for label, is_present in labels.items():
                if( label not in UNIVERSAL_QUESTIONS):
                    continue
                questions = UNIVERSAL_QUESTIONS[label]
                if is_present:
                    yes_questions.extend(questions)
                else:
                    no_questions.extend(questions)
            if not yes_questions and not no_questions:
                continue
            self.items.append(
                {
                    "file_name": file_name,
                    "yes_questions": yes_questions,
                    "no_questions": no_questions,
                }
            )

        print(
            f"Dataset initialized for split '{split}'. Loaded {len(self.items)} images."
        )

    def __len__(self):
        return 4 * len(self.items)

    def __getitem__(self, idx):
        image_index = idx // 4
        yes_bucket = (idx % 4) < 2
        item = self.items[image_index]
        file_name = item["file_name"]
        question = None
        ans = None
        if yes_bucket:
            if not item["yes_questions"]:
                # If there are no yes questions, force no question
                question = random.choice(item["no_questions"])
                ans = 0.0
            else:
                question = random.choice(item["yes_questions"])
                ans = 1.0
        else:
            if not item["no_questions"]:
                # If there are no no questions, force yes question
                question = random.choice(item["yes_questions"])
                ans = 1.0
            else:
                question = random.choice(item["no_questions"])
                ans = 0.0
        img_fileA = os.path.join(self.image_dir, self.split, "A", file_name)
        img_fileB = os.path.join(self.image_dir, self.split, "B", file_name)
        imgA = imread(img_fileA).astype(np.float32)
        imgB = imread(img_fileB).astype(np.float32)

        imgA = np.moveaxis(imgA, -1, 0)
        imgB = np.moveaxis(imgB, -1, 0)

        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        return (
            torch.from_numpy(imgA),
            torch.from_numpy(imgB),
            question,
            torch.tensor([ans], dtype=torch.float32),
        )


if __name__ == "__main__":
    dataset = LevirCCVQADataset(
        image_dir="/home/ab6106/Levir-CC-dataset/images",
        semantic_labels_file="./data/LEVIR_CC/semantic_labels.json",
        split="train",
    )
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        imgA, imgB, question, ans = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image A shape: {imgA.shape}")
        print(f"  Image B shape: {imgB.shape}")
        print(f"  Question: {question}")
        print(f"  Answer: {ans.item()}")
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch in dataloader:
        imgA_batch, imgB_batch, questions_batch, ans_batch = batch
        print(f"Batch Image A shape: {imgA_batch.shape}")
        print(f"Batch Image B shape: {imgB_batch.shape}")
        print(f"Batch Questions: {questions_batch}")
        print(f"Batch Answers: {ans_batch}")
        break