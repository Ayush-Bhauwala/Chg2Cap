This repository provides pretrained image encoders (from the `Chg2Cap` project) that can be reused for related tasks such as Visual Question Answering (VQA). This repository has been extended with additional code and scripts to support VQA tasks as part of a course project for Columbia University's "Deep Learning for Computer Vision" course (COMS 4995).

`data/LEVIR_CC/create_semantic_labels.py` contains the script for generating semantic labels for each image pair.

The semantic labels are stored in `semantic_labels.json`.

`data/LEVIR_CC/vqa_dataset.py` defines the dataset class for the VQA task.

The `model` directory contains the models used.

`vqa_train.py` is the script to train the model.

`vqa_test.py` is for testing.

`vqa_inference.py` is for performing inference for a single image pair and question.
