Data Exploration
================

What is the train/test split?
What does the coverage look like for each chip?
What does each image look like? How are the different channels, bands represented in the image?
How does each input pixel map to the AGBM ground truth value?
How are the images split between the satellites (S1 and S2)?


Steps
=====

Build the inference endpoint

Support both single image and batch inference.
How to detect single v. batch?

Write the submission file format
Save predictions to TIFF image format
Build a data pipeline
Build a model pipeline

Use pydantic for managing typing and configuration settings
Use PyTorch Lightning for model development
Understand batch inference with PyTorch DataLoaders
