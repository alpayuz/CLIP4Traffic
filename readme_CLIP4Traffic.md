# Introduction & Motivation:

Based on CLIP by OpenAI, we will build our own Encoder-Encoder architecture ("CLIP4Traffic" (C4T)) image-retrieval/-search (CBIR-system). Based on a given user-query in the form of keywords or sentences, the images with the highest (cosine) similarity are retrieved. 
For this, we train/fine-tune 2 encoders (DistilBERT for text and ResNet-50 for images) with our generated dataset (A2D2) consisting of image-text pairs.
Using ProjectionHeads, we can project both embeddings (feature-vectors) into a common dimension and convert them into same-size vectors.

# CLIP4Traffic

This repository contains the the developed codes to train and run C4T as well as the used dataset(s).
The main file also enables the image-retrieval/-search for a given database and measure the performances.

The original CLIP model is integrated and leveraged as benchmark.
It's possible to only use the encoders (image or text) of both models and work with the corresponding embeddings.
The "models" folder contains the trained models as well as the final and best-performing version of C4T that was describes in the corresponding paper.

# Usage

1. Training a new C4T-model
If you want to train a new C4T-model based on your parameters, you can use the generated datasets in the "Data"-folder (e.g. A2D2).
With these image-text pairs you can then train your C4T inside the "CLIP4Traffic"-folder with the C4T_main.py script. Alternatively, you can also work with your own dataset. 

2. Using the trained C4T-model for image-retrieval
The C4T_main.py script also enables to use an already trained C4T (on the A2D2-data) to process given images and texts and generate corresponding embeddings. 
It then can be used for image-retrieval tasks of all kind. Furthermore, it enables to do the same with the original CLIP model and therefore compare their results.

You can do this by starting in the C4T_main.py file at "Inference". 

# Evaluation

After training, we will evaluate the performance of C4T and compare it to CLIP.
For this, we will use an unknown validation-set and measure the Precision and Recall.


# Dependencies

The necessary packages and modules will be inside each folders requirements.txt 
