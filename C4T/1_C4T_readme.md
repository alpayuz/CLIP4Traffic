# General
The CLIP4Traffic-model builds an own image-text encoder architecture and projects their embeddings into a joint dimension.
Afterwards, we compare these vectors using the cosine similarity, to retrieve the matching images based on a given user-query. 

1. ##Training a C4T-model
If you aim to train a new C4T-model based on your parameters, you can run the entire "C4T_main.py" and use a prepared image-text dataset (e.g. A2D2) from the "Data"-folder. 
After training the model, you can then start embedding the data, querying and retrieving which is also available for the original CLIP model. 

2. ##Embedding data
By using a pretrained C4T and/or the integrated CLIP-model, you can embedd given data in the "Inference" section. 

3. ##Image-retrieval
At the end, the find_matches() function enables to load a model (C4T or CLIP) and query with text to retrieve images out of the given "validation-set" (database)
If needed, the "Evaluation" using the Precision-Recall where one can directly compare the performances of C4T against CLIP is done at the very end. 