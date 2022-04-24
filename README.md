# CLIP4Traffic (C4T)
CLIP4Traffic (C4T) is an image-retrieval/-search system (CBIR-system) designed to efficiently filter high-resolution traffic-data. Given a (user) text-query, the images with the highest similarity are retrieved by the system. 

Based on CLIP by OpenAI, we train two encoders (image & text) separetely and project their corresponding feature-vectors (embeddings) into a joint dimension. 
The original CLIP model tough shows severe weaknesses for such specific use-cases (e.g. multiple objects, distances etc.), therefore we build and train our own architecture. 

# Dataset
Most benchmark models utilizing language supervision (e.g. CLIP, ConVIRT, VirTEX) are trained on an immense amount of data, however these are mostly crowd-labeled web-images with low-resolution and a bad quality of captions. CLIP uses image-classification/-detection to generate captions based on the represented object(s). 

For ADAS-development, such captions don't contain enough information. Therefore, we generate our own image-text datasets containing high-resolution traffic-scenarios and automatically generated captions based on a variety of object-classes, their quantities and distances to eachother. 

# Evaluation
Performance evaluation for modern CBIR-systems are rare or poorly designed as there is no clear true or false for a given output (True-Positive, False-Positive). Calculated by a given similarity metric, the highest matches are retrieved without a classification. This makes it very complex to evaluate a CBIR-system or compare different ones. 

We leverage the generated dataset and use the corresponding caption as ground-truth. We querying in a standardized way, we can compare the caption of the retrieved images with the given query and therefore calculate the Precision and Recall. Additionally, we process the same with the original CLIP model and show the significant performance-gain of C4T for the use-case of image-retrieval for traffic-scenarios. 


...Further description tbc...
