# Image-Text Traffic Datasets

The A2D2-dataset - which is the main one - can be found here: https://www.a2d2.audi/a2d2/en/download.html
Make sure to only download the folder "Dataset-Semantic Segmentation"

In the folder "A2D2_preprocessed" the dataframes and further files can be found that were used to train this version of C4T.


Additional tests were conducted using the TJU-DHD dataset according to the corresponding scripts.

These notebooks generate captions for the corresponding images, based on their available object-classes (segmentation).


As we connected them via the Google Cloud, it will be necessary to give the notebooks access to the corresponding images (locally or via cloud).
Also, for utilizing the generated dataset, a storage is needed to access the image-text pairs later. 


The following datasets were also analyzed in this process and may be suitable: 

    1. Cityscapes (Cordts et al., 2016)
    2. Kitti/Kitti-360 (Geiger et al., 2012)
    3. Waymo OD (Sun et al., 2019)
    4. ApolloScape (Huang et al., 2019)