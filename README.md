# Graphical User Interface for Image Search Engine (in this case, for Vehicle Image Re-Identification system)

## User Interface
![User Interface](https://github.com/Rajat-Mehta/Vehicle-Re-identification-UI/blob/master/ui.png)

## Working
![Working](https://github.com/Rajat-Mehta/Vehicle-Re-identification-UI/blob/master/veri_ui.png)

## Overview
- This project is a simple content based image retrieval system for vehicle images using PyTorch + Flask. It uses the following two script files for extracting features from vehicle images and comparing different vehicle images based on those features.
- `offline.py`: This script file is used to extract part-level features from the vehicle images. We use a publicly available dataset VeRi-776 that consists of a Gallery set from which our model fetches the most similar images to the given query image. For your experimental purposes you can use your own vehicle dataset and if it's a non vehicle dataset then you might need to finetune our model before using it with your dataset. Refer to my this repository [Vehicle Re-Identification System](https://github.com/Rajat-Mehta/Vehicle_Reidentification) to learn how to train our part level feature extraction model on a new dataset.
- `server.py`: Once you have your extracted features, you can run this script to start a web-server. You can upload or drag-and-drop your query vehicle image to the server and our model will fetch the most similar images to the given query image using Cosine Similarity distance measure.
- The feature extraction is done using a Cluster-based Convolutional Baseline (CCB) model that uses the part-level features present in the vehicle images. Refer to the followwing GitHub repository to know more about our CCB model.

## Links

- [Vehicle Re-Identification Model](https://github.com/Rajat-Mehta/Vehicle_Reidentification)

## Usage
```bash
# Clone the code and install libraries
$ git clone https://github.com/Rajat-Mehta/Vehicle-Re-identification-UI.git
$ pip install -r requirements.txt

# Put your image files (*.jpg) on static/img

$ python offline.py
# Then part-level features are extracted using our Cluster Based Convolutional Baseline model and saved on static/feature
# Note that it takes time for the first time because of expensive feature extraction step

$ python server.py
# Now you can do the search via localhost:5000
```



