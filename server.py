import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template
from random import random
import torch

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')
features = torch.stack(features)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/vehicle" + str(random()) + "_" + file.filename 
        img.save(uploaded_img_path)

        query = fe.extract(img).unsqueeze_(1)
        score = torch.mm(features, query)
        score = score.squeeze(1).detach().cpu()
        score = score.numpy()
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1]
        
        ids = index[:10] # Top 30 results
        scores = [(score[idd], img_paths[idd]) for idd in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")
