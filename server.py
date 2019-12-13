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
from flask_uploads import UploadSet, configure_uploads, ALL,DATA
import scipy.io

app = Flask(__name__)

files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploaded'
configure_uploads(app, files)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for image_path in glob.glob("static/img/*"):
    #features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append(image_path)
features = torch.stack(pickle.load(open('static/feature/features.pkl', 'rb')))

"""
uncomment this to use previously exrtracted feature maps

result = scipy.io.loadmat('model/pytorch_result_VeRi.mat')
features = torch.FloatTensor(result['gallery_f'])
print("Gallery set size: ", features.shape)
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file_up = request.files['query_img']
        if not file_up:
            return render_template('index.html')
        res = []
        try:
            query_img_name=int(str(file_up).split('\'')[1].split('\\')[-1].split('_')[0])
        except:
            print("exception")
            query_img_name = random()
        
        for path in img_paths:
            if int(path.split('\\')[-1].split('_')[0]) == query_img_name:
                res.append(True)
            else:
                res.append(False)
        img = Image.open(file_up.stream)  # PIL image
        uploaded_img_path = "static/uploaded/vehicle" + str(random()) + "_" + file_up.filename 
        img.save(uploaded_img_path)
        
        query = fe.extract(img).unsqueeze_(1)
        score = torch.mm(features, query)
        score = score.squeeze(1).detach().cpu()
        score = score.numpy()
        
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1]
        ids = index[:10] # Top 30 results
        scores = [(round(score[idd],2), img_paths[idd], res[idd]) for idd in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")
