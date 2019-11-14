import glob
import os
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor

fe = FeatureExtractor()

for img_path in sorted(glob.glob('static/img/*.jpg')):
    print(img_path)
    img = Image.open(img_path)  # PIL image
    feature = fe.extract(img)
    feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
    pickle.dump(feature, open(feature_path, 'wb'))

"""
f=[]
i=1
for img_path in sorted(glob.glob('static/img/*.jpg')):
    
    img = Image.open(img_path)  # PIL image
    feature = fe.extract(img)
    #feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
    f.append(feature)
    if i %100 ==0:
        print(i)
        break
    i+=1
print(len(f))
pickle.dump(f, open('static/feature/f.pkl', 'wb'))
"""