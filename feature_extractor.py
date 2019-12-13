import numpy as np
import torch
import torchvision
import os
from model import PCB, PCB_test
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable

class FeatureExtractor:
    def __init__(self):
        #base_model = VGG16(weights='imagenet')
        #self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        save_path = os.path.join('./model', 'net.pth')
        model = PCB(575)
        self.model = PCB_test(model, num_parts=6, cluster_plots=False)
        self.model.load_state_dict(torch.load(save_path), strict=False)
        self.avgpool =  nn.AdaptiveAvgPool2d((3, 2))
        self.model = self.model.eval()

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        data_transforms = transforms.Compose([
                transforms.Resize((384, 192), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        x = data_transforms(img).unsqueeze_(0)
        feature = torch.FloatTensor(1, 12288).zero_() # we have six parts
        for i in range(2):
            if(i==1):
                x = self.fliplr(x)
            x = Variable(x)
            #if opt.fp16:
            #    input_img = input_img.half()
            outputs = self.model(x)
            outputs = outputs.view(x.size(0), -1)
            f = outputs.data.cpu().float()
            feature = feature+f

        fnorm = torch.norm(feature, p=2, dim=1, keepdim=True) * np.sqrt(6) 
        feature = feature.div(fnorm.expand_as(feature))
        feature = feature.view(feature.size(0), -1).squeeze()
        return feature

    def fliplr(self,img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def forward(self, x):
        x = self.model.model.conv1(x)
        x = self.model.model.bn1(x)
        x = self.model.model.relu(x)
        x = self.model.model.maxpool(x)
        x = self.model.model.layer1(x)
        x = self.model.model.layer2(x)
        x = self.model.model.layer3(x)
        x = self.model.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), -1)
        return y