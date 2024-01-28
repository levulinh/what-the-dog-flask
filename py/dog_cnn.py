import copy
import os
import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from flask import Flask
from flask_restful import Api, Resource, reqparse
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

CLASS_NUM = 120
int2label = torch.load("./variables/int2label.pt")
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available.  Training on CPU ...")
else:
    print("CUDA is available!  Training on GPU ...")

device = "cuda" if train_on_gpu else "cpu"

# Read images into train and test datasets


def transform(img):
    compose = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img_transformed = compose(img)
    return img_transformed


root_dir = "train/"


def load_img(file):
    img = Image.open(root_dir + file)
    img.load()
    return img


net = models.resnet34()

# Freeze training for all "features" layers
for param in net.layer1.parameters():
    param.requires_grad = False
for param in net.layer2.parameters():
    param.requires_grad = False
for param in net.layer3.parameters():
    param.requires_grad = False
for param in net.layer4.parameters():
    param.requires_grad = False

n_inputs = net.fc.in_features

# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
fc1 = nn.Linear(n_inputs, CLASS_NUM)
net.fc = fc1

# if GPU is available, move the model to GPU
if train_on_gpu:
    net.cuda()

net.load_state_dict(torch.load("./states/dog_breed.pth", map_location=device))
net.eval()


def load_image_from_url(url):
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Read the content of the response
        image_content = response.content

        # Create a BytesIO object from the image content
        image_bytes = BytesIO(image_content)

        # Open the image using Pillow
        image = Image.open(image_bytes)

        return image
    else:
        print(f"Failed to load image from URL. Status code: {response.status_code}")
        return None


# @title Enter url to predict the breed of the dog
# @markdown URL to the dog image file
root_dir = "./"
# @param {type: "string"}
url = "https://www.pitpat.com/wp-content/uploads/2020/06/PP_German-Shepherd-1536x1063.jpg"


def predict(url):
    test_img = transform(load_image_from_url(url))
    test_input = test_img.unsqueeze(0).to(device)
    test_output = net(test_input)
    soft_output = nn.functional.softmax(test_output.squeeze(), dim=0)
    values, indices = torch.topk(soft_output, 3)
    top3 = [
        {"label": int2label[indice.item()], "confidence": values[ii].detach().cpu().numpy() * 100}
        for ii, indice in enumerate(indices)
    ]
    return top3


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("url")


class DogClf(Resource):
    def post(self):
        args = parser.parse_args()
        url = args["url"]
        top3 = predict(url)
        print(top3)
        return {"result": top3}, 200


# api.add_resource(DogClf, '/predict')

# if __name__ == '__main__':
#     app.run(debug=True, port=9999)
