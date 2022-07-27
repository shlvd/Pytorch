import torch
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models

URL = "https://pytorch.tips/coffee"
FPATH = "coffee.jpg"

urllib.request.urlretrieve(URL, FPATH) # retrieve coffee.jpg to project folder

img = Image.open("coffee.jpg")

transform = transforms.Compose([ 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize( 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_tensor = transform(img)

#print(type(img_tensor), img_tensor.shape)

batch = img_tensor.unsqueeze(0)

model = models.alexnet(pretrained=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()

model.to(device)

y = model(batch.to(device))

y_max, index = torch.max(y, 1)

urllib.request.urlretrieve("https://pytorch.tips/imagenet-labels", "imagenet_class_labels.txt")

with open("imagenet_class_labels.txt") as f:
    classes = [line.strip() for line in f.readlines()]

prob = torch.nn.functional.softmax(y, dim=1)[0] * 100
_, indices = torch.sort(y, descending=True)
for idx in indices[0][:5]:
    print(classes[idx], prob[idx].item())