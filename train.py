from torch import FloatTensor
import torch.optim as optim
from waldo_model import WaldoModel
import torch.nn as nn
import torch
import os
from PIL import Image
import numpy as np
import random

np.random.seed(1)
torch.manual_seed(3)
seeded_random = random.Random(0)
net = WaldoModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=.001)
file_list = [f for f in os.listdir('images/sub') if os.path.isfile(os.path.join('images',f))]

def load_data(path,image_name):
    data = []
    for simage_name in os.listdir(f'{path}/{image_name}'):
        x = Image.open(f'{path}/{image_name}/{simage_name}')
        x = np.asarray(x)
        x = np.moveaxis(x, -1, 0)
        x = torch.FloatTensor(x)
        data.append((simage_name,x))
    seeded_random.shuffle(data)
    return data

for epoch in range(5):  # loop over the dataset 3 times

    running_loss = 0.0
    for im_num, image_name in enumerate(file_list):
        for i, data in enumerate(load_data('images/sub',image_name)):
            # get the inputs; data is a list of [inputs, labels]
            image_name, image = data
            label = [int(image_name[0] == "W"),int(image_name[0] != "W")]
            labelt = torch.FloatTensor(label)[None,...]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(image[None,...])
            loss = criterion(outputs, labelt)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if label == [1,0]:  
                print(f'Epoch: {epoch}')
                print(f'Image num: {im_num}')
                print(f'Grid num: {i}')
                running_loss = 0.0
                print(labelt,outputs)

print('Finished Training')

data = load_data('images/test','9.png')
max_grid_chance = 0
chances = []
for i,data in enumerate(data):
    with torch.no_grad():
        image_name, image = data
        out = net(image[None,...])
        out = out[0]
        chances.append((image_name,out[0])) # should be out[0]
chances.sort(key= lambda x:-x[1])
print(chances)
# Find way to evaluate accuracy of rankings
current_rank = 0
diff = 0
n = len(chances)
for i,pair in enumerate(chances):
    rank = i
    name,_ = pair
    if name[0] == 'W':
        diff += (rank - current_rank)
        current_rank += 1
        print("Rank:",rank)
accuracy = (diff/current_rank) / n
print('Accuracy:', accuracy)

    