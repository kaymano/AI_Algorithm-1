from torch import FloatTensor
import torch.optim as optim
from waldo_model import WaldoModel
import torch.nn as nn
import torch
import os
from PIL import Image
import numpy as np

np.random.seed(1)
torch.manual_seed(3)

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
    return data

for epoch in range(3):  # loop over the dataset 3 times

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
            if i % 25 == 0 or label == [1,0]:    # print every 2000 mini-batches
                print(f'Epoch: {epoch}')
                print(f'Image num: {im_num}')
                print(f'Grid num: {i}')
                running_loss = 0.0
                print(labelt,outputs)

print('Finished Training')

data = load_data('images/test','6.png')
max_grid_chance = 0
chances = []
for i,data in enumerate(data):
    with torch.no_grad():
        image_name, image = data
        out = net(image[None,...])
        out = out[0]
        chances.append((image_name,out[1])) # should be out[0]
chances.sort(key= lambda x:x[1])
print(chances)

# Find way to evaluate accuracy of rankings