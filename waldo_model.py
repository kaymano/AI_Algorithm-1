import torch
import torch.nn as nn
import torch.nn.functional as F
NUM_OUTPUTS = 1
batch_size = 1
class WaldoModel(torch.nn.Module):

    def __init__(self):
        super(WaldoModel, self).__init__()
        self.history = []
        self.accuracy_map ={}
        self.accuracy = 0
        self.train_accuracy = 0


        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(25568, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 25568)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.softmax(x)
        return x
