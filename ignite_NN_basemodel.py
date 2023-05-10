import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

ImageFile.LOAD_TRUNCATED_IMAGES = True
INPUT_DIM = 350 * 350 * 3
OUTPUT_DIM = 2

paths = ['data/test', 'data/train', 'data/valid']
transform = T.Compose([
                # T.Grayscale(), # uncomment to use color
                T.PILToTensor()
            ])

test_set = ImageFolder(root=paths[0], transform=transform)
train_set = ImageFolder(root=paths[1], transform=transform)
validation_set = ImageFolder(root=paths[2], transform=transform)



# The layout of the base model
class baseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(baseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5000)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5000, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Function that trains the baseModel
def train_model():
    model = (baseModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)).to(device)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size = 1)

    print("Started Training")
    
    for epoch in range(20):

        for i, (images, labels) in enumerate(train_loader):
            
            images = ((images.view(-1, 350*350*3).to(torch.float32)).requires_grad_()).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels.to(device))

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # Used to keep track of progress
            if i % 100 == 0:
               print(i)

        # evaluating    
        correct = 0
        total = 0
        print("evaluating")
        for images, labels in val_loader:

            images = ((images.view(-1, 350*350*3).to(torch.float32)).requires_grad_()).to(device)
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            correct += (predicted == labels.to(device)).sum()

        accuracy = 100 * correct / total

        # Print Loss
        print('Iteration: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))

    correct = 0
    total = 0
    for images, labels in test_loader:
       
       images = ((images.view(-1, 350*350).to(torch.float32)).requires_grad_()).to(device)
       outputs = model(images)

       _, predicted = torch.max(outputs.data, 1)

       total += labels.size(0)

       correct += (predicted == labels.to(device)).sum()

    accuracy = 100 * correct / total

    print('Test Accuracy: {}'.format(accuracy))



def main():
    train_model()


if __name__ == "__main__":
   main()