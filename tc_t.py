#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from _0_preprocessing import train_loader,val_loader,test_loader
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters   #we use Adam
num_epochs = 3
batch_size = 5000
learning_rate = 0.01 

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
i=1
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()                         #   depth        size
        #       in_chan, out_chan kernel_size
        self.pool = nn.MaxPool2d(2, 2)    

        self.conv1 = nn.Conv2d(3, 32, 5,stride=1,padding=2)     #   3 -> 16     32 -> 32    # floor[(32+2*1-3)/1] + 1 = 32
        self.bn1   = nn.BatchNorm2d(32)
                              #                32 -> 16
        self.conv2 = nn.Conv2d(32, 64, 5,stride=1,padding=0)    #   16 -> 32    16 -> 16
        self.bn2   = nn.BatchNorm2d(64)                 #                16 -> 8 
        self.conv3 = nn.Conv2d(64, 128, 5,stride=1,padding=0)    #  32 -> 64      8  -> 8
        self.bn3   = nn.BatchNorm2d(128) 
        
        #   self.pool = nn.MaxPool2d(2, 2)                      #                8  -> 4
        self.leaky_relu = nn.ReLU()
        self.linear_input = 128*1*1
        self.fc1 = nn.Linear(128, 10)
    def forward(self, x):
        #                                        m, 3, 32, 32
        x = self.conv1(x)
        # print(torch.mean(x,axis = [0,2,3],keepdim=True)[0][0:5].cpu().reshape(5))
        # print(torch.min(x),torch.max(x))
        global ss,i
        if i ==1:
            ss = self.conv1.state_dict()
            print(ss)
            i=2
        # x = self.leaky_relu(self.bn1(x))    #  m, 6, 28, 28 
        x = self.leaky_relu(x)  
        x = self.pool(x)                      #  m, 6, 14, 14


        x = self.conv2(x)
        # x = self.leaky_relu(self.bn2(x))    #  m, 16, 10, 10
        x = self.leaky_relu(x) 
        x = self.pool(x)                      #  m, 16, 5, 5

        x = self.conv3(x)
        # x = self.leaky_relu(self.bn3(x))    #  m, 16, 10, 10
        x = self.leaky_relu(x) 
        x = self.pool(x)                      #  m, 16, 5, 5
        
        x = x.view(-1, self.linear_input)            # -> n, 512
        x = self.fc1(x)                       # -> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,weight_decay=0)
n_total_steps = len(train_loader)


import time 
start = time.time()
loss_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            loss_list.append(loss.item())
end = time.time()

print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

fig,ax=plt.subplots(1,1,figsize=(8,8))
plt.ylim([0, 1])
plot_y = torch.tensor(n_class_correct)/torch.tensor(n_class_samples)
plot_x = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
ax.plot(plot_x ,plot_y)


print("training time",format(end-start))

fig,ax=plt.subplots(1,1,figsize=(8,8))
plot_y = torch.tensor(loss_list)
plot_x = torch.arange(len(loss_list))
plt.ylim([0, 3])
ax.plot(plot_x ,plot_y)


for key in model.state_dict():
    md = model.state_dict()
    if "bn1" in key:
        print(key,md[key].cpu().tolist())

#18
# %%
