import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms

train_transforms = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(
      mean=(0.4914, 0.4822, 0.4465),
      std=(0.2023, 0.1994, 0.2010)
  )
])

test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
      mean=(0.4914, 0.4822, 0.4465),
      std=(0.2023, 0.1994, 0.2010)
  )
])

train_data = CIFAR10(root='./train/', train=True, download=True, transform=train_transforms)
test_data = CIFAR10(root='./test/', train=False, download=True, transform=test_transforms)

train_set, val_set = random_split(train_data, [40000, 10000])

trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=16,
    shuffle=True
)

valloader = torch.utils.data.DataLoader(
    val_set,
    batch_size=16,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=16,
    shuffle=False
)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = LeNet5().to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

N_EPOCHS = 10

for epoch in range(N_EPOCHS):
    train_loss = 0.0
    model.train()
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    val_loss = 0.0
    model.eval()
    for inputs, labels in valloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
    
    num_correct = 0.0
    for inputs, labels in testloader:
        model.eval()
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        num_correct += (predicted == labels).float().sum()
    accuracy = num_correct / (len(testloader) * testloader.batch_size) * 100

    print(f"Epoch: {epoch} Train loss: {train_loss/len(trainloader)} Val loss:  {val_loss/len(valloader)} Test accuracy: {round(accuracy.item(), 2)} %")
