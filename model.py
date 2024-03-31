
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models

# Define transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Resize((224, 224)),  # Resize images to 224x224,
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images with mean and standard deviation
])

# Load Fashion MNIST dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Subsample
subset_indices = torch.randperm(len(trainset))[:len(trainset)//4]
trainset_subsampled = torch.utils.data.Subset(trainset, subset_indices)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset_subsampled, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

len(trainset_subsampled),len(testset)

len(trainloader)

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm
# Training the model with different optimizers
num_epochs = 5
results = {}
optimizers_list = ["adam","adagrad", "adadelta"]
device = "cuda"
for optimizer_name in optimizers_list :

    # Load pre-trained ResNet101 model
    resnet18 = models.resnet18(pretrained=True)

    # Freeze all layers except the last one
    for param in resnet18.parameters():
      resnet18.requires_grad = False

    # Modify the last layer to match the number of classes in Fashion MNIST (10)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)

    resnet18.fc.weight.requires_grad = True
    resnet18.fc.bias.requires_grad = True

    resnet18 = resnet18.to(device)

    print(f"Training with {optimizer_name} optimizer...")
    if optimizer_name == "adam":
        optimizer = optim.Adam(resnet18.parameters())
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(resnet18.parameters())
    elif optimizer_name == "adadelta":
        optimizer = optim.Adadelta(resnet18.parameters())

    resnet18.train()  # Set model to training mode
    train_losses = []
    train_accuracy = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} : ")
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        train_losses.append(running_loss / len(trainloader))
        train_accuracy.append(100 * correct / total)

        # Evaluate model on test set
        resnet18.eval()  # Set model to evaluation mode
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = resnet18(images)
                _, predicted = torch.topk(outputs, 5, dim=1)
                total += labels.size(0)
                for i in range(len(labels)):
                    if labels[i] in predicted[i]:
                        correct_top5 += 1

        top5_accuracy = 100 * correct_top5 / total
        print(f"Top-5 Test Accuracy: {top5_accuracy:.2f}%")


    results[optimizer_name] = {"loss": train_losses, "accuracy": train_accuracy}

# Plotting curves for training loss and training accuracy
plt.figure(figsize=(10, 5))
for optimizer_name, result in results.items():
    plt.plot(result['loss'], label=f"{optimizer_name} Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for optimizer_name, result in results.items():
    plt.plot(result['accuracy'], label=f"{optimizer_name} Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Curves')
plt.legend()
plt.show()

