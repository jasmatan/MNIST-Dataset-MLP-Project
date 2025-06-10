#Step 1: imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report


#We are using the CPU for everything, because CUDA is being annoying.
device = "cpu"

#Create variables for the training and test MNIST data, respectively.
#Datasets are pulled from torchvision, and downloaded to the "data" folder.
training_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

#Initialize the dataloaders. This is some kind of torch built-in to process
#the datasets.

#the batch_size variable controls how large the "batches" of features and
#labels that are pulled at once are. I don't know why they chose 64.
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#define device to use for training
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#print(f"Using {device} device")

#Define the actual model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
#use our class to assemble the model
model = NeuralNetwork().to(device)

#define a loss function (cross entropy loss) and an optimizer (stochastic gradient descent)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.1)


#Defining our training function. We use the training dataset for this.

def train(dataloader, model, loss_fn, optimizer):
    print("Training start!")
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print("Training end!")

#Defining our testing function. 
def test(dataloader, model, loss_fn):

    #Initialize an array which will record our model's predictions and the actual values, per class.
    #This will be used for our data analysis later.
    predictions = []
    test_values = []

    print("Testing start!")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            predictions.extend(pred.argmax(1).cpu().numpy())  # Shape [batch_size]
            test_values.extend(y.cpu().numpy())  # Already shape [batch_size]

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("Testing end!")

    #return the loss (this will help us plot it later), as well as the confusion matrix
    return(test_loss, np.array(test_values), np.array(predictions))


def print_metrics(test_values, predictions):
    # 1. Accuracy
    test_values = np.array(test_values).flatten()
    predictions = np.array(predictions).flatten()
    
    # 2-4. Per-class statistics
    print("\nClassification Report:")
    print(classification_report(test_values, predictions))
    
    cm = confusion_matrix(test_values, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    accuracy = accuracy_score(test_values, predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f}")


#This part actually calls the functions we've defined.
def run_nn(epochs):
    indexed_losses = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        testing_data = test(test_dataloader, model, loss_fn)
        indexed_losses.append(testing_data[0])
        print_metrics(testing_data[1], testing_data[2])

    
    #this is all just stuff to plot our results
    fig, ax = plt.subplots()             # Create a figure containing a single Axes.
    ax.plot(range(epochs), indexed_losses)  # Plot some data on the Axes.
    ax.set_title("Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (test)")
    ax.set_xbound(0, epochs - 1)
    ax.set_xticks(range(epochs))
    plt.show()
    print("Done!")

run_nn(10)