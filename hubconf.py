import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

transform_tensor_to_pil = ToPILImage()
transform_pil_to_tensor = ToTensor()

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot",]


def mithlesh():
  print ('Mithlesh')
         
         
class cs21m010NN(nn.Module):
  def __init__(self):
        super(cs21m010NN, self).__init__()
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

      
def get_model(train_data_loader=None, n_epochs=10):
  model = cs21m010NN().to(device)

  training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
  
  test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
  batch_size=128
  train_dataloader = DataLoader(training_data, batch_size=batch_size)
  test_dataloader = DataLoader(test_data, batch_size=batch_size)

 # for X, y in test_dataloader:
 #       print(f"Shape of X [N, C, H, W]: {X.shape}")
 #       print(f"Shape of y: {y.shape} {y.dtype}")
 #       break


  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(get_model().parameters(), lr=1e-3)


  def _train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
  def _test(dataloader, model, loss_fn):
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
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
  def train(train_dataloader, test_dataloader, model1, loss_fn1, optimizer1, epochs=5):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train(train_dataloader, model1, loss_fn1, optimizer1)
        _test(test_dataloader, model1, loss_fn1)

  print ('Returning model... (rollnumber: cs21m010)')       

  return model


