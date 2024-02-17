from torch.utils.data import DataLoader
from tqdm import tqdm
# from plotly_utils import line
from typing import Optional
import time

#custom
from engine.tensor import Tensor
from nn.core.module import Module
from nn.linear import Linear
from nn.relu import ReLU
from nn.optim.sgd import SGD
from nn.functional.cross_entropy import cross_entropy
from nn.no_grad import NoGrad
from utils import get_mnist, visualize

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 64)
        self.linear2 = Linear(64, 64)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.output = Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape((x.shape[0], 28 * 28))
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.output(x)
        return x

def train(model: MLP, train_loader: DataLoader, optimizer: SGD, epoch: int, train_loss_list: Optional[list] = None):
    print(f"Epoch: {epoch}")
    progress_bar = tqdm(enumerate(train_loader))
    for (batch_idx, (data, target)) in progress_bar:
        data = Tensor(data.numpy())
        target = Tensor(target.numpy())
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, target).sum() / len(output)
        loss.backward()
        progress_bar.set_description(f"Train set: Avg loss: {loss.item():.3f}")
        optimizer.step()
        if train_loss_list is not None: train_loss_list.append(loss.item())


def test(model: MLP, test_loader: DataLoader, test_loss_list: Optional[list] = None):
    test_loss = 0
    correct = 0
    with NoGrad():
        for (data, target) in test_loader:
            data = Tensor(data.numpy())
            target = Tensor(target.numpy())
            output: Tensor = model(data)
            test_loss += cross_entropy(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"Test set:  Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({correct / len(test_loader.dataset):.1%})")
    if test_loss_list is not None: test_loss_list.append(test_loss)

train_loader, test_loader = get_mnist()
visualize(train_loader)

num_epochs = 5
model = MLP()
start = time.time()
train_loss_list = []
test_loss_list = []
optimizer = SGD(model.parameters(), 0.01)
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, epoch, train_loss_list)
    test(model, test_loader, test_loss_list)
    optimizer.step()
print(f"\nCompleted in {time.time() - start: .2f}s")

# line(
#     train_loss_list,
#     yaxis_range=[0, max(train_loss_list) + 0.1],
#     labels={"x": "Batches seen", "y": "Cross entropy loss"},
#     title="ConvNet training on MNIST",
#     width=800,
#     hovermode="x unified",
#     template="ggplot2", # alternative aesthetic for your plots (-:
# )