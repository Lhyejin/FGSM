from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms


# FC Network - default
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(28*28, 500)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    output = self.fc2(x)
    return output


# Convolutional Network
class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(36864, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output


# ConvNet + Dropout
class NetDrop(nn.Module):
    def __init__(self):
        super(NetDrop, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


# Training phase
def train(args, model, device, train_loader, optimizer):
    model.train()
    # cross entropy loss
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (image, target) in enumerate(train_loader):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        # Save the model checkpoint
        if args.save_model:
            torch.save(model.state_dict(), 'model/model-{}.ckpt'.format(epoch))


# Test phase
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    adv_correct = 0
    misclassified = 0
    criterion = nn.CrossEntropyLoss()
    for images, targets in test_loader:
        images = Variable(images.to(device), requires_grad=True)
        targets = Variable(targets.to(device))

        outputs = model(images)
        loss = criterion(outputs, targets)
        test_loss += loss
        loss.backward()

        # Generate perturbation
        grad_j = torch.sign(images.grad.data)
        adv_images = torch.clamp(images.data + args.epsilon * grad_j, 0, 1)

        adv_outputs = model(Variable(adv_images))

        _, preds = torch.max(outputs.data, 1)
        _, adv_preds = torch.max(adv_outputs.data, 1)

        correct += (preds == targets).sum().item()
        adv_correct += (adv_preds == targets).sum().item()
        misclassified += (preds != adv_preds).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('\nAdversarial Test: Accuracy: {}/{} ({:.0f})\n'.format(
        adv_correct, len(test_loader.dataset),
        100. * adv_correct / len(test_loader.dataset)))
    print('\nmisclassified examples : {}/ {}\n'.format(
        misclassified, len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=640, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--dataset-normalize', action='store_true' , default=False,
                        help='input whether normalize or not (default: False)')
    parser.add_argument('--network', type=str, default='fc',
                        help='input Network type (Selected: fc, conv, drop / default: \'fc\')')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transformation = transforms.ToTensor()
    # Dataset normalize
    if args.dataset_normalize:
        transformation = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transformation),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transformation),
        batch_size=args.test_batch_size, shuffle=True)

    # Network Type
    if args.network == 'conv':
        model = NetConv().to(device)
    elif args.network == 'drop':
        model = NetDrop().to(device)
    elif args.network == 'fc':
        model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, device, train_loader, optimizer)

    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
