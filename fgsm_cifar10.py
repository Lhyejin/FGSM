from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



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

    if args.model_parameter is not None:
      model.load_state_dict(torch.load(args.model_parameter))
    
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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--dataset-normalize', action='store_true' , default=False,
                        help='input whether normalize or not (default: False)')
    parser.add_argument('--train-mode', action='store_false', default=True,
                        help='input whether training or not (default: True')
    parser.add_argument('--model_parameter', type=str, default=None,
                        help='if test mode, input model parameter path')
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

    train_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('./data', train=True, download=True,
                        transform=transformation),
      batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('./data', train=False, download=True,
                        transform=transformation),
      batch_size=args.test_batch_size, shuffle=True)
  
    model = GoogLeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.train_mode:
        train(args, model, device, train_loader, optimizer)
        test(args, model, device, test_loader)
    else:
        test(args, model, device, test_loader)

if __name__ == '__main__':
    main()
