from base.base_trainer import BaseTrainer
from dataprovider.data_loader import CIFAR10DataLoader
from dataprovider.data_setter import CIFAR10DataSetter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from model.resnet.loss import resnet_loss
from model.resnet.metric import resnet_accuracy
from model.resnet.model import ResNet18


class ResNetTrainer(BaseTrainer):

    def __init__(self, model, loss, optimizer, metric, train_data_loader, valid_data_loader, mac_gpu=True, *args, **kwargs):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.mac_gpu = mac_gpu

        if self.mac_gpu:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model)

    def train(self, epoch):
        print(f'{epoch} 번의 학습 시작')
        self.model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            batch_loss = self.loss(outputs, targets)
            batch_loss.backward()

            self.optimizer.step()
            train_loss += batch_loss.item()

            batch_total, batch_correct = self.metric(outputs, targets)
            train_total += batch_total
            train_correct += batch_correct

        train_acc = 100. * train_correct / train_total

        return train_loss, train_acc

    def validate(self):
        val_loss = 0
        total = 0
        correct = 0

        self.model.eval()
        for batch_idx, (inputs, targets) in enumerate(self.valid_data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            total += targets.size(0)

            outputs = self.model(inputs)
            val_loss += self.loss(outputs, targets).item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        val_acc = 100 * correct / total
        return val_loss, val_acc

    def adjust_learning_rate(self, epoch, learning_rate):
        lr = learning_rate
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    epochs = 10
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 0.0002

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10DataSetter(root='./data', train=True, download=False, transform=transform_train)
    test_dataset = CIFAR10DataSetter(root='./data', train=False, download=False, transform=transform_test)

    train_loader = CIFAR10DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = CIFAR10DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    model = ResNet18()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    loss_fn = resnet_loss
    metric_fn = resnet_accuracy

    resnet_trainer = ResNetTrainer(model=model, loss=loss_fn, optimizer=optimizer, metric=metric_fn,
                                   train_data_loader=train_loader, valid_data_loader=test_loader,
                                   mac_gpu=True)

    for epoch in range(0, epochs):
        resnet_trainer.adjust_learning_rate(epoch, learning_rate)
        train_loss, train_acc = resnet_trainer.train(epoch)
        val_loss, val_acc = resnet_trainer.validate()
        
        print("train_loss", train_loss)
        print("train_acc", train_acc)

        print("val_loss", val_loss)
        print("val_acc", val_acc)
