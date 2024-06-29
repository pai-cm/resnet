from base.base_trainer import BaseTrainer
from dataprovider.data_loader import CIFAR10DataLoader
from dataprovider.data_setter import CIFAR10DataSetter
import torch
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

    def _train_epoch(self, epoch):
        batch_loss = 0
        batch_total = 0
        batch_correct = 0
        self.model.train()

        for inputs, targets in tqdm(self.train_data_loader):
            if self.mac_gpu:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()

            self.optimizer.step()

            batch_loss += loss.item()
            batch_total, batch_correct = self.metric(outputs, targets)

        epoch_loss = batch_loss / len(self.train_data_loader.dataset)
        epoch_accuracy = 100 * batch_correct / batch_total

        return epoch_loss, epoch_accuracy

    def train(self, epochs):
        print(f'{epochs} 번의 학습 시작')
        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self._train_epoch(epoch)
            print(f'Epoch: {epoch} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy}')
        print(f'{epochs} 번의 학습 완료')
        return epoch_loss, epoch_accuracy

    def validate(self):
        val_loss = 0
        total = 0
        correct = 0

        self.model.eval()
        for inputs, labels in self.valid_data_loader:

            inputs, targets = inputs.to(self.device), labels.to(self.device)
            total += targets.size(0)

            outputs = self.model(inputs)

            val_loss = self.loss(outputs, labels).item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        val_acc = 100 * correct / total
        return val_loss, val_acc


if __name__ == '__main__':
    epochs = 1
    batch_size = 64

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
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
    loss_fn = resnet_loss
    metric_fn = resnet_accuracy
    resnet_trainer = ResNetTrainer(model=model, loss=loss_fn, optimizer=optimizer, metric=metric_fn,
                                   train_data_loader=train_loader, valid_data_loader=test_loader,
                                   mac_gpu=True)

    train_loss, train_acc = resnet_trainer.train(epochs)
    val_loss, val_acc = resnet_trainer.validate()
