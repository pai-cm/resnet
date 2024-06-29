from dataprovider.data_setter import CIFAR10DataSetter
from dataprovider.data_loader import CIFAR10DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from model.resnet.loss import resnet_loss
from model.resnet.metric import resnet_accuracy
from model.resnet.model import ResNet18
from trainer import ResNetTrainer


def train(model, loss_fn, metric_fn, epochs=20, momentum=0.9, weight_decay=0.0002, lr=0.001):
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

    if model == 'resnet':
        model = ResNet18()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if loss_fn == 'resnet':
        loss_fn = resnet_loss

    if metric_fn == 'resnet':
        metric_fn = resnet_accuracy

    resnet_trainer = ResNetTrainer(model=model, loss=loss_fn, optimizer=optimizer, metric=metric_fn,
                                   train_data_loader=train_loader, valid_data_loader=test_loader, mac_gpu=True)
    train_loss, train_acc = resnet_trainer.train(epochs)
    print(f"Trian Loss : {train_loss}, Trian accuracy : {train_acc}")

    val_loss, val_acc = resnet_trainer.validate()
    print(f"Validation Loss : {val_loss}, Validation accuracy : {val_acc}")


if __name__ == '__main__':
    train(model="resnet",
          loss_fn="resnet",
          metric_fn="resnet",
          epochs=2,
          momentum=0.9,
          weight_decay=0.0002,
          lr=0.1)
