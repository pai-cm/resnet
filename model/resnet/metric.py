import torch


def resnet_accuracy(output, target):
    total = 0
    correct = 0

    _, predicted = output.max(1)
    total += target.size(0)
    correct += predicted.eq(target).sum().item()

    return total, correct
