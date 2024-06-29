import torch.nn as nn


def resnet_loss(outputs, target):
    cross_entropy_loss = nn.CrossEntropyLoss()
    if isinstance(outputs, tuple):
        main_outputs, aux_outputs = outputs
        loss1 = cross_entropy_loss(main_outputs, target)
        loss2 = cross_entropy_loss(aux_outputs, target)
        loss = loss1 + 0.4 * loss2
    else:
        loss = cross_entropy_loss(outputs, target)

    return loss
