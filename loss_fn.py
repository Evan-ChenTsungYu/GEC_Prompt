
import torch.nn as nn
import torch
def GAN_loss(input, label, train_DGI):
    criterion = nn.BCELoss()
    # if train_DGI == 'D':
    #     return -torch.sum(torch.log(D_true) + torch.log(1- D_fake)) # train 出是否能分辨出正確的句子
    # else : #train Generator
    #     return -torch.sum(torch.log(D_fake)) # 想辦法讓 generator 在 discriminator 中的分數上升
    return criterion(input, label)

    