import torch.nn as nn
from module import Linear, Conv2d, Sequential, Linear_, Conv2d_, BatchNorm2d, BasicBlock, BasicBlock_wo_BN


class ResNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_w_para = nn.Sequential(Sequential(Conv2d_(3, 8, (3, 3), (1, 1), 1, 1e-3)),
                                           Sequential(Conv2d_(8, 16, (3, 3), (1, 1), 1, 1e-3)),
                                           Sequential(Conv2d(16, 32, (3, 3), (1, 1), 1, 1e-1)),
                                           Sequential(Conv2d(32, 32, (3, 3), (1, 1), 1, 1e-1)),
                                           Sequential(Linear(32 * 4 * 4, 10, 1e-1))
                                           )
        self.module_wo_para = nn.Sequential(nn.Sequential(nn.ReLU(inplace=True)),
                                            nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(2)),
                                            nn.Sequential(nn.ReLU(inplace=True)),
                                            nn.Sequential(nn.ReLU(inplace=True)),
                                            nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                                            )

    def forward(self, x, add_noise=None):
        if add_noise is None:
            add_noise = len(self.module_w_para) * [False]
        x = self.module_wo_para[0](self.module_w_para[0](x, add_noise[0]))
        x = self.module_wo_para[1](self.module_w_para[1](x, add_noise[1]))
        x = self.module_wo_para[2](self.module_w_para[2](x, add_noise[2]))
        x = self.module_wo_para[3](self.module_w_para[3](x, add_noise[3])) + x
        x = self.module_w_para[4](self.module_wo_para[4](x), add_noise[4])
        return x

    def backward(self, loss):
        for seq in self.module_w_para:
            seq.backward(loss)

    def fetch_gradient(self):
        return [seq.fetch_gradient() for seq in self.module_w_para]


class VGG8(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_w_para = nn.Sequential(Sequential(Conv2d_(3, 16, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d_(16, 16, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(2)),
                                           Sequential(Conv2d_(16, 32, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d_(32, 32, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(2)),
                                           Sequential(Conv2d(32, 64, (3, 3), (1, 1), 1, 1e-2),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d(64, 64, (3, 3), (1, 1), 1, 1e-2),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(2)),
                                           Sequential(nn.Flatten(),
                                                      Linear(64 * 4 * 4, 256, 1e-2),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Linear(256, 10, 1e-2))
                                           )

    def forward(self, x, add_noise=None):
        if add_noise is None:
            add_noise = len(self.module_w_para) * [False]
        for i, (block, add) in enumerate(zip(self.module_w_para,add_noise)):
            x = block(x, add)
        return x

    def backward(self, loss):
        for seq in self.module_w_para:
            seq.backward(loss)

    def fetch_gradient(self):
        return [seq.fetch_gradient() for seq in self.module_w_para]
    

class ResNet5_w_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_w_para = nn.Sequential(Sequential(Conv2d_(3, 8, (3, 3), (1, 1), 1, 1e-3)),
                                           Sequential(BatchNorm2d(8, 1e-3),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d_(8, 16, (3, 3), (1, 1), 1, 1e-3)),
                                           Sequential(BatchNorm2d(16, 1e-3),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(2)),
                                           Sequential(Conv2d(16, 32, (3, 3), (1, 1), 1, 1e-1)),
                                           Sequential(BatchNorm2d(32, 1e-1),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d(32, 32, (3, 3), (1, 1), 1, 1e-1)),
                                           Sequential(BatchNorm2d(32, 1e-1),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(4), 
                                                      nn.Flatten()),
                                           Sequential(Linear(32 * 4 * 4, 10, 1e-1))
                                           )
        
    def forward(self, x, add_noise=None):
        if add_noise is None:
            add_noise = len(self.module_w_para) * [False]
        for i, (block, add) in enumerate(zip(self.module_w_para,add_noise)):
            x = block(x, add)
        return x

    def backward(self, loss):
        for seq in self.module_w_para:
            seq.backward(loss)
            
    def fetch_gradient(self):
        return [seq.fetch_gradient() for seq in self.module_w_para]
    

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_w_para = nn.Sequential(Sequential(Conv2d_(3, 64, (7, 7), (2, 2), 3, 1e-3)),
                                           Sequential(BatchNorm2d(64, 1e-3),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(3, 2, 1)),
                                           Sequential(BasicBlock(Conv2d_, 64, 64, (3,3), 1, 1e-3)),
                                           Sequential(BasicBlock(Conv2d_, 64, 64, (3,3), 1, 1e-3)),
                                           Sequential(BasicBlock(Conv2d_, 64, 128, (3,3), 2, 1e-3)),
                                           Sequential(BasicBlock(Conv2d_, 128, 128, (3,3), 1, 1e-3)),
                                           Sequential(BasicBlock(Conv2d, 128, 256, (3,3), 2, 1e-3)),
                                           Sequential(BasicBlock(Conv2d, 256, 256, (3,3), 1, 1e-3)),
                                           Sequential(BasicBlock(Conv2d, 256, 512, (3,3), 2, 1e-3)),
                                           Sequential(BasicBlock(Conv2d, 512, 512, (3,3), 1, 1e-3)),
                                           Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                                      nn.Flatten(),
                                                      Linear(512, 10, 1e-1))
                                           )
        
    def forward(self, x, add_noise=None):
        if add_noise is None:
            add_noise = len(self.module_w_para) * [False]
        for i, (block, add) in enumerate(zip(self.module_w_para,add_noise)):
            x = block(x, add)
        return x
        # 0 torch.Size([5000, 64, 16, 16])
        # 1 torch.Size([5000, 64, 8, 8])
        # 2 torch.Size([5000, 64, 8, 8])
        # 3 torch.Size([5000, 64, 8, 8])
        # 4 torch.Size([5000, 128, 4, 4])
        # 5 torch.Size([5000, 128, 4, 4])
        # 6 torch.Size([5000, 256, 2, 2])
        # 7 torch.Size([5000, 256, 2, 2])
        # 8 torch.Size([5000, 512, 1, 1])
        # 9 torch.Size([5000, 512, 1, 1])
        # 10 torch.Size([5000, 10])

    def backward(self, loss):
        for seq in self.module_w_para:
            seq.backward(loss)
            
    def fetch_gradient(self):
        return [seq.fetch_gradient() for seq in self.module_w_para]


class ResNet18_wo_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_w_para = nn.Sequential(Sequential(Conv2d_(3, 64, (7, 7), (2, 2), 3, 1e-3),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(3, 2, 1)),
                                           Sequential(BasicBlock_wo_BN(Conv2d, 64, 64, (3,3), 1, 1e-3)),
                                           Sequential(BasicBlock_wo_BN(Conv2d, 64, 64, (3,3), 1, 1e-3)),
                                           Sequential(BasicBlock_wo_BN(Conv2d, 64, 128, (3,3), 2, 1e-3)),
                                           Sequential(BasicBlock_wo_BN(Conv2d, 128, 128, (3,3), 1, 1e-3)),
                                           Sequential(BasicBlock_wo_BN(Conv2d, 128, 256, (3,3), 2, 1e-3)),
                                           Sequential(BasicBlock_wo_BN(Conv2d, 256, 256, (3,3), 1, 1e-3)),
                                           Sequential(BasicBlock_wo_BN(Conv2d, 256, 512, (3,3), 2, 1e-3)),
                                           Sequential(BasicBlock_wo_BN(Conv2d, 512, 512, (3,3), 1, 1e-3)),
                                           Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                                      nn.Flatten(),
                                                      Linear(512, 10, 1e-1))
                                           )
        
    def forward(self, x, add_noise=None):
        if add_noise is None:
            add_noise = len(self.module_w_para) * [False]
        for i, (block, add) in enumerate(zip(self.module_w_para,add_noise)):
            x = block(x, add)
        return x

    def backward(self, loss):
        for seq in self.module_w_para:
            seq.backward(loss)
            
    def fetch_gradient(self):
        return [seq.fetch_gradient() for seq in self.module_w_para]

