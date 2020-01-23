import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
from torchvision import models
import os
import numpy as np
from utilities import *

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
    def output_num(self):
        pass

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Conv_Block_gn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1):
        super(Conv_Block_gn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.GroupNorm(groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Dense_Block_unnorm(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block_unnorm, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class Generator_c2s(nn.Module):
    def __init__(self):
        super(Generator_c2s, self).__init__()
        self.conv1_1 = Conv_Block_gn(3, 32, kernel_size=3, groups=32)
        self.conv1_2 = Conv_Block_gn(32, 32, kernel_size=3, groups=32)
        self.conv1_3 = Conv_Block_gn(32, 32, kernel_size=3, groups=32)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = Conv_Block_gn(32, 64, kernel_size=3, groups=16)
        self.conv2_2 = Conv_Block_gn(64, 64, kernel_size=3, groups=16)
        self.conv2_3 = Conv_Block_gn(64, 64, kernel_size=3, groups=16)
        self.pool2 = nn.MaxPool2d(2, stride=2) 
        self.drop1 = nn.Dropout()
        self.fc1 = Dense_Block_unnorm(7*7*64, 100)
        self.drop2 = nn.Dropout()
        self.fc2 = Dense_Block_unnorm(100, 100)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x =  self.fc1(x)
        x = self.drop2(x)
        x =  self.fc2(x)
        return x

class Generator_m2m(nn.Module):
    def __init__(self):
        super(Generator_m2m, self).__init__()
        self.conv1 = Conv_Block(3, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = Conv_Block(20, 50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.drop = nn.Dropout()
        self.fc = Dense_Block(800, 500)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x




class CIFAR10Fc(BaseFeatureExtractor):
    def __init__(self, normalize=False):
        super(CIFAR10Fc, self).__init__()
        self.model_cifar = Generator_c2s()
        if normalize:
            self.normalize=True
            self.mean=False
            self.std=False
        else:
            self.normalize=False
        self.__in_features = 100

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.model_cifar(x)
        return x

    def output_num(self):
        return self.__in_features

class MNISTFc(BaseFeatureExtractor):
    def __init__(self, normalize=False):
        super(MNISTFc, self).__init__()
        self.model_mnist = Generator_m2m()
        if normalize:
            self.normalize=True
            self.mean=False
            self.std=False
        else:
            self.normalize=False
        self.__in_features = 500

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.model_mnist(x)
        return x

    def output_num(self):
        return self.__in_features

class AlexNetFc(BaseFeatureExtractor):
    def __init__(self, normalize=True):
        super(AlexNetFc, self).__init__()
        self.model_alexnet = models.alexnet(pretrained=True)
        if normalize:
            self.normalize=True
            self.mean=False
            self.std=False
        else:
            self.normalize=False

        self.model_alexnet_features = nn.Sequential(*list(self.model_alexnet.features.children()))
        self.model_alexnet_classifier = nn.Sequential(*list(self.model_alexnet.classifier.children())[:-2])

        self.__in_features = self.model_alexnet_classifier[-1].out_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.model_alexnet_features(x)
        x = x.view(x.size(0), -1)
        x = self.model_alexnet_classifier(x)
        return x

    def output_num(self):
        return self.__in_features




class ResNetFc(BaseFeatureExtractor):
    def __init__(self, model_name='resnet50',model_path=None, normalize=True):
        super(ResNetFc, self).__init__()
        self.model_resnet = resnet_dict[model_name](pretrained=True)
        #if not os.path.exists(model_path):
        #    model_path = None
        #    print('invalid model path!')
        #if model_path:
        #    self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = self.grl(x)
        for module in self.main.children():
            x = module(x)
        return x

class LargeAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )

class SmallAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(SmallAdversarialNetwork, self).__init__()
        self.ad_layer1 = Dense_Block_unnorm(in_feature, 100)
        self.ad_layer2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.ad_layer1,
            self.ad_layer2,
            self.sigmoid
        )

class Discriminator(nn.Module):
    def __init__(self, n=10):
        super(Discriminator, self).__init__()
        self.n = n
        def f():
            return nn.Sequential(
                #nn.Linear(2048, 256),
                nn.Linear(100, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)

class CLS_0(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS_0, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out
    
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs
    
class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)
