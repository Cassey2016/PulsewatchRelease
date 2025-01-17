"""
Copied from https://github.com/zdzdliu/PPGArrhythmiaDetection
Multiclass Arrhythmia Detection and Classification from Photoplethysmography Signals Using a Deep Convolutional Neural Network

@article{liu2022multiclass,
  title={Multiclass Arrhythmia Detection and Classification From Photoplethysmography Signals Using a Deep Convolutional Neural Network},
  author={Liu, Zengding and Zhou, Bin and Jiang, Zhiming and Chen, Xi and Li, Ye and Tang, Min and Miao, Fen},
  journal={Journal of the American Heart Association},
  volume={11},
  number={7},
  pages={e023555},
  year={2022},
  publisher={Am Heart Assoc}
}
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

class VGG(nn.Module):

    def __init__(self, features, ngpu, num_classes=4, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.ngpu = ngpu

        self.classifier = nn.Sequential(
            # nn.Linear(1024, 256),
            nn.Linear(1536,256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.features, x, range(self.ngpu))
            x = x.view(x.size(0), -1)
            x = nn.parallel.data_parallel(self.classifier, x, range(self.ngpu))
        else:
            # print('forward: x.shape',x.shape)
            x = self.features(x)
            # print('forward after features: x.shape',x.shape)
            x = x.view(x.size(0), -1)
            # print('forward after x.view: x.shape',x.shape)
            x = self.classifier(x)
            # print('forward after classifier: x.shape',x.shape)
            # output = torch.squeeze(x,dim=-1) # (none, 3, 1) -> (none, 3)
            # print('output.shape',output.shape)
            output_prob = F.softmax(x, dim=1) # Along the second dim (3) in (none, 3, 1) dimension.
            # output_prob = torch.squeeze(output_prob,dim=-1) # (none, 3, 1) -> (none, 3)
            # print('output_prob.shape',output_prob.shape)
            # print('x and output equal:',torch.equal(x, output))
        return x, output_prob

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data,mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,mode='fan_out')
                m.bias.data.zero_()

def make_layers(cfg, in_channels, batch_norm=True):
    layers = []
    # in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=3, stride=3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'E': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
}

def vgg16_bn(in_channels, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    model = VGG(make_layers(cfg['D'], in_channels, batch_norm=True), **kwargs)
    return model