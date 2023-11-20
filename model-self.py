import torch.nn as nn
import torch

# 预训练模型
model_urls = {
    'vgg11':'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13':'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16':'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19':'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

# VGG网络模型
class VGG(nn.Module):#继承Module这个类
    def __init__(self,features,num_classes=1000,init_weights=False):#继承VGG这个类
        super(VGG,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.features(x),
        x = torch.flatten(x,start_dim=1),
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias,0)

def make_features(cfg:list):
    layers = []
    in_channels = 3
    for v in cfg:
       if v == "M":
           layers += [nn.MaxPool2d(kernel_size=3,stride=2)]
       else:
           conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
           layers += [conv2d,nn.ReLU(True)]
           in_channels = v
    return nn.Sequential(*layers)

# 各模型feature结构
cfgs = {
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}

def vgg(model_name='vgg16',**kwargs):
    assert model_name in cfgs,"Warning:model munber {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg),**kwargs)
    return model