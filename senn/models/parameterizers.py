import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearParameterizer(nn.Module):
    def __init__(self, num_concepts, num_classes, hidden_sizes=(10, 5, 5, 10), dropout=0.5, **kwargs):
        """Parameterizer for compas dataset.
        
        Solely consists of fully connected modules.

        Parameters
        ----------
        num_concepts : int
            Number of concepts that should be parameterized (for which the relevances should be determined).
        num_classes : int
            Number of classes that should be distinguished by the classifier.
        hidden_sizes : iterable of int
            Indicates the size of each layer in the network. The first element corresponds to
            the number of input features.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        layers = []
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(nn.Linear(h, h_next))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        layers.pop()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of compas parameterizer.

        Computes relevance parameters theta.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        parameters : torch.Tensor
            Relevance scores associated with concepts. Of shape (BATCH, NUM_CONCEPTS, NUM_CLASSES)
        """
        return self.layers(x).view(x.size(0), self.num_concepts, self.num_classes)


class ConvParameterizer(nn.Module):
    def __init__(self, num_concepts, num_classes, cl_sizes=(1, 10, 20), kernel_size=5, hidden_sizes=(10, 5, 5, 10), dropout=0.5,
                 **kwargs):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.cl_sizes = cl_sizes
        self.kernel_size = kernel_size
        self.dropout = dropout

        cl_layers = []
        for h, h_next in zip(cl_sizes, cl_sizes[1:]):
            cl_layers.append(nn.Conv2d(h, h_next, kernel_size=self.kernel_size, padding=self.kernel_size // 2))
            cl_layers.append(nn.MaxPool2d(2, stride=2))
            cl_layers.append(nn.ReLU())

        cl_layers.insert(-2, nn.Dropout2d(self.dropout))
        self.cl_layers = nn.Sequential(*cl_layers)

        fc_layers = []
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            fc_layers.append(nn.Linear(h, h_next))
            fc_layers.append(nn.Dropout(self.dropout))
            fc_layers.append(nn.ReLU())
        fc_layers.pop()
        fc_layers.append(nn.Tanh())
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        cl_output = self.cl_layers(x)
        flattened = cl_output.view(x.size(0), -1)
        return self.fc_layers(flattened).view(-1, self.num_concepts, self.num_classes)
    


# taken from  https://github.com/dmelis/SENN/blob/master/SENN/models.py#L35





#===============================================================================
#====================      VGG MODELS FOR CIFAR  ===============================
#===============================================================================

# Note that these are tailored to native 32x32 resolution

cfg_cifar = {
    'vgg8':  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_CIFAR(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG_CIFAR, self).__init__()
        self.features = self._make_layers(cfg_cifar[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11_cifar():
    return VGG_CIFAR('vgg11')


def vgg13_cifar():
    return VGG_CIFAR('vgg13')


def vgg16_cifar():
    return VGG_CIFAR('vgg16')


def vgg19_cifar():
    return VGG_CIFAR('vgg19')



class vgg_parametrizer(nn.Module):
    """ Parametrizer function - VGG

        Args:
            din (int): input concept dimension
            dout (int): output dimension (1 or number of label classes usually)

        Inputs:
            x:  Image tensor (b x c x d^2) [TODO: generalize to set maybe?]

        Output:
            Th:  Theta(x) vector of concept scores (b x nconcept x dout) (TODO: generalize to multi-class scores)
    """

    def __init__(self, din, nconcept, dout, arch = 'alexnet', nchannel = 1, only_positive = False):
        super(vgg_parametrizer, self).__init__()
        self.nconcept = nconcept
        self.dout = dout
        self.din  = din
        self.net = VGG_CIFAR(arch, num_classes = nconcept*dout)
        # if arch == 'alexnet':
        #     self.net = torchvision.models.alexnet(num_classes = nconcept*dout)
        # elif arch == 'vgg11':
        #     self.net = torchvision.models.vgg11(num_classes = nconcept*dout)
        # elif arch == 'vgg16':
        #     self.net = torchvision.models.vgg16(num_classes = nconcept*dout)
        # elif arch == 'vgg16':
        #     self.net = torchvision.models.vgg16(num_classes = nconcept*dout)

        self.positive = only_positive

    def forward(self, x):
        p = self.net(x)
        out = F.dropout(p, training=self.training).view(-1,self.nconcept,self.dout)
        if self.positive:
            #out = F.softmax(out, dim = 1) # For fixed outputdim, sum over concepts = 1
            out = F.sigmoid(out) # For fixed outputdim, sum over concepts = 1
        else:
            out = F.tanh(out)
        return out