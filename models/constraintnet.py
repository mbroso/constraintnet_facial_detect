"""ConstraintNet allows to define constraints for the output in each 
forward pass separately.
"""


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import importlib


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AddConstrParaReprBottleneck(nn.Module):
    """Incorporate the constraint parameter representation g(s) via an
    additional input to the Bottleneck.

    This module has the constraint parameter representation g(s) as an
    additional input. The constraint parameter tensor is concatenated to the
    input of the first conv1x1 layer and the output channels are kept constant.
    The conv3x3 and the second conv1x1 within the bottleneck are untouched.

    Attributes:
        expansion (int): Factor of output channels to channels within 
            bottleneck (planes). Expansion is a class attribute.
        c_constr_para_repr (int): Number of channels of constraint parameter
            representation.
        inplanes (int): Number of input channels.
        downsample (obj): Torch nn.Module for replacing identity bypass with 
            conv1x1 as a parametric bypass. Especially for the first Bottleneck
            within a block.
        convx, bnx (obj):Convolutional layers and batch normalization as 
            nn.Modules.
        relu (obj): ReLU as activation function.
        stride (int): Stride for second conv layer for reducing tensor 
            dimensions.
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, 
            c_constr_para_repr=1):
        """Initialization.

        Args:
            c_constr_para_repr (int): Number of channels of constraint parameter
                representation tensor.
            inplanes (int): Number of input channels.
            downsample (obj): Torch nn.Module for replacing identity bypass 
                with conv1x1 as a parametric bypass. Especially for the first 
                Bottleneck within a block.
            convx, bnx (obj):Convolutional layers and batch normalization as 
                nn.Modules.
            relu (obj): ReLU as activation function.
            stride (int): Stride for second conv layer for reducing tensor 
                dimensions.
        """
        super(AddConstrParaReprBottleneck, self).__init__()
        if planes <= c_constr_para_repr:
            raise ValueError('Number of planes within \
                    AddConstrParaReprBottleneck should be greater than number of \
                    channels of constraint parameter representation.')
        self.c_constr_para_repr = c_constr_para_repr
        self.conv1 = conv1x1(inplanes + self.c_constr_para_repr, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass with constraint parameter representation as additional
        input.

        Constraint parameter representation tensor (constr_para_repr) is an
        additional input. constr_para_repr is concatenated 
        to the input of the first conv1x1 layer.

        Args:
            x (dict): Dictionary with keys 'x_out', 'constr_para_repr'
                out (obj): Torch out input tensor.
                constr_para_repr (obj): Torch tensor encodig the constraints. 
                    The width and height of the tensor is equal to width and 
                    height of x but with self.c_constr_para_repr channels.
        Returns:
            out (obj): Torch output tensor.
        """
        out = x['out']
        constr_para_repr = x['constr_para_repr']

        identity = torch.cat((out, constr_para_repr), dim=1)

        #concatenate the input of the first conv with the region features
        #concatenate over channel dimension N x C x W x H
        out = torch.cat((out, constr_para_repr), dim=1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def opts2constraintnet(opts):
    """Creates ConstraintNet model by calling its constructor with options 
    from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        constraintnet (obj): Instantiated ConstraintNet object.
    """
    block_dict = {
            'bottleneck': Bottleneck, 
            'basicblock': BasicBlock
            }
    #reads parameters from options
    block = block_dict[opts.block]

    constraint_import = 'models.' + opts.constr_module
    constraint_lib = importlib.import_module(constraint_import)
    
    my_constr_para_repr = None
    for name, fct in constraint_lib.__dict__.items():
        if name==opts.opts2constr_para_repr:
            my_constr_para_repr = fct(opts)

    if my_constr_para_repr==None:
        raise NotImplementedError(
                """In {constraint_import} module is no opts2constr_para_repr 
                functor {opts2constr_para_repr} implemented.""".format(
                    constraint_import = constraint_import,
                    opts2constr_para_repr = opts.opts2constr_para_repr
                    )
                )

    print('Loaded constr_para_repr functor via {opts2constr_para_repr}.'.format(
        opts2constr_para_repr = opts.opts2constr_para_repr)
        )

    my_constr_para_trf = None
    for name, fct in constraint_lib.__dict__.items():
        if name==opts.opts2constr_para_trf:
            my_constr_para_trf = fct(opts)

    if my_constr_para_trf==None:
        raise NotImplementedError(
                """In {constraint_import} module is no opts2constr_para_trf 
                functor {opts2constr_para_trf} implemented.""".format(
                    constraint_import = constraint_import,
                    opts2constr_para_trf = opts.opts2constr_para_trf
                    )
                )

    print('Loaded constr_para_trf functor via {opts2constr_para_trf}.'.format(
        opts2constr_para_trf = opts.opts2constr_para_trf)
        )

    
    my_constr_guard_layer = None
    for name, fct in constraint_lib.__dict__.items():
        if name==opts.opts2constr_guard_layer:
            my_constr_guard_layer = fct(opts)

    if my_constr_guard_layer == None:
        raise NotImplementedError(
                """In {constraint_import} module is no opts2constr_guard_layer 
                nn.Module {opts2constr_guard_layer} implemented.""".format(
                    constraint_import = constraint_import,
                    opts2constr_guard_layer = opts.opts2constr_guard_layer
                    )
                )

    print('Loaded constr_guard_layer nn.Module via {opts2constr_guard_layer}.'.format(
        opts2constr_guard_layer = opts.opts2constr_guard_layer)
        )


    print("""Model was constructed by calling function 
            {opts2model} in model module {model_module}.""".format(
            opts2model=opts.opts2model,
            model_module=opts.model_module
            )
        )
    
    return ConstraintNet(block, opts.block_structure, opts.c_constr_para_repr, 
            opts.z_dim, my_constr_para_repr, my_constr_para_trf, 
            my_constr_guard_layer, zero_init_residual=opts.zero_init_residual)




class ConstraintNet(nn.Module):
    """ConstraintNet is a modified ResNet to constrain the output domain in each
    forward pass independently.

    The constraints are encoded by constraint parameters which are summarized 
    in a tensor and called constr_para.
    """

    def __init__(self, block, layers, c_constr_para_repr, z_dim, constr_para_repr, 
            constr_para_trf, constr_guard_layer, zero_init_residual=False):
        """Initialization.

        Args:
            block (cls): Subclass of nn.Module for instantiating blocks (e.g. 
                Bottleneck block).
            layers (list): List comprising 4 integers defining the number of 
                blocks within each of the 4 parts of resnet with same
                width/height.
            c_constr_para_repr (int): Number of feature planes reserved for 
                constraint parameter representation g(s).
            z_dim (int): Dimension of latent representation created by fully
                connected layer output.
            constr_para_repr (obj): Functor to create constraint parameter
                representation tensor based on constraint parameters.
            constr_para_trf (obj): Functor to create a tranformation of the 
                constraint parameters. E.g. transformation in vertex
                representation for constraints in form of convex polytopes.
            constr_guard_layer (obj): PyTorch module which describes the
                constraint-guard layer.
        """

        super(ConstraintNet, self).__init__()
        self.constr_para_repr = constr_para_repr
        self.constr_para_trf = constr_para_trf
        self.constr_guard_layer = constr_guard_layer
        
        self.c_constr_para_repr = c_constr_para_repr
        self.z_dim = z_dim

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_add_constr_para_repr = self._make_add_constr_para_repr_layer( \
                block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #output dimensions correspond to the number of vertices of the convex 
        #region
        self.fc = nn.Linear(512 * block.expansion, self.z_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_add_constr_para_repr_layer(self, block, planes, blocks, stride=1):
        """First Block is AddConstrParaReprBottleneck.
        """

        downsample = None
        if stride != 1 or self.inplanes + self.c_constr_para_repr != \
                planes * block.expansion:
            
            downsample = nn.Sequential(
                    conv1x1(self.inplanes + self.c_constr_para_repr,
                        planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                    )

        layers = []
        layers.append(AddConstrParaReprBottleneck(self.inplanes, planes, stride, \
                downsample, self.c_constr_para_repr))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_img, constr_para):
        """
        Args:
            x_img (obj): Pytorch tensor with image batch with shape
                    (N, C, H, W) 
            constr_para (obj): Pytorch tensor summarizing the parameters
                    describing the constraints. With shape (N, n_constr_para).

        Returns:
            out (obj): Pytorch tensor 
        """

        out = self.conv1(x_img)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        
        #constr_para_repr is concatenated to the output of second building block
        constr_para_repr = self.constr_para_repr(constr_para)
        out_constr_para_repr = {'out': out, 'constr_para_repr': constr_para_repr}
        out = self.layer3_add_constr_para_repr(out_constr_para_repr)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #create intermediate variable z with fully connected layer
        #softmax is part of the constr_guard_layer functor
        z = self.fc(out)
        #create an appropriate representation of the constraint parameters for 
        #constr_guard_layer functor. 
        reprs = self.constr_para_trf(constr_para)
        #apply constr_guard_layer nn.Module
        y = self.constr_guard_layer(z, reprs)

        return y


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    




