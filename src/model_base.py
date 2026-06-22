import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet
#from pretrainedmodels import se_resnext101_32x4d, inceptionresnetv2
#from efficientnet_pytorch import EfficientNet

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting == True:  # made sure to change to '== True' since just 'if feat:' is true for 'partial'
        for param in model.parameters():
            param.requires_grad = False

class ECG_CNN(nn.Module):
        def __init__(self, num_conv_layers=6, in_channels=2, first_layer_filters=128, num_classes=3, kernel_size=7):
            super(ECG_CNN, self).__init__()

            self.num_conv_layers = num_conv_layers
            self.in_channels = in_channels
            self.first_layer_filters = first_layer_filters

            # Initialize the convolutional layers dynamically
            conv_layers = []
            current_filters = first_layer_filters # 128
            for i in range(num_conv_layers): # 10

                if i % 2 != 0:
                    current_filters = int(current_filters)*2 # reset back to last number of filters for odd numbered layers (repeated num filters)
                    out_channels = int(current_filters)

                    # add max pooling layer at every other conv block
                    conv_layers.extend([
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                        nn.MaxPool1d(kernel_size=2,stride=2),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ])
                else: # add conv block without max pooling
                    out_channels = int(current_filters)
                    conv_layers.extend([
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ])

                in_channels = out_channels  # Update in_channels for the next layer
                current_filters /= 2  # Halve the number of filters for the next layer

            self.conv_blocks = nn.Sequential(*conv_layers)
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)

            self._initialize_fc_layers()

        def _initialize_fc_layers(self):
            # Compute output size dynamically
            example_input = torch.randn(16, 2, 2500)  # Example input tensor
            with torch.set_grad_enabled(True):
                conv_output = self.conv_blocks(example_input)
                max_pool_output = self.global_max_pool(conv_output)
                conv_output_shape = max_pool_output.shape[1] # Number of channels = 8



            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_shape, 16),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 3)
            )
        
        def forward(self, x):
            # Convolutional layers
            x = self.conv_blocks(x)

            x = self.global_max_pool(x)

            #x, _ = torch.max(x, dim=1, keepdim=True)
            
            x = x.squeeze(-1) 
            #x = x.squeeze(1) 
            

            # Fully connected layers
            #x = self.fc_layers(x)

            #x = F.softmax(x, dim=-1)

            return x

def initialize_model(params):
    """To add another model into this function:
    1) install the model; 2) create a model_name; 3) copy the format of the other models and create the
    model_ft, set_parameter, etc. 3.5) the num_ftrs might need to load different names (i.e. model_ft.fc or
    model_ft._fc - different models will have different final layers but will generally be 'fc' for pytorch pretrained
    models; '_fc' for efficientNet and 'last_layer' for pretrainedmodels)
    4) add the final layer as just a linear layer*
    4.5)* final layer needs to be a linear layer because pytorch loss functions include softmax and sigmoid activations
    within them
    5) make sure to read the originial architecture and set the input size

    Currently the models I have added are:
    EfficientNet-B7, Resnet18, Resnet50, Resnet101, Resnet152, Alexnet, VGG11_bn, VGG19_bn, Squeezenet1.1, Densenet121,
    Densenet169, Densenet161, MobileNetV2, and Inception v3

    see link for more models and details on each: https://pytorch.org/docs/stable/torchvision/models.html
    """

    print('initializing {} model'.format(params['model_name']))
    model_name = params['model_name']
    num_classes = params['num_classes']
    feature_extract = False
    use_pretrained = False
    fusion = False  # fusion1 or fusion2 for different fusion models
    if fusion != 'fusion1' and fusion != 'fusion2' and fusion != 'fusion3':
        demo_ftrs = 0
        icd_ftrs = 0
        vit_ftrs = 0
    else:
        demo_ftrs = params['demo_ft_len']
        icd_ftrs = params['icd_ft_len']
        vit_ftrs = params['vit_ft_len']
    if demo_ftrs != None and icd_ftrs != None and vit_ftrs != None:
        text_ftrs = demo_ftrs + icd_ftrs + vit_ftrs
    elif demo_ftrs == None and icd_ftrs != None and vit_ftrs == None:
        text_ftrs = icd_ftrs
    elif demo_ftrs != None and icd_ftrs == None and vit_ftrs == None:
        text_ftrs = demo_ftrs
    elif demo_ftrs == None and icd_ftrs == None and vit_ftrs != None:
        text_ftrs = vit_ftrs

    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnext101":
        """resnext101_32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "se-resnext101":
        """ SE-ResNeXt101
        """
        if use_pretrained:
            model_ft = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            model_ft = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        if fusion == 'fusion1':
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.last_linear = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "inceptionresnetv2":
        """ Inception ResNet v2
        """
        if use_pretrained:
            model_ft = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        else:
            model_ft = inceptionresnetv2(num_classes=1000, pretrained=None)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        if fusion == 'fusion1':
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.last_linear = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "effnet-b7":
        """ EfficientNet-B7
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b7')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b5":
        """ EfficientNet-B5
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b5')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b5')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b3":
        """ EfficientNet-B3
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b3')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b3')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b2":
        """ EfficientNet-B2
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b2')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b1":
        """ EfficientNet-B1
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b1')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b1')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b0":
        """ EfficientNet-B0
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b0')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == 'resnet9':
        """ Modified Resnet18 with only one block per layer so that it's 'half'
        - this is a good way to make custom models from pytorch classes
        """
        model_ft = resnet._resnet('resnet18_half', resnet.BasicBlock, [1, 1, 1, 1], False, False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        if fusion == 'fusion1':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier[6] = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg11bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        if fusion == 'fusion1':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier[6] = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg19bn":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        if fusion == 'fusion1':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier[6] = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet1.1":
        """ Squeezenet1.1
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if fusion == 'fusion1':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if fusion == 'fusion1':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet161":
        """ Densenet161
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if fusion == 'fusion1':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet":
        """ MobileNetV2
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        if fusion == 'fusion1':
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier[1] = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class ECG_CNN_Echo_CNN_Fusion(nn.Module):
    def __init__(self, weight_type):
        super(ECG_CNN_Echo_CNN_Fusion, self).__init__()


        #self.model_ft = nn.Sequential(*list(ap4_model.children())[:-1])

        model = models.resnext101_32x8d(pretrained=True) 
        self.model_ap4 = nn.Sequential(*list(model.children())[:-1])

        self.ecg_cnn = ECG_CNN(num_conv_layers=6, first_layer_filters=128, kernel_size=7)

    
        # downsample output features from Echo CNN
        self.fc_layers = nn.Sequential(
                nn.Linear(2048, 1000),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(1000, 8)
            )

        self.ecg_mlp = nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8)
                    # output of 8
            )

        # get 3 value output using fusion input
        self.final_mlp = nn.Sequential(
                    nn.Linear(16, 8), #use for regular model
                    nn.ReLU(),
                    nn.Linear(8, 4)
                )


    def forward(self, echo_input, ecg_input):

        echo_features = self.model_ap4(echo_input)
        echo_features = echo_features.squeeze() 

        # downsample Echo features to match ECG output
        echo_down_feats = self.fc_layers(echo_features)

        ecg_features = self.ecg_cnn(ecg_input) 
        ecg_features = self.ecg_mlp(ecg_features) 

        concat_feats = torch.concat([ecg_features.float(), echo_down_feats.float()], dim=1)


        # feed concatenated features into final MLP thats output has 3 values (softmax in training base code)
        output = self.final_mlp(concat_feats)


        return output

class ECG_Model(nn.Module):
        def __init__(self, num_conv_layers=6, in_channels=2, first_layer_filters=128, num_classes=3, kernel_size=7):
            super(ECG_Model, self).__init__()

            self.num_conv_layers = num_conv_layers
            self.in_channels = in_channels
            self.first_layer_filters = first_layer_filters

            # Initialize the convolutional layers dynamically
            conv_layers = []
            current_filters = first_layer_filters # 128
            for i in range(num_conv_layers): # 10

                if i % 2 != 0:
                    current_filters = int(current_filters)*2 # reset back to last number of filters for odd numbered layers (repeated num filters)
                    out_channels = int(current_filters)

                    # add max pooling layer at every other conv block
                    conv_layers.extend([
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                        nn.MaxPool1d(kernel_size=2,stride=2),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ])
                else: # add conv block without max pooling
                    out_channels = int(current_filters)
                    conv_layers.extend([
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ])

                in_channels = out_channels  # Update in_channels for the next layer
                current_filters /= 2  # Halve the number of filters for the next layer

            self.conv_blocks = nn.Sequential(*conv_layers)
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)

            self._initialize_fc_layers()

        def _initialize_fc_layers(self):
            # Compute output size dynamically
            example_input = torch.randn(16, 2, 2500)  # Example input tensor
            with torch.set_grad_enabled(True):
                conv_output = self.conv_blocks(example_input)
                max_pool_output = self.global_max_pool(conv_output)
                conv_output_shape = max_pool_output.shape[1] # Number of channels = 8



            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_shape, 16),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 3)
            )
        
        def forward(self, x):
            # Convolutional layers
            x = self.conv_blocks(x)

            x = self.global_max_pool(x)

            #x, _ = torch.max(x, dim=1, keepdim=True)
            
            x = x.squeeze(-1) 
            

            # Fully connected layers
            x = self.fc_layers(x)

            x = F.softmax(x, dim=-1)

            return x