"""Defined functions for general classification model building
Essentially the functions from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
import torch
import torch.nn as nn
import math
from torchvision import models
from torchvision.models import resnet
from pretrainedmodels import se_resnext101_32x4d, inceptionresnetv2
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting == True:  # made sure to change to '== True' since just 'if feat:' is true for 'partial'
        for param in model.parameters():
            param.requires_grad = False


class ECG_CNN1D(nn.Module):
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
                nn.Linear(8, 4)
            )
        
        def forward(self, x):
            # Convolutional layers
            x = self.conv_blocks(x)

            x = self.global_max_pool(x)

            #x, _ = torch.max(x, dim=1, keepdim=True)
            
            x = x.squeeze(-1) 
            #x = x.squeeze(1) 
            

            # Fully connected layers
            x = self.fc_layers(x)

            #x = F.softmax(x, dim=-1)

            return x

class ECG_CNN_Echo_CNN_Fusion(nn.Module):
    def __init__(self, weight_type):
        super(ECG_CNN_Echo_CNN_Fusion, self).__init__()

        # Echo CNN
        if weight_type == 'AP4':

            #ap4_model = torch.load('./models/AP4_resnext101_best.pth')
            ap4_model = torch.load('./models/4class_resnext101_best.pth')

            self.model_ft = nn.Sequential(*list(ap4_model.children())[:-1])

        else:
            self.model_ft = models.resnext101_32x8d(pretrained=True) 

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

        # get 4 value output using fusion input
        self.final_mlp = nn.Sequential(
                    nn.Linear(16, 8), #use for regular model
                    nn.ReLU(),
                    nn.Linear(8, 4)
                )


    def forward(self, echo_input, ecg_input):

        echo_features = self.model_ft(echo_input)
        echo_features = echo_features.squeeze() 

        # downsample Echo features to match ECG output
        echo_down_feats = self.fc_layers(echo_features)

        ecg_features = self.ecg_cnn(ecg_input) 
        ecg_features = self.ecg_mlp(ecg_features) 

        concat_feats = torch.concat([ecg_features.float(), echo_down_feats.float()], dim=1)


        # feed concatenated features into final MLP thats output has 3 values (softmax in training base code)
        output = self.final_mlp(concat_feats)


        return output


class ECGEcho_Fusion(nn.Module):
    def __init__(self, ecg_encoder):
        super(ECGEcho_Fusion, self).__init__()

 

        ap4_model = torch.load('./models/4class_echo_resnext101_best.pth')

        self.model_ft = nn.Sequential(*list(ap4_model.children())[:-1])

        ecg_cnn = ecg_encoder
        self.ecg_cnn = ecg_cnn.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) 

    
        # downsample output features from Echo CNN
        self.fc_layers = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(1024, 8)
            )

        self.ecg_mlp = nn.Sequential(
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8)
                    # output of 8
            )

        # get 4 value output using fusion input
        self.final_mlp = nn.Sequential(
                    nn.Linear(16, 8), #use for regular model
                    nn.ReLU(),
                    nn.Linear(8, 4)
                )


    def forward(self, echo_input, ecg_input):

        echo_features = self.model_ft(echo_input)
        echo_features = echo_features.squeeze() 

        # downsample Echo features to match ECG output
        echo_down_feats = self.fc_layers(echo_features)

        ecg_features = self.ecg_cnn(ecg_input) 
        ecg_features = self.pool(ecg_features)
        ecg_features = torch.flatten(ecg_features, 1) 
        ecg_features = self.ecg_mlp(ecg_features) 

        concat_feats = torch.concat([ecg_features.float(), echo_down_feats.float()], dim=1)

        # feed concatenated features into final MLP thats output has 3 values (softmax in training base code)
        output = self.final_mlp(concat_feats)

        return output

class ImageConvNet(nn.Module):
    def __init__(self, in_channels=3):
        super(ImageConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.drop2 = nn.Dropout(0.2)

        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Like GlobalAveragePooling1D

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.global_pool(x)

        return x.view(x.size(0), -1)  # Flatten to (batch_size, 32)
        
class ICI_Fusion(nn.Module):
    def __init__(self, ecg_encoder):
        super(ICI_Fusion, self).__init__()

        ecg_cnn = ecg_encoder
        self.ecg_cnn = ecg_cnn.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.ecg_mlp = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32)
        )

        self.tab_layer = nn.Linear(14, 32)

        self.final_mlp = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, ecg_input, tab_input):
        ecg_feats = self.ecg_cnn(ecg_input)
        ecg_feats = self.pool(ecg_feats)
        ecg_feats = torch.flatten(ecg_feats, 1) 
        ecg_feats = self.ecg_mlp(ecg_feats)

        tab_feats = self.tab_layer(tab_input)

        concat_feats = torch.concat([ecg_feats.float(), tab_feats.float()], dim=1)

        output = self.final_mlp(concat_feats)

        return output




class Fusion_Modelv3(nn.Module):
    def __init__(self, model, model_ftrs, num_classes, demo_ftrs=None, icd_ftrs=None, vital_ftrs=None):
        """v3 where the demographic and vital information features are just concatinated to the others to reduce
        information reduction as well as a dense layer (torch.linear) to the maximum feature vector - although there
        will be the problem of information dilution, it's better than the previous information loss from going to a very
        small layer (I think)

        model is the actual model with the linear output being the same size as linear input (image features)
        model_ftrs is the length of the image feature vector
        demo_ftrs is the length of the demographic feature vector
        icd_ftrs is the length of the icd code feature vector
        vital_ftrs is the length of the vital information feature vector
        """
        super(Fusion_Modelv3, self).__init__()
        self.image_branch = model

        # make all that is none is zero so the vectors are consistent
        if demo_ftrs == None:
            demo_ftrs = 0
        if icd_ftrs == None:
            icd_ftrs = 0
        if vital_ftrs == None:
            vital_ftrs = 0
        # summing up the feature length for non-image features since they are going to be concatinated
        non_img_ftrs = sum((demo_ftrs, icd_ftrs, vital_ftrs))
        # find the maximum feature vector
        max_feat = max(model_ftrs, demo_ftrs, icd_ftrs, vital_ftrs)
        # non-image branch is just the concat features densely connected to the larger feature vector
        self.non_img_branch = nn.Linear(non_img_ftrs, max_feat)
        self.fused_output = nn.Linear(max_feat*2, num_classes)

    def forward(self, image, demo_feat=None, icd_feat=None, vital_feat=None):
        img_ft = self.image_branch(image)  # shape: 32, 1024, 4, 4
        # make all that is none as an empty tensor so that torch cat ignores empty ones
        if demo_feat != None and icd_feat != None and vital_feat != None:
            non_img_ft = self.non_img_branch(
                torch.cat([torch.squeeze(demo_feat), torch.squeeze(icd_feat), torch.squeeze(vital_feat)], 1))
        elif demo_feat != None:
            non_img_ft = self.non_img_branch(torch.squeeze(demo_feat))
        elif icd_feat != None:
            non_img_ft = self.non_img_branch(torch.squeeze(icd_feat))
        elif vital_feat != None:
            non_img_ft = self.non_img_branch(torch.squeeze(vital_feat))

        # concat the image and non-image features together
        fused_ft = torch.cat([img_ft, non_img_ft], 1)

        return self.fused_output(fused_ft)


class Fusion_Modelv2(nn.Module):
    # v2.1 where the demographic features are just concatinated
    def __init__(self, model, model_ftrs, num_classes, demo_ftrs=None, icd_ftrs=None):
        """model is the actual model with the linear output being the same size as linear input
        model_ftrs is the length of the model feature vector
        text_ftrs is the length of the input text features"""
        super(Fusion_Modelv2, self).__init__()
        self.image_branch = model

        # find the smallest features (while checking if demo or icd or both are inputs)
        if demo_ftrs != None and icd_ftrs != None:
            min_feat = min(model_ftrs, demo_ftrs, icd_ftrs)
            self.demo_branch = nn.Linear(demo_ftrs, min_feat)
            self.icd_branch = nn.Linear(icd_ftrs, min_feat)
        elif demo_ftrs != None and icd_ftrs == None:
            min_feat = min(model_ftrs, demo_ftrs)
            self.demo_branch = nn.Linear(demo_ftrs, min_feat)
        elif demo_ftrs == None and icd_ftrs != None:
            min_feat = min(model_ftrs, icd_ftrs)
            self.icd_branch = nn.Linear(icd_ftrs, min_feat)

        if demo_ftrs != None and icd_ftrs != None:
            self.fused_output = nn.Linear(min_feat*3, num_classes)
        else:
            self.fused_output = nn.Linear(min_feat*2, num_classes)

    def forward(self, image, demo_feat=None, icd_feat=None):
        img_ft = self.image_branch(image)  # shape: 32, 1024, 4, 4
        if demo_feat != None and icd_feat != None:
            demo_ft = self.demo_branch(demo_feat)
            icd_ft = self.icd_branch(icd_feat)
            fused_ft = torch.cat([img_ft, torch.squeeze(demo_ft), torch.squeeze(icd_ft)], 1)
        elif demo_feat != None and icd_feat == None:
            demo_ft = self.demo_branch(demo_feat)
            fused_ft = torch.cat([img_ft, torch.squeeze(demo_ft)], 1)
        elif demo_feat == None and icd_feat != None:
            icd_ft = self.icd_branch(icd_feat)
            fused_ft = torch.cat([img_ft, torch.squeeze(icd_ft)], 1)

        return self.fused_output(fused_ft)


class Fusion_Modelv1(nn.Module):
    def __init__(self, model, model_ftrs, text_ftrs, num_classes):
        """model is the actual model with the linear output being the same size as linear input
        model_ftrs is the length of the model feature vector
        text_ftrs is the length of the input text features"""
        super(Fusion_Modelv1, self).__init__()
        # find the smaller features
        min_feat = min(model_ftrs, text_ftrs)
        self.image_branch = nn.Sequential(
            model, nn.Linear(model_ftrs, min_feat)
        )
        # creates a linear layer to make it the same size as the model output
        self.text_branch = nn.Linear(text_ftrs, min_feat)
        # average pool
        self.fused_output = nn.Linear(min_feat*2, num_classes)

    def forward(self, image, text_feat):
        img_ft = self.image_branch(image)  # shape: 32, 1024, 4, 4
        txt_ft = self.text_branch(text_feat)
        fused_ft = torch.cat([img_ft, torch.squeeze(txt_ft)], 1)
        return self.fused_output(fused_ft)


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
    feature_extract = params['feature_extract']
    use_pretrained = params['use_pretrained']
    fusion = params['fusion_model']  # fusion1 or fusion2 for different fusion models
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

# ---- Cross Attention Block ----
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, embed_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(dim_q, embed_dim)
        self.key_proj = nn.Linear(dim_kv, embed_dim)
        self.value_proj = nn.Linear(dim_kv, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, kv):
        # q: [B, Dq], kv: [B, Dkv]
        q = q.unsqueeze(1)   # [B, 1, Dq]
        kv = kv.unsqueeze(1) # [B, 1, Dkv]

        q_proj = self.query_proj(q)
        k_proj = self.key_proj(kv)
        v_proj = self.value_proj(kv)

        attn_out, _ = self.attn(q_proj, k_proj, v_proj)
        out = self.out_proj(attn_out).squeeze(1)  # [B, embed_dim]
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, embed_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(dim_q, embed_dim)
        self.key_proj   = nn.Linear(dim_kv, embed_dim)
        self.value_proj = nn.Linear(dim_kv, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        q = q.unsqueeze(1)   # [B,1,Dq]
        kv = kv.unsqueeze(1) # [B,1,Dkv]

        q_proj = self.query_proj(q)
        k_proj = self.key_proj(kv)
        v_proj = self.value_proj(kv)

        attn_out, _ = self.attn(q_proj, k_proj, v_proj)   # [B,1,E]
        attn_out = self.out_proj(attn_out)
        
        # residual + norm
        out = self.norm(q_proj + self.dropout(attn_out))
        return out.squeeze(1)  # [B,E]

class BiCrossAttentionBlock(nn.Module):
    def __init__(self, dim_tab, dim_img, embed_dim=128, num_heads=2, dropout=0.1):
        super().__init__()
        # Tab → Img
        self.tab2img = CrossAttentionBlock(dim_q=dim_tab, dim_kv=dim_img,
                                           embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # Img → Tab
        self.img2tab = CrossAttentionBlock(dim_q=dim_img, dim_kv=dim_tab,
                                           embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, tab, img):
        attended_img = self.tab2img(tab, img)  # tab queries image
        attended_tab = self.img2tab(img, tab)  # image queries tab
        return attended_tab, attended_img

# ---- 2D ECG image branch ----
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.seq(x)

class ECGImageCNN(nn.Module):

    def __init__(self, in_channels=1, dropout=0.2):
        super().__init__()
        chs = [64, 128, 256, 512]
        self.s1 = nn.Sequential(ConvBlock(in_channels, chs[0]),
                                nn.MaxPool2d(2),
                                nn.Dropout(0.2))
        self.s2 = nn.Sequential(ConvBlock(chs[0], chs[1]),
                                nn.MaxPool2d(2),
                                nn.Dropout(0.2))
        self.s3 = nn.Sequential(ConvBlock(chs[1], chs[2]),
                                nn.MaxPool2d(2),
                                nn.Dropout(0.2))
        self.s4 = nn.Sequential(ConvBlock(chs[2], chs[3]),
                                nn.MaxPool2d(2))

        self.pool = nn.AdaptiveAvgPool2d(1)  
        self.proj = nn.Linear(chs[-1], 256)



    def forward(self, x):
        x = self.s1(x); x = self.s2(x); x = self.s3(x); x = self.s4(x)
        x = self.pool(x).flatten(1) 
        x = self.proj(x)   
        return x 


# ---- 1D (tabular/EHR) branch ----
class TabularMLP(nn.Module):

    def __init__(self, n_features=14, dropout=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(n_features, 64)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, n_features)
        x = self.bn(x)
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.act2(self.fc2(x))
        return x


#---- Fusion head ----
class ECGFusionModel(nn.Module):
    def __init__(self, n_features=14, img_in_channels=3,
                 tab_dropout=0.20, img_dropout=0.01, head_dropout=0.01):
        super().__init__()
        self.tab = TabularMLP(n_features=n_features, dropout=tab_dropout)
        self.img = ECGImageCNN(in_channels=img_in_channels, dropout=img_dropout)

        fusion_in = 32 + 256  # tab + img features
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 32),     nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
        self.ln_img = nn.LayerNorm(256)
        self.ln_tab = nn.LayerNorm(32)

    def forward(self, tab_x, img_x):

        t = self.tab(tab_x)
        v = self.img(img_x)

        
        v = self.ln_img(v)
        t = self.ln_tab(t)
        z = torch.cat([t, v], dim=1)
        logit = self.head(z)
        return logit

class EchoECGCoAttentionClassifier(nn.Module):
    def __init__(self,
                 echo_encoder,        # frozen encoder producing 2048-d
                 ecg_encoder,         # frozen encoder producing 1024-d
                 embed_dim=1024,      # final shared dimension
                 num_heads=4,
                 hidden_dim=512,
                 num_classes=1,
                 dropout=0.1):
        super().__init__()

        # Store encoders and freeze them
        self.echo_encoder = echo_encoder
        self.ecg_encoder = ecg_encoder

        for p in self.echo_encoder.parameters():
            p.requires_grad = False
        for p in self.ecg_encoder.parameters():
            p.requires_grad = False

        # Projection for echo embedding: 2048 → 1024
        # ECG already 1024 → stays the same
        self.echo_proj = nn.Linear(2048, embed_dim)
        self.ecg_proj = nn.Linear(1024, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Cross-Attention (bidirectional)
        # echo attends to ecg, and ecg attends to echo
        self.attn_echo_to_ecg = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ecg_to_echo = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Classification Head
        self.fc = nn.Linear(embed_dim, num_classes)


    def forward(self, echo_img, ecg_img):
        """
        echo_img : [B, C, H, W]
        ecg_img  : [B, C, H, W]
        """

        B = echo_img.size(0)

        # Encode images (frozen)
        #with torch.no_grad():
        echo_embed = self.echo_encoder(echo_img)   # [B, 2048]
        ecg_embed  = self.ecg_encoder(ecg_img)     # [B, 1024]

        # Project echo → 1024 so both are equal dimension
        echo_embed = self.echo_proj(echo_embed)        # [B, 1024]
        #ecg_embed  = self.ecg_proj(ecg_embed)          # [B, 1024]

        # Add sequence dimension for multihead attention
        echo_seq = echo_embed.unsqueeze(1)   # [B, 1, 1024]
        ecg_seq  = ecg_embed.unsqueeze(1)    # [B, 1, 1024]

        # Cross Attention (Echo ↔ ECG)

        # Echo queries ECG
        attn_echo_out, _ = self.attn_echo_to_ecg(
            echo_seq, ecg_seq, ecg_seq
        )
        
        # ECG queries Echo
        attn_ecg_out, _ = self.attn_ecg_to_echo(
            ecg_seq, echo_seq, echo_seq
        )

        # Fuse them (average)
        fused = 0.5 * (attn_echo_out + attn_ecg_out)   # [B, 1, 1024]

        # Transformer Encoder
        x = self.transformer(fused)   # [B, 1, 1024]
        x = x.squeeze(1)              # [B, 1024]

        # Classification Head
        logits = self.fc(x)           # [B, num_classes]

        return logits
