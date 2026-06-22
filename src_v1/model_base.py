"""Defined functions for general classification model building
Essentially the functions from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
import torch
import torch.nn as nn
import math
from torchvision import models
from torchvision.models import resnet
import torch.nn.functional as F
import numpy as np



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

        #ecg_input = ecg_input.unsqueeze(0)

        ecg_features = self.ecg_cnn(ecg_input) 
        ecg_features = self.pool(ecg_features)
        ecg_features = torch.flatten(ecg_features, 1) 
        ecg_features = self.ecg_mlp(ecg_features) 

        #echo_down_feats = echo_down_feats.unsqueeze(0)  # shape becomes [1, 8]
        concat_feats = torch.concat([ecg_features.float(), echo_down_feats.float()], dim=1)

        # feed concatenated features into final MLP thats output has 3 values (softmax in training base code)
        output = self.final_mlp(concat_feats)

        return output

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, embed_dim, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.query_proj = nn.Linear(dim_q, embed_dim)
        self.key_proj = nn.Linear(dim_kv, embed_dim)
        self.value_proj = nn.Linear(dim_kv, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, kv):
        # q: [B, Dq], kv: [B, Dkv]
        # Add sequence dimension (L=1)
        q = q.unsqueeze(1)  
        kv = kv.unsqueeze(1)

        q_proj = self.query_proj(q)
        k_proj = self.key_proj(kv)
        v_proj = self.value_proj(kv)

        attn_out, _ = self.attn(q_proj, k_proj, v_proj)
        out = self.out_proj(attn_out).squeeze(1)  # back to [B, D]
        return out

# class ECGEcho_Fusion(nn.Module):
#     def __init__(self, ecg_encoder):
#         super(ECGEcho_Fusion, self).__init__()

#         # Echo encoder
#         ap4_model = torch.load('./models/4class_echo_resnext101_best.pth')
#         self.echo_cnn = nn.Sequential(*list(ap4_model.children())[:-1])  

#         # ECG encoder
#         self.ecg_cnn = ecg_encoder.features
#         self.pool = nn.AdaptiveAvgPool2d((1, 1)) 

#         # Dim reduction
#         self.echo_fc = nn.Linear(2048, 128)  
#         self.ecg_fc = nn.Linear(1024, 128)  

#         # Cross-attention
#         self.cross_attn_echo = CrossAttentionBlock(128, 128, 128, num_heads=4)
#         self.cross_attn_ecg = CrossAttentionBlock(128, 128, 128, num_heads=4)

#         # Final classifier
#         self.final_mlp = nn.Sequential(
#             nn.Linear(128 * 2, 64),  # fused [echo→ecg, ecg→echo]
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 4)
#         )

#     def forward(self, echo, ecg):
#         # Echo features
#         echo = self.echo_cnn(echo)
#         echo = echo.view(echo.size(0), -1)  
#         echo = self.echo_fc(echo)

#         # ECG features
#         ecg = self.ecg_cnn(ecg)
#         ecg = self.pool(ecg).view(ecg.size(0), -1)
#         ecg = self.ecg_fc(ecg)

#         # Cross attention
#         echo_attn = self.cross_attn_echo(echo, ecg)  # echo queries ecg
#         ecg_attn = self.cross_attn_ecg(ecg, echo)    # ecg queries echo

#         fused = torch.cat([echo_attn, ecg_attn], dim=1)
#         out = self.final_mlp(fused)
#         return out
