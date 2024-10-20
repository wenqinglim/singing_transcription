import torch
import torch.nn as nn
import torch.nn.functional as nnF
from huggingface_hub import PyTorchModelHubMixin


'''
Each convolutional layer is
preceded with batch normalization [21] and followed by a dropout
layer [22] with the dropout probability 0.25
'''

class PrintLayer(nn.Module):
    """
    Printing class for debugging
    """
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class CREPEModel(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="omgitsqing/CREPE_MIR-1K",
    pipeline_tag="transcription, pitch detection",
    license="mit",
    ):
    """The CREPE model"""
    def __init__(self, mult=4):
        super(CREPEModel, self).__init__()
        self.mult=mult

        self.model = nn.Sequential(            
            # input dim = 1024*1*1
            nn.Conv2d(in_channels=1, out_channels=self.mult*32, kernel_size=(512, 1), stride=(4, 1), padding=(254,0), bias=False), 
            #output dim = [(1024+2(0)-512)/4 + 1] x [(1+2(0)-1)/1 + 1] = 129x1x32
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.mult*32),
            nn.MaxPool2d(kernel_size=(2,1), padding=0), 
            nn.Dropout(0.25),
            #output size = [(129+2(0)-2)/2 + 1] x [(1+2(0)-1)/1 + 1] = 64x1x32
            # PrintLayer(),
            
            nn.Conv2d(in_channels=self.mult*32, out_channels=self.mult*4, kernel_size=(64, 1), stride=(1, 1), padding='same', bias=False), 
            #output size = [(64+2(0)-64)/1 + 1] x [(1+2(0)-1)/1 + 1] = 1x1x4
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.mult*4),
            nn.MaxPool2d(kernel_size=(2,1), padding=0), 
            nn.Dropout(0.25),
            #output size = [(60+2(0)-2)/2 + 1] x [(1+2(0)-1)/1 + 1] = 30x1x4
            # PrintLayer(),
            
            nn.Conv2d(in_channels=self.mult*4, out_channels=self.mult*4, kernel_size=(64, 1), stride=(1, 1), padding='same', bias=False), 
            #output size = [(30+2(0)-4)/1 + 1] x [(4+2(0)-3)/1 + 1] = 4x1x128
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.mult*4),
            nn.MaxPool2d(kernel_size=(2,1), padding=0), 
            nn.Dropout(0.25),
            #output size = [(5+2(0)-2)/2 + 1] x [(2+2(0)-1)/1 + 1] = 2x1x128
            # PrintLayer(),

            nn.Conv2d(in_channels=self.mult*4, out_channels=self.mult*4, kernel_size=(64, 1), stride=(1, 1), padding='same', bias=False), 
            #output size = [(7+2(0)-3)/1 + 1] x [(4+2(0)-3)/1 + 1] = 4x1x128
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.mult*4),
            nn.MaxPool2d(kernel_size=(2,1), padding=0), 
            nn.Dropout(0.25),
            #output size = [(5+2(0)-2)/2 + 1] x [(2+2(0)-1)/1 + 1] = 2x1x128
            # PrintLayer(),

            nn.Conv2d(in_channels=self.mult*4, out_channels=self.mult*8, kernel_size=(64, 1), stride=(1, 1), padding='same', bias=False), 
            #output size = [(7+2(0)-3)/1 + 1] x [(4+2(0)-3)/1 + 1] = 4x1x128
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.mult*8),
            nn.MaxPool2d(kernel_size=(2,1), padding=0), 
            nn.Dropout(0.25),
            #output size = [(5+2(0)-2)/2 + 1] x [(2+2(0)-1)/1 + 1] = 2x1x128
            # PrintLayer(),

            nn.Conv2d(in_channels=self.mult*8, out_channels=self.mult*16, kernel_size=(64, 1), stride=(1, 1), padding='same', bias=False), 
            #output size = [(7+2(0)-3)/1 + 1] x [(4+2(0)-3)/1 + 1] = 4x1x128
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.mult*16),
            nn.MaxPool2d(kernel_size=(2,1), padding=0), 
            nn.Dropout(0.25),
            #output size = [(5+2(0)-2)/2 + 1] x [(2+2(0)-1)/1 + 1] = 2x1x128
            # PrintLayer(),
            
            nn.Flatten(),
            nn.Linear(in_features=4*1*self.mult*16, out_features=410+1, bias=True),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)


    def forward(self, x):
        return self.model(x.unsqueeze(1))