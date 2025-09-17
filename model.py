import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim,image_dim, image_channel,kernel_size,stride,padding):
        super(Generator, self).__init__()
        
        self.tsconv1 = nn.ConvTranspose2d(latent_dim, image_dim * 8, kernel_size=8, stride=2, padding=2, bias=False)
        self.norm1 = nn.BatchNorm2d(image_dim * 8)
        
        self.tsconv2 = nn.ConvTranspose2d(image_dim * 8, image_dim * 4, kernel_size=6, stride=2, padding=2, bias=False)
        self.norm2 = nn.BatchNorm2d(image_dim * 4)
        
        self.tsconv3 = nn.ConvTranspose2d(image_dim * 4, image_dim * 2, kernel_size=5, stride=2, padding=2, bias=False)
        self.norm3 = nn.BatchNorm2d(image_dim * 2)
        
        self.tsconv4 = nn.ConvTranspose2d(image_dim * 2, image_dim, kernel_size=5, stride=2, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(image_dim)
        
        self.output = nn.ConvTranspose2d(image_dim, image_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        

    def forward(self, z,debug = False):
        if debug:
            print(f"input dim: {z.shape}")
            
        z = self.tsconv1(z)
        z = self.norm1(z)
        z = F.leaky_relu(z,0.15)
        
        if debug:
            print(f"layer1 dim: {z.shape}")

        
        z = self.tsconv2(z)
        z = self.norm2(z)
        z = F.leaky_relu(z,0.15)
        if debug:
            print(f"layer2 dim: {z.shape}")
        
        z = self.tsconv3(z)
        z = self.norm3(z)
        z = F.leaky_relu(z,0.15)
        if debug:
            print(f"layer3 dim: {z.shape}")
        
        
        z = self.tsconv4(z)
        z = self.norm4(z)
        z = F.leaky_relu(z,0.15)
        if debug:
            print(f"layer4 dim: {z.shape}")
                       
        z = self.output(z)

        z = F.sigmoid(z)
    
        if debug:
            print(f"final layer dim: {z.shape}")
        return z
    
    
    
    
    
    
    
    
    
class Discriminator(nn.Module):
    def __init__(self, image_channel):
        super().__init__( )

        self.conv1 = nn.Conv2d(image_channel,32,kernel_size=(3,3),stride=1, padding=1)  # 1 - black or white, 32- filters, 3,3 - matrix 
        self.pool1 = nn.MaxPool2d(kernel_size=2)
            
            
        self.conv2=    nn.Conv2d(32,64,kernel_size=(3,3),stride=1, padding=1)  # 1 - black or white, 32- filters, 3,3 - matrix 
        self.pool2 = nn.MaxPool2d(kernel_size=2)
            
            
        self.conv3=     nn.Conv2d(64,128,kernel_size=(3,3),stride=1, padding=1)  # 1 - black or white, 32- filters, 3,3 - matrix 
        self.pool3 = nn.MaxPool2d(kernel_size=2)
            
        self.flatten = nn.Flatten()
        
        self.ff1 = nn.Linear(8192,516)
        
            
        self.output=   nn.Linear(516  , 1      ) # 10 outputs for each classes
            
        self.dropout = nn.Dropout(0.5)    
            
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x,0.1)
        x=self.pool1(x)
        
        x = self.conv2(x)
        x = F.leaky_relu(x,0.1) 
        x=self.pool2(x)     
        
        
        x = self.conv3(x)
        x = F.leaky_relu(x,0.1)     
        x=self.pool3(x)     
        
        x = self.flatten(x)
        
        x = self.dropout(x)
        
        #print(x.shape)
        
        x = self.ff1(x)
        x = F.leaky_relu(x,0.1) 
        
        x = self.output(x)
        
        x = F.sigmoid(x)
        
        #print(x)
        
        return x
        