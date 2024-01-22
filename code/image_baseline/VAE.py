import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

#from torch.distributions import Normal, Independent
#from torch.distributions.kl import kl_divergence as KLD


#Residual down sampling block for the encoder
#Average pooling is used to perform the downsampling
class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size = 3, scale = 2, padding = 'same'):
        super(Res_down, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, kernel_size = kernel_size, stride = 1, padding = 'same')
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, kernel_size=kernel_size, stride = 1, padding = padding)
        self.BN2 = nn.BatchNorm2d(channel_out)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride = 1, padding = padding)

        self.MaxPool = nn.MaxPool2d(kernel_size=2, dilation=1, ceil_mode=False)
        
    def forward(self, x):
        skip = self.conv3(self.MaxPool(x)) # out = channel_out

        x = self.act1(self.BN1(self.conv1(x))) # out = channel_out // 2
        x = self.MaxPool(x)
        x = self.BN2(self.conv2(x)) # out = channel_out

        x = self.act2(x + skip)
        return x

    
#Residual up sampling block for the decoder
#Nearest neighbour is used to perform the upsampling
class Res_up(nn.Module):
    def __init__(self, channel_in, 
                 channel_out, 
                 scale = 2,
                 padding = 'same'):
        super(Res_up, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, padding='same')
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, padding=padding)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, padding=padding)
        self.act2 = nn.ReLU()
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        x = self.act1(self.BN1(self.conv1(x)))
        x = self.UpNN(x)  
        x = self.BN2(self.conv2(x))
        x = self.act2(x + skip)
        return x


class Encoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 32):
        super(Encoder, self).__init__()
        self.conv1 = Res_down(channels, ch)#64
        self.conv2 = Res_down(ch, 2*ch)#32
        self.conv3 = Res_down(2*ch, 4*ch)#16
        self.conv4 = Res_down(4*ch, 8*ch)#8
        self.conv5 = Res_down(8*ch, 8*ch, padding = 0)#4
        self.mlp = nn.Linear(in_features=14, out_features=1)
        self.conv_mu = nn.Conv2d(8*ch, z, kernel_size=1, stride = 4, padding = 0)#2
        self.conv_logvar = nn.Conv2d(8*ch, z, kernel_size = 1, stride = 4, padding = 0)#2

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x, Train = True):
        logger.debug(f'{Train=}')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.mlp(x)
        
        if Train:
            logger.debug(f'If Train: {x.shape=}')
            mu = self.conv_mu(x)
            log_var = self.conv_logvar(x)
            z = self.sample(mu, log_var)
            # z shape:
            ## torch.Size([128, 100, 1, 1])
            logger.debug(f'{z.shape=}')
            kl = torch.mean(-0.5*torch.sum(1+log_var-mu**2-log_var.exp()))
            logger.debug(f'{kl=}')
        else:
            z = self.conv_mu(x)
            kl = None
        return kl, z
    
#Decoder block
#Built to be a mirror of the encoder block
class Decoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 32):
        super(Decoder, self).__init__()
        self.dim_z = z
        self.mlp = nn.Linear(in_features=1, out_features=4)
        self.dropout = nn.Dropout2d(p = .3)
        self.conv1 = Res_up(z, ch*16)
        self.conv12 = Res_up(ch*16, ch*8)
        self.conv2 = Res_up(ch*8, ch*8)
        self.conv3 = Res_up(ch*8, ch*4)
        self.conv4 = Res_up(ch*4, ch*2)
        self.conv5 = Res_up(ch*2, ch)
        self.conv6 = Res_up(ch, ch//2)
        self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.conv1(x)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return self.act(x) 
    
#VAE network, uses the above encoder and decoder blocks 
class VAE(nn.Module):
    def __init__(self, device, channel_in=1, z = 32):
        super(VAE, self).__init__()
        self.encoder = Encoder(channel_in, z = z).to(device)
        self.decoder = Decoder(channel_in, z = z).to(device)

    def forward(self, x, Train = True):
        kl, z = self.encoder(x, Train)
        recon = self.decoder(z)
        return kl, recon