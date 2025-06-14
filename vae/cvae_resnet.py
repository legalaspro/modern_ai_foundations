
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    """Basic ResNet block with optional conditioning"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, num_classes=None):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
        # Conditional batch normalization (optional)
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, out_channels * 2)
    
    def forward(self, x, class_label=None):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        
        # Apply class conditioning if available
        if class_label is not None and self.num_classes is not None:
            class_emb = self.class_embed(class_label)  # [batch, out_channels * 2]
            gamma, beta = class_emb.chunk(2, dim=1)  # [batch, out_channels] each
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [batch, out_channels, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)   # [batch, out_channels, 1, 1]
            out = gamma * out + beta
        
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetTransposeBlock(nn.Module):
    """Transposed ResNet block for decoder"""
    def __init__(self, in_channels, out_channels, stride=1, upsample=None, num_classes=None):
        super(ResNetTransposeBlock, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                                       stride=stride, padding=1, output_padding=stride-1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, 
                                       stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.upsample = upsample
        
        # Conditional batch normalization
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, out_channels * 2)
    
    def forward(self, x, class_label=None):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        
        # Apply class conditioning
        if class_label is not None and self.num_classes is not None:
            class_emb = self.class_embed(class_label)
            gamma, beta = class_emb.chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            out = gamma * out + beta
        
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.upsample is not None:
            identity = self.upsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetEncoder(nn.Module):
    """ResNet-based encoder for CVAE"""
    def __init__(self, input_channels=3, latent_dim=128, num_classes=10):
        super(ResNetEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Class embedding for conditioning
        self.class_embedding = nn.Embedding(num_classes, 512)
        
        # Latent space projections
        self.fc_mu = nn.Linear(512 + 512, latent_dim)  # +512 for class embedding
        self.fc_logvar = nn.Linear(512 + 512, latent_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample, self.num_classes))
        
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, num_classes=self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, class_label):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers with class conditioning
        for layer in self.layer1:
            x = layer(x, class_label)
        for layer in self.layer2:
            x = layer(x, class_label)
        for layer in self.layer3:
            x = layer(x, class_label)
        for layer in self.layer4:
            x = layer(x, class_label)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Concatenate with class embedding
        class_emb = self.class_embedding(class_label)
        x = torch.cat([x, class_emb], dim=1)
        
        # Latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    

class ResNetDecoder(nn.Module):
    """ResNet-based decoder for CVAE"""
    def __init__(self, latent_dim=128, output_channels=3, num_classes=10):
        super(ResNetDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.num_classes = num_classes
        
        # Class embedding for conditioning
        self.class_embedding = nn.Embedding(num_classes, 512)
        
        # Initial projection from latent space
        self.fc = nn.Linear(latent_dim + 512, 512 * 2 * 2)  # +512 for class embedding
        
        # Transposed ResNet layers
        self.layer1 = self._make_layer(512, 256, 2, stride=2)
        self.layer2 = self._make_layer(256, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 64, 2, stride=2)
        self.layer4 = self._make_layer(64, 32, 2, stride=2)
        
        # Final convolution to output channels
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()  # Output in [-1, 1]
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        upsample = None
        if stride != 1 or in_channels != out_channels:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, 
                                  stride=stride, output_padding=stride-1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(ResNetTransposeBlock(in_channels, out_channels, stride, upsample, self.num_classes))
        
        for _ in range(1, blocks):
            layers.append(ResNetTransposeBlock(out_channels, out_channels, num_classes=self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, z, class_label):
        # Concatenate latent code with class embedding
        class_emb = self.class_embedding(class_label)
        z = torch.cat([z, class_emb], dim=1)
        
        # Project to feature map
        x = self.fc(z)
        x = x.view(x.size(0), 512, 2, 2)  # Reshape to [batch, 512, 2, 2]
        
        # Transposed ResNet layers with class conditioning
        for layer in self.layer1:
            x = layer(x, class_label)
        for layer in self.layer2:
            x = layer(x, class_label)
        for layer in self.layer3:
            x = layer(x, class_label)
        for layer in self.layer4:
            x = layer(x, class_label)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.tanh(x)
        
        return x


class ResNetCVAE(nn.Module):
    """Complete ResNet-based Conditional VAE"""
    def __init__(self, input_channels=3, latent_dim=128, num_classes=10):
        super(ResNetCVAE, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder and Decoder
        self.encoder = ResNetEncoder(input_channels, latent_dim, num_classes)
        self.decoder = ResNetDecoder(latent_dim, input_channels, num_classes)
        
    def encode(self, x, class_label):
        """Encode input to latent parameters"""
        return self.encoder(x, class_label)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, class_label):
        """Decode latent code to reconstruction"""
        return self.decoder(z, class_label)
    
    def forward(self, x, class_label):
        """Full forward pass"""
        mu, logvar = self.encode(x, class_label)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, class_label)
        return recon_x, mu, logvar
    
    def sample(self, num_samples, class_label, device):
        """Generate samples from the prior"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            if isinstance(class_label, int):
                class_label = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
            samples = self.decode(z, class_label)
            return samples
