import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class galNetwork(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32

        # Network architecture is given in run.md
        # Input images are greyscale 256 x 256
        self.conv1 = nn.Conv2d(1, 2, 5, bias=False, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(2, 4, 5, bias=False, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(4, 8, 5, bias=False, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
        self.pool4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.pool5 = nn.MaxPool2d(2)
        self.bn5 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.pool6 = nn.MaxPool2d(2)
        self.bn6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(64*4*4, 256, bias=False)
        self.fc2 = nn.Linear(256, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool3(F.leaky_relu(self.bn3(x)))
        x = self.conv4(x)
        x = self.pool4(F.leaky_relu(self.bn4(x)))
        x = self.conv5(x)
        x = self.pool5(F.leaky_relu(self.bn5(x)))
        x = self.conv6(x)
        x = self.pool6(F.leaky_relu(self.bn6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class galNetwork_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 2, 5, bias=False, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(2, 4, 5, bias=False, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(4, 8, 5, bias=False, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
        self.pool4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.pool5 = nn.MaxPool2d(2)
        self.bn5 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.pool6 = nn.MaxPool2d(2)
        self.bn6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(64*4*4, 256, bias=False)
        self.fc2 = nn.Linear(256, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 64, 5, bias=False, padding=2)
        self.bn7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        self.bn8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=2)
        self.bn9 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(16, 8, 5, bias=False, padding=2)
        self.bn10 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(8, 4, 5, bias=False, padding=2)
        self.bn11 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(4, 2, 5, bias=False, padding=2)
        self.bn12 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
        self.deconv7 = nn.ConvTranspose2d(2, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool3(F.leaky_relu(self.bn3(x)))
        x = self.conv4(x)
        x = self.pool4(F.leaky_relu(self.bn4(x)))
        x = self.conv5(x)
        x = self.pool5(F.leaky_relu(self.bn5(x)))
        x = self.conv6(x)
        x = self.pool6(F.leaky_relu(self.bn6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn7(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn8(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn9(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.bn10(x)), scale_factor=2)
        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.bn11(x)), scale_factor=2)
        x = self.deconv6(x)
        x = F.interpolate(F.leaky_relu(self.bn12(x)), scale_factor=2)
        x = self.deconv7(x)
        x = torch.sigmoid(x)

        return x

def build_network():
    """Builds the neural network."""
    net = galNetwork()

    return net


def build_autoencoder():
    """Builds the corresponding autoencoder network."""
    ae_net = galNetwork_Autoencoder()

    return ae_net


