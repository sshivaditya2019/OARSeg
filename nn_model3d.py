import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, IterableDataset








class UNet_down_block_3d(torch.nn.Module):
  def __init__(self, input_channels, output_channels, down_size):
    super(UNet_down_block_3d, self).__init__()
    self.conv1 = torch.nn.Conv3d(input_channels,output_channels,3,padding = 1)
    self.bn1 = torch.nn.BatchNorm3d(output_channels)
    self.conv2 = torch.nn.Conv3d(output_channels,output_channels,3,padding = 1)
    self.bn2 = torch.nn.BatchNorm3d(output_channels)
    self.conv3 = torch.nn.Conv3d(output_channels,output_channels,3,padding = 1)
    self.bn3 = torch.nn.BatchNorm3d(output_channels)
    self.max_pool = torch.nn.MaxPool3d(3,3)
    self.relu = torch.nn.ReLU()
    self.down_size = down_size

  def forward(self, x):
    if self.down_size:
      x = self.maxpool(x)
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    return x

class UNet_up_block_3d(torch.nn.Module):
  def __init__(self, prev_channels, input_channels, output_channels):
    super(UNet_up_block_3d, self).__init__()
    self.up_sampling = torch.nn.Upsample(scale_factor=2, mode = 'bilinear')
    self.conv1 = torch.nn.Conv3d(prev_channels + input_channels,output_channels, 3,padding =1)
    self.bn1 = torch.nn.BatchNorm3d(output_channels)
    self.conv2 = torch.nn.Conv3d(output_channels, output_channels, 3, padding = 1)
    self.bn2 = torch.nn.BatchNorm3d(output_channels)
    self.conv3 = torch.nn.Conv3d(output_channels,output_channels, 3, padding = 1)
    self.bn3 = torch.nn.BatchNorm3d(output_channels)
    self.relu = torch.nn.ReLU()

  def forward(sefl, prev_feature_map, x):
    x = self.up_sampling(x)
    x = torch.cat((x, prev_feature_map), dim = 1)
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    return x

class UNet_3d(torch.nn.Module):
  def __init__(self):
    super(UNet_3d, self).__init__()
    self.down_block1 = UNet_down_block_3d(8,16, False)
    self.down_block2 = UNet_down_block_3d(16,32, True)
    self.down_block3 = UNet_down_block_3d(32,64, True)
    self.down_block4 = UNet_down_block_3d(64,128, True)
    self.down_block5 = UNet_down_block_3d(128,256, True)
    self.down_block6 = UNet_down_block_3d(256,512, True)
    self.down_block7 = UNet_down_block_3d(512,1024, True)
    
    self.mid_conv1 = torch.nn.Conv3d(1024, 1024, 3, padding=1)
    self.bn1 = torch.nn.BatchNorm3d(1024)
    self.mid_conv2 = torch.nn.Conv3d(1024, 1024, 3, padding=1)
    self.bn2 = torch.nn.BatchNorm3d(1024)
    self.mid_conv3 = torch.nn.Conv3d(1024, 1024, 3, padding=1)
    self.bn3 = torch.nn.BatchNorm3d(1024)

    self.up_block1 = UNet_up_block_3d(512,1024,512)
    self.up_block2 = UNet_up_block_3d(256,512,256)
    self.up_block3 = UNet_up_block_3d(128,256,128)
    self.up_block4 = UNet_up_block_3d(64,128,64)
    self.up_block5 = UNet_up_block_3d(32,64,32)
    self.up_block6 = UNet_up_block_3d(16,32,16)

    self.last_conv1 = torch.nn.Conv3d(16,16,3, padding = 1)
    self.last_bn = torch.nn.BatchNorm3d(16)
    self.last_conv2 = torch.nn.Conv3d(16,8,1,padding = 1)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x
    

