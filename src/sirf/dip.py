import torch
import torch.nn as nn
import math

class Block(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(Block, self).__init__()
        self.block = nn.Sequential(
                        nn.Conv3d(ch_in, ch_out, 3, stride=stride, padding=1),
                        nn.BatchNorm3d(ch_out),
                        nn.LeakyReLU(inplace=True))
    def forward(self, x):
        return self.block(x)

class PETUNet(nn.Module):
    def __init__(self, ch = 16, size = (90,128,128)):
        super(PETUNet, self).__init__()
        # Encoder
        self.block1 = Block(ch_in = 1, ch_out = ch, stride = 1)
        self.block2 = Block(ch_in = ch, ch_out = ch, stride = 1)
        self.stridedblock1 = Block(ch_in = ch, ch_out = ch, stride = 2)
        
        self.block3 = Block(ch_in = ch, ch_out = 2*ch, stride = 1)
        self.block4 = Block(ch_in = 2*ch, ch_out = 2*ch, stride = 1)
        self.stridedblock2 = Block(ch_in = 2*ch, ch_out = 2*ch, stride = 2)

        self.block5 = Block(ch_in = 2*ch, ch_out = 2*2*ch, stride = 1)
        self.block6 = Block(ch_in = 2*2*ch, ch_out = 2*2*ch, stride = 1)
        self.stridedblock3 = Block(ch_in = 2*2*ch, ch_out = 2*2*ch, stride = 2)

        self.block7 = Block(ch_in = 2*2*ch, ch_out = 2*2*2*ch, stride = 1)
        self.block8 = Block(ch_in = 2*2*2*ch, ch_out = 2*2*2*ch, stride = 1)

        # Decoder
        self.block9 = Block(ch_in = 2*2*2*ch, ch_out = 2*2*ch, stride = 1)
        self.upsample1 = nn.Upsample(size=tuple(math.ceil(si/4) for si in size), mode='trilinear', align_corners=True)
        
        self.block10 = Block(ch_in = 2*2*2*ch, ch_out = 2*2*ch, stride = 1)
        self.block11 = Block(ch_in = 2*2*ch, ch_out = 2*ch, stride = 1)
        self.upsample2 = nn.Upsample(size=tuple(math.ceil(si/2) for si in size), mode='trilinear', align_corners=True)

        self.block12 = Block(ch_in = 2*2*ch, ch_out = 2*ch, stride = 1)
        self.block13 = Block(ch_in = 2*ch, ch_out = ch, stride = 1)
        self.upsample3 = nn.Upsample(size=size, mode='trilinear', align_corners=True)
       
        self.block14 = Block(ch_in = 2*ch, ch_out = ch, stride = 1)
        self.block15 = Block(ch_in = ch, ch_out = 1, stride = 1)

        self.output = nn.ReLU()

    def forward(self, x):
        # Encoder
        x = self.block1(x)
        x_skip_1 = self.block2(x)
        x = self.stridedblock1(x_skip_1)
        
        x = self.block3(x)
        x_skip_2 = self.block4(x)
        x = self.stridedblock2(x_skip_2)

        x = self.block5(x)
        x_skip_3 = self.block6(x)
        x = self.stridedblock3(x_skip_3)
        
        x = self.block7(x)
        x = self.block8(x)
        
        # Decoder
        x = self.block9(x)
        x = self.upsample1(x)
        
        x = torch.cat([x, x_skip_3], dim=1)
        x = self.block10(x)
        x = self.block11(x)
        x = self.upsample2(x)
        
        x = torch.cat([x, x_skip_2], dim=1)
        x = self.block12(x)
        x = self.block13(x)
        x = self.upsample3(x)
        
        x = torch.cat([x, x_skip_1], dim=1)
        x = self.block14(x)
        x = self.block15(x)
        return self.output(x)
    
if __name__ == "__main__":
    model = PETUNet()
    print(sum(p.numel() for p in model.parameters()))