import torch #imports pytorch core (tensors, autograd, cuda)
import torch.nn as nn #imports layer classes (conv2d, batchnorm2d, linear, module)
import torch.nn.functional as F #imports functional ops (relu, pooling) used in forward

class BasicBlock(nn.Module): #custom layer type: conv-bn-relu, conv-bn, add skip, relu
    def __init__(self, in_ch, out_ch, stride=1): #in_ch/out_ch; stride controls downsampling (1 = same size, 2 = half size)
        super().__init__() #calls nn.module constructor so parameters are tracked

        #1st convolution layer: in_ch = 3 at 1st, out_ch = # of filters, kernel = 3x3, uses padding to preserve size if stride =1
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3,
            stride=stride,
            padding=1, #adds 1-pixelborder of 0s on all 4 sides before convolution
            bias=False #bias not needed because batchnorm has learnable shift (beta), if true waste memory and computation
        ) #first 3x3 conv learns local features and can downsample when needed

        #normalizes conv1 output per channel via mean and variance; improves training stability; leanable parameters gamma (scale) and beta (shift) 
        self.bn1 = nn.BatchNorm2d(out_ch) 

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3,
            stride=1, #no downsample in second conv (if needed happens in conv1)
            padding=1, #preserves  size for 3x3
            bias=False #no bias because bn follows
        ) #second 3x3 conv refines features at same resolution

        self.bn2 = nn.BatchNorm2d(out_ch) #normalizes conv2 output

        #out=main_path(x)+shortcut(x), need same shape (batch size, h, w, channels) for addition
        self.shortcut = nn.Identity() #output = input if in_ch out_ch same size; used as a skip connection for gradient flow in backpropagation to learn residuals instead of full transforms (easier to optimize)
        if stride != 1 or in_ch != out_ch: #if downsampling or channel size mismatch
            self.shortcut = nn.Sequential( #chains 1x1 conv (for channel mixing) and bn to match dimensions for addition
                nn.Conv2d(
                    in_ch, out_ch, 1,
                    stride=stride, #matches the downsampling of conv1 so spatial sizes align
                    bias=False #bias not needed since bn follows
                ), 
                nn.BatchNorm2d(out_ch) #keeps shortcut scale comparable to main branch for stable addition
            )

    def forward(self, x): #forward pass of one residual block
        out = self.conv1(x) #applies first convolution
        out = self.bn1(out) #batchnorm after conv1
        out = F.relu(out, inplace=True) #relu nonlinearity; inplace=true saves memory by modifying tensor in place

        out = self.conv2(out) #applies second convolution
        out = self.bn2(out) #batchnorm after conv2

        out = out + self.shortcut(x) #residual addition: main branch + skip branch (shapes must match)
        out = F.relu(out, inplace=True) #relu after addition (standard resnet ordering)
        return out #returns block output

class CIFARResNet(nn.Module): #resnet for cifar: stem conv, 4 stages of residual blocks, global average pool, linear classifier
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, dropout=0.0): #num_blocks: number of BasicBlocks in each stage (resnet18 style); num_classes=10 for cifar-10; disable dropout.
        super().__init__() #registers layers/params, calls nn.Module constructor
        self.in_ch = 64 #tracks current channel width while building stages

        #in_ch = 3 (rgb), out_ch = 64 filters, kernel=3x3, stride=1 (no downsample), padding=1 (preserves 32x32 size), no bias because bn follows; this is the stem conv that processes raw image pixels into initial features
        self.conv1 = nn.Conv2d(
            3, 64, 3,
            stride=1, #keeps 32x32 resolution (important for small cifar images)
            padding=1, #preserves spatial size for 3x3
            bias=False #bias not needed because bn follows
        ) 

        self.bn1 = nn.BatchNorm2d(64) #bn for stem output

        #each stage has 2 blocks
        self.layer1 = self._make_layer(64,  num_blocks[0], stride=1) #stage1: out_ch = 64, no downsample, stays 32x32
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2) #stage2: out_ch = 128, downsample to 16x16
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2) #stage3: out_ch = 256, downsample to 8x8
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2) #stage4: out_ch = 512, downsample to 4x4

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity() #dropout if >0, else do nothing (identity)
        self.fc = nn.Linear(512, num_classes) #final fully-connected layer maps 512-d feature to class logits (not probabilities)

    def _make_layer(self, out_ch, blocks, stride): #creates one stage (a sequence of residual blocks)
        layers = [BasicBlock(self.in_ch, out_ch, stride)] #first block may downsample using stride
        self.in_ch = out_ch #after first block, channel width is now out_ch
        for _ in range(1, blocks): #add remaining blocks (same shape)
            layers.append(BasicBlock(self.in_ch, out_ch, 1)) #stride=1 keeps size inside the stage
        return nn.Sequential(*layers) #packs blocks so forward runs them in order

    def forward(self, x): #full network forward pass
        x = self.conv1(x) #stem convolution
        x = self.bn1(x) #stem batchnorm
        x = F.relu(x, inplace=True) #stem relu; inplace saves memory

        x = self.layer1(x) #stage1 output: (batch,64,32,32); batch size, channels (feature maps), height, width
        x = self.layer2(x) #stage2 output: (batch,128,16,16)
        x = self.layer3(x) #stage3 output: (batch,256,8,8)
        x = self.layer4(x) #stage4 output: (batch,512,4,4)

        #reduces each channel to a single feature in image value, reduces parameters so less overfitting, forces network to learn global features instead of local
        x = F.adaptive_avg_pool2d(x, 1) #global average pooling to (batch,512,1,1) [output = 1 value]; adaptive works for any input size
        x = x.flatten(1) #flattens from dim=1 to get (batch,512), make it 2d (rows = batch, columns = features)
        x = self.drop(x) #applies dropout if enabled, during training randomly zeroes some features to prevent co-adaptation and improve generalization; during eval does nothing
        x = self.fc(x) #produces logits (batch, 10), raw scores for each class, linear layer without activation
        return x #returns logits for cross entropy loss

def build_model(): #factory function so train.py can call one function to construct the model
    return CIFARResNet(num_blocks=(2, 2, 2, 2), dropout=0.0) #resnet18-style; dropout disabled 