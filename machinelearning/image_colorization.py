import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision.transforms import transforms
import math
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()
        self.conv64_1=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2)
        self.norm1=nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        self.norm5 = nn.BatchNorm2d(256)
        self.norm6 = nn.BatchNorm2d(512)
        self.norm7 = nn.BatchNorm2d(512)
        self.conv64_2=Conv2dSame(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.conv128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
        self.conv128_2 = Conv2dSame(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.conv256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
        self.conv256_2 = Conv2dSame(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.conv512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same")
        self.conv512_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same")
        self.conv256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding="same")
        self.conv128 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding="same")
        self.upsample=nn.UpsamplingNearest2d(scale_factor=2)
        self.conv64=nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same")
        self.conv32 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.conv16 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding="same")
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()


    def forward(self, x):
        x=self.conv64_1(x)
        #x=self.conv64_2(self.relu(x))
        x=self.norm1(x)
        x=self.conv128_1(self.relu(x))
        x = self.norm2(x)
        x = self.conv128_2(self.relu(x))
        x = self.norm3(x)
        x = self.conv256_1(self.relu(x))
        x = self.norm4(x)
        x = self.conv256_2(self.relu(x))
        x = self.norm5(x)
        x = self.conv512(self.relu(x))
        x = self.norm6(x)
        x = self.conv512_2(self.relu(x))
        x = self.norm7(x)
        x=self.conv256(self.relu(x))
        x = self.conv128(self.relu(x))
        x=self.upsample(self.relu(x))
        x = self.conv64(x)
        x = self.upsample(self.relu(x))
        x = self.conv32(x)
        x = self.conv16(self.relu(x))
        x = self.conv2(self.relu(x))
        return self.upsample(self.tanh(x))




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Colorizer().to(device=device)
print(summary(model.cuda(), (1, 256, 256)))


def prepare_img(img):
    img=img.resize((128, 128), Image.BILINEAR)
    img = np.asarray(img, dtype=np.float32)
    size=img.shape
    lab = rgb2lab(img/255)
    X, Y=lab[:,:,0], lab[:, :, 1:]
    cur1=np.zeros((size[0], size[1], 3))

    Y/=128

    X=X.reshape(1, size[0], size[1])
    Y = Y.reshape(2, size[0], size[1])
    return torch.tensor(X), torch.tensor(Y), size, cur1

def increase_number(num, tens):
    lst=[]
    tens=tens.numpy().copy()
    for i in range(num):

        lst.append(tens)
    return torch.tensor(np.array(lst))
img=Image.open("cats400 (1).jpg")

X, Y, size, cur1=prepare_img(img)
X=increase_number(32, X)
Y=increase_number(32, Y)

X=X.to(device=device)
Y=Y.to(device=device)
Loss=nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.0015)

for i in range(1000):
    optimizer.zero_grad()
    Y_pred=model(X)
    loss=Loss(Y_pred, Y)
    loss.backward()
    optimizer.step()
    print(f"epoch={i}, loss={loss}")
with torch.no_grad():
    # Y_pred = model(X).squeeze(0)
    Y_pred = model(X)[0]
    Y_pred=Y_pred.to(device="cpu").numpy()
    Y_pred=Y_pred.reshape(size[0], size[1], 2)*128
    cur=np.zeros((size[0], size[1], 3))
    X=X.to(device="cpu").numpy()
    cur[:,:,0]=np.clip(X[0][0, :, :], 0, 100)
    cur[:, :, 1]=Y_pred[:, :, 0]
    cur[:, :, 2] = Y_pred[:, :, 1]
    plt.subplot(1, 2, 1)
    plt.imshow(np.asarray(img, dtype=np.float32)/255)
    plt.subplot(1, 2, 2)
    plt.imshow(lab2rgb(cur))
    plt.show()







