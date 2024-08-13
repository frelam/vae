import sys
from matplotlib import pyplot as plt 
import torch
import torch.nn.backends
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
train_dataset = datasets.MNIST('./mnist', train=True, download=False, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# data, lable = next(iter(train_dataloader))


# sys.exit()
# data_dim = 2
hidden_dim = 200
batch_size = 100
sameple_size = 2
eps=1e-6
# data_point_size = 3
# data_size = 50000

class DecoderPerLayer(torch.nn.Module):
    def __init__(self, channals):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3,
                                      3,
                                      kernel_size=3,
                                      padding='same')
        # self.bn = torch.nn.InstanceNorm1d(num_features)
        self.bn = torch.nn.BatchNorm2d(3)
        # self.bn = torch.nn.LayerNorm(num_features)
        # self.gelu = torch.nn.GELU()
    def forward(self, x):
        output = self.conv2d(x)
        output = torch.nn.functional.leaky_relu(output, 0.1)
        output = x + output
        output = self.bn(output)
        return output


class DecoderLayer(torch.nn.Module):
    def __init__(self, layers_num, num_features):
        super().__init__()
        layers = []
        for i in range(layers_num):
            layers.append(DecoderPerLayer(num_features))
        self.layers = torch.nn.ModuleList(layers)
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

class EncoderPerLayer(torch.nn.Module):
    def __init__(self, channals):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(channals,
                                      channals,
                                      kernel_size=3,
                                      padding='same')
        # self.bn = torch.nn.InstanceNorm1d(num_features)
        self.bn = torch.nn.BatchNorm2d(channals)
        # self.bn = torch.nn.LayerNorm(num_features)
        # self.gelu = torch.nn.GELU()
    def forward(self, x):
        output = self.conv2d(x)
        output = torch.nn.functional.leaky_relu(output, 0.1)
        output = x + output
        output = self.bn(output)
        return output


class EncoderLayer(torch.nn.Module):
    def __init__(self, layers_num, channals):
        super().__init__()
        layers = []
        for i in range(layers_num):
            layers.append(EncoderPerLayer(channals))
        self.layers = torch.nn.ModuleList(layers)
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

class Encoder(torch.nn.Module):
    def __init__(self, layers_num, channal):
        super().__init__()
        self.layer = EncoderLayer(layers_num, channal)
        self.conv2d = torch.nn.Conv2d(1, 3, kernel_size=3, stride=2)
        self.channal = channal
        self.bn = torch.nn.BatchNorm2d(1)
        self.param5 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(507, hidden_dim * 2)))
    def forward(self, x):
        # x = torch.unsqueeze(x, 0)
        o1 = self.conv2d(x)
        o1 = self.layer(o1)
        o1_shape = o1.shape
        o1 = o1.reshape(o1_shape[0], 1, -1)
        o2 = torch.matmul(o1, self.param5)
        o2 = o2.reshape(o2.shape[0], 1, hidden_dim, 2)
        return o2


class Decoder(torch.nn.Module):
    def __init__(self, layers_num, num_feature):
        super().__init__()
        self.num_feature = num_feature
        self.layer = DecoderLayer(layers_num, num_feature)
        self.param0 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(hidden_dim, num_feature * num_feature * 3)))
        self.deconv = torch.nn.ConvTranspose2d(3, 1, 3)
    def forward(self, x):
        # x = torch.unsqueeze(x, 0)
        o0 = torch.matmul(x, self.param0)
        o0 = o0.reshape(o0.shape[0], 3, self.num_feature, self.num_feature)
        o1 = self.layer(o0)
        o2 = self.deconv(o1, output_size=(x.shape[0], 1, 28, 28))
        return o2

encoder = Encoder(2, 3)
decoder = Decoder(2, 26)


def sample_z_from_u(uv, sample_size):
    # uv : (b, dim, 2)
    # result = (b, sample_size, dim)
    b = uv.shape[0]
    dim = uv.shape[2]
    uv_mean = uv[:, :, :, 0]
    uv_variance = uv[:, :, :, 1]
    uv_mean = uv_mean.squeeze(-1)
    uv_variance = uv_variance.squeeze(-1)
    uv_mean = uv_mean.unsqueeze(1)
    uv_mean = uv_mean.broadcast_to(b, sample_size, 1, dim)
    uv_variance = uv_variance.unsqueeze(1)
    uv_variance = uv_variance.broadcast_to(b, sample_size, 1, dim)    
    z = torch.randn(b, sample_size, 1, dim)
    z = (z + uv_mean) * torch.abs(uv_variance)
    return z

optimizer = torch.optim.AdamW([{'params':encoder.parameters()}, {'params':decoder.parameters()}], lr=0.001, weight_decay=2.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.7)

def compute_loss_2(x, output, z):
    loss1 = (x - output).pow(2)  # b, sample_size, dim
    loss1 = loss1.sum()
    output1 = output.repeat_interleave(sameple_size, dim=1)
    output2 = output.repeat((1, sameple_size, 1, 1, 1))
    loss2 = torch.log((output1 - output2).pow(2) + 1)

    z1 = z.repeat_interleave(sameple_size, dim=1)
    z2 = z.repeat((1, sameple_size, 1, 1))

    loss2_factor = torch.log((z1 - z2).pow(2) + 1)
    #loss2 = loss2.sum(dim=-1)
    #loss2_factor = loss2_factor.sum(dim=-1)
    loss2 = loss2.sum(-1).sum(-1)
    loss2_factor = loss2_factor.sum(-1)
    loss3 = torch.abs(loss2 - loss2_factor)

    # loss2 = loss2 * loss2_factor
    # loss2 = loss2.sum() / sameple_size
    loss3 = loss3.sum() / (sameple_size ** 2)
    return loss1 + loss3
    # return loss1

def train_step():
    x, _ = next(iter(train_dataloader)) # b, 1, 28, 28

    # z = sample_z(sameple_size)
    uv = encoder(x) # b, 1, dim, 2
    z = sample_z_from_u(uv, sameple_size) # b, sample_size, 1, dim
    z_1 = z.reshape(z.shape[0] * z.shape[1], 1, z.shape[3])
    output = decoder(z_1) # b, sample_size, dim_out
    x = x.unsqueeze(1)
    bshape = list(x.shape)
    bshape[1] = sameple_size
    bshape = tuple(bshape)
    x = x.broadcast_to(bshape) # b, sample_size, 1, 28, 28
    # output = output.unsqueeze(0)
    # origin_output = output
    # output = output.broadcast_to((x.shape[0],) + output.shape[1:])
    # loss = compute_loss(x, output, origin_output, z)
    output = output.unsqueeze(1)
    output_shape = list(output.shape)
    output_shape[0] = output_shape[0] // sameple_size
    output_shape[1] = sameple_size

    output = output.reshape(output_shape)
    loss = compute_loss_2(x, output, z)
    loss = loss / (batch_size * sameple_size ** 2)
    return loss


for i in range(100000):
    loss = train_step()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i % 10000 == 0:
        torch.save(encoder.state_dict(), './encoder_10k.pth')
        torch.save(decoder.state_dict(), './decoder_10k.pth')
    print(f"{i}_iteration_loss is :{loss}")

# torch.save(encoder.state_dict(), './encoder_10k.pth')
# torch.save(decoder.state_dict(), './decoder_10k.pth')

# encoder_state_dict = torch.load('./encoder_10k.pth')
# decoder_state_dict = torch.load('./decoder_10k.pth')

# encoder.load_state_dict(encoder_state_dict)
# decoder.load_state_dict(decoder_state_dict)

encoder.eval()
decoder.eval()


import matplotlib.pyplot as plt
# data = torch.randn(10, data_dim)
# test_size = 2000
# # for i in range(20):
# data = torch.randn((test_size, data_dim)) - 0.7 + 0.01 * torch.randint(low=0, high=data_point_size, size=(test_size, 1))
# z = encoder(data)
# z_mean = z[:, :, 0]
# z_mean = z_mean.squeeze(-1)
# z_mean_origin = z_mean.detach()
# z_mean = z_mean.unsqueeze(1)
# z_mean = z_mean.broadcast_to(test_size, sameple_size, hidden_dim)

randn_num = torch.randn(1, 1, hidden_dim)
img = decoder(randn_num).detach().numpy() * 100
img = img.squeeze(0)
img = img.swapaxes(0, -1)
plt.imshow(img, cmap='Greys_r')

plt.show()
