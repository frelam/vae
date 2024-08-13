import torch
import torch.nn.backends

# param0 = torch.nn.Parameter(torch.randn(3, 6))

# param1 = torch.nn.Parameter(torch.randn(6, 6))
# param2 = torch.nn.Parameter(torch.randn(6, 6))
# param3 = torch.nn.Parameter(torch.randn(6, 6))
# param4 = torch.nn.Parameter(torch.randn(6, 6))
# param5 = torch.nn.Parameter(torch.randn(6, 3))

# bn1 = torch.nn.InstanceNorm1d(3)
# bn2 = torch.nn.InstanceNorm1d(3)
# bn3 = torch.nn.InstanceNorm1d(3)

# bn1 = torch.nn.BatchNormd(300)
# bn2 = torch.nn.BatchNorm1d(300)

data_dim = 2
hidden_dim = 3
batch_size = 100
sameple_size = 1
eps=1e-6
data_point_size = 3
data_size = 50000

class DecoderPerLayer(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.param0 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_features, num_features)))
        self.param1 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_features, num_features)))
        # self.bn = torch.nn.InstanceNorm1d(num_features)
        self.bn = torch.nn.BatchNorm1d(sameple_size)
        # self.bn = torch.nn.LayerNorm(num_features)
        # self.gelu = torch.nn.GELU()
    def forward(self, x):
        output = torch.matmul(x, self.param0)
        output = torch.nn.functional.leaky_relu(output, 0.1)
        # output = self.gelu(output)
        output = torch.matmul(x, self.param1)
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
    def __init__(self, num_features):
        super().__init__()
        self.param0 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_features, num_features)))
        self.param1 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_features, num_features)))
        # self.bn = torch.nn.InstanceNorm1d(num_features)
        self.bn = torch.nn.BatchNorm1d(num_features)
        # self.bn = torch.nn.LayerNorm(num_features)
        # self.gelu = torch.nn.GELU()
    def forward(self, x):
        output = torch.matmul(x, self.param0)
        output = torch.nn.functional.leaky_relu(output, 0.1)
        # output = self.gelu(output)
        output = torch.matmul(x, self.param1)
        output = x + output
        output = self.bn(output)
        return output


class EncoderLayer(torch.nn.Module):
    def __init__(self, layers_num, num_features):
        super().__init__()
        layers = []
        for i in range(layers_num):
            layers.append(EncoderPerLayer(num_features))
        self.layers = torch.nn.ModuleList(layers)
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output


# def decoder(x):
#     x = torch.unsqueeze(x, 0)
#     o0 = torch.matmul(x, param0)

#     o1 = torch.matmul(o0, param1)
#     o2 = torch.matmul(o1, param2)
#     o3 = torch.nn.functional.leaky_relu(o2, 0.6)
#     o3 = o1 + o3
#     o3 = bn1(o3)
#     o4 = torch.matmul(o3, param3)
#     o5 = torch.matmul(o4, param4)
#     o5 = torch.nn.functional.leaky_relu(o5, 0.6)
#     o5 = o3 + o5
#     o5 = bn2(o5)
#     o6 = torch.matmul(o5, param5)
#     return o6
#     # o8 = torch.softmax(o5, -1)
#     # return o8

class Encoder(torch.nn.Module):
    def __init__(self, layers_num, num_feature):
        super().__init__()
        self.layer = EncoderLayer(layers_num, num_feature)
        self.param0 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(data_dim, num_feature)))
        self.param5 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_feature, 2 * hidden_dim)))
        self.num_feature = num_feature
        self.bn = torch.nn.BatchNorm1d(2 * hidden_dim)
    def forward(self, x):
        # x = torch.unsqueeze(x, 0)
        o0 = torch.matmul(x, self.param0)
        o1 = self.layer(o0)
        o2 = torch.matmul(o1, self.param5)
        o2 = self.bn(o2)
        o2 = o2.reshape(-1, hidden_dim, 2)
        return o2


class Decoder(torch.nn.Module):
    def __init__(self, layers_num, num_feature):
        super().__init__()
        self.layer = DecoderLayer(layers_num, num_feature)
        self.param0 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(hidden_dim, num_feature)))
        self.param5 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_feature, data_dim)))
    def forward(self, x):
        # x = torch.unsqueeze(x, 0)
        o0 = torch.matmul(x, self.param0)
        o1 = self.layer(o0)
        o2 = torch.matmul(o1, self.param5)
        return o2

encoder = Encoder(4, 3)
decoder = Decoder(4, 3)

# def decoder(x):
#     x = torch.unsqueeze(x, 0)
#     o0 = torch.matmul(x, param0)

#     o1 = layer(o0)

#     o2 = torch.matmul(o1, param5)

#     return o2



# def compute_loss(x, output, origin_output, z):
#     # x = x.squeeze(0)
#     # x = x.broadcast_to(output.shape)
#     u = 0
#     sigma = 1
    
#     loss1 = (x - output).pow(2)
#     loss1 = loss1.sum()
#     # loss2 = torch.exp(-(origin_output - origin_output.mean(axis=1, keepdims=True)).pow(2))
#     origin_output1 = origin_output.repeat_interleave(sameple_size, dim=1)
#     origin_output2 = origin_output.repeat((1, sameple_size, 1))

#     z = z.broadcast_to((batch_size,) + z.shape)
#     z1 = z.repeat_interleave(sameple_size, dim=1)
#     z2 = z.repeat((1, sameple_size, 1))

#     loss2_factor = torch.exp(-((z1 - z2).pow(2)))
#     loss2 = (origin_output1 - origin_output2).pow(2)
#     loss2 = loss2 * loss2_factor
#     loss2 = loss2.sum() / sameple_size

#     # loss2 = loss2.sum() * batch_size
#     # return loss1 + 0.01 * loss2
#     return loss1 + loss2 * 2
# def sample_step(x):
#     sameple_size = 50
#     z_total = []
#     for i in range(sameple_size):
#         z_total.append(torch.randn(3) / 5 - 1 + 0.8 * (i // 100))
#     z = torch.stack(z_total)
#     # print("z is :", z)
#     output = decoder(z)
#     loss = compute_loss(x, output)
#     return loss / sameple_size

# def sample_z(sameple_size):
#     z_total = []
#     data_point = torch.randint(low=0, high=data_point_size, size=(sameple_size, 1))
#     z_total = torch.randn((sameple_size, data_dim)) / 5 - 0.7
#     data_point = data_point.broadcast_to(z_total.shape)
#     z_total += data_point * 0.3
#     return z_total

def sample_z_from_u(uv, sample_size):
    # uv : (b, dim, 2)
    # result = (b, sample_size, dim)
    b = uv.shape[0]
    dim = uv.shape[1]
    uv_mean = uv[:, :, 0]
    uv_variance = uv[:, :, 1]
    uv_mean = uv_mean.squeeze(-1)
    uv_variance = uv_variance.squeeze(-1)
    uv_mean = uv_mean.unsqueeze(1)
    uv_mean = uv_mean.broadcast_to(b, sample_size, dim)
    uv_variance = uv_variance.unsqueeze(1)
    uv_variance = uv_variance.broadcast_to(b, sample_size, dim)    
    z = torch.randn(b, sample_size, dim)
    z = (z + uv_mean) * torch.abs(uv_variance)
    return z

optimizer = torch.optim.AdamW([{'params':encoder.parameters()}, {'params':decoder.parameters()}], lr=0.001, weight_decay=2.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 0.6)



# def train_step(x):
#     total_loss = 0
#     for i in range(batch_size):
#         loss = sample_step(x[i])
#         total_loss += loss
#     total_loss = total_loss
#     total_loss = total_loss / batch_size
#     total_loss.backward()
#     optimizer.step()
#     return loss


def compute_loss_2(x, output, z):
    loss1 = (x - output).pow(2)  # b, sample_size, dim
    loss1 = loss1.sum()
    output1 = output.repeat_interleave(sameple_size, dim=1)
    output2 = output.repeat((1, sameple_size, 1))
    loss2 = torch.log((output1 - output2).pow(2) + 1)

    z1 = z.repeat_interleave(sameple_size, dim=1)
    z2 = z.repeat((1, sameple_size, 1))

    loss2_factor = torch.log((z1 - z2).pow(2) + 1)
    #loss2 = loss2.sum(dim=-1)
    #loss2_factor = loss2_factor.sum(dim=-1)
    loss2 = loss2.sum(-1)
    loss2_factor = loss2_factor.sum(-1)
    loss3 = torch.abs(loss2 - loss2_factor)

    # loss2 = loss2 * loss2_factor
    # loss2 = loss2.sum() / sameple_size
    loss3 = loss3.sum() / sameple_size
    return loss1 + loss3
    # return loss1

# def encoder(x):
#     dim = hidden_dim
#     b = x.shape[0]
#     uv = torch.empty(b, dim, 2, dtype=torch.float32)
#     uv[:, 0:30, 0] = -0.8
#     uv[:, 0:30, 1] = 0.2
#     uv[:, 30:60, 0] = -0.2
#     uv[:, 30:60, 1] = 0.5
#     uv[:, 60:, 0] = -0.2
#     uv[:, 60:, 1] = 0.8
#     return uv   


def train_step(data):
    indices = torch.randint(low=0, high=data_size - 1, size=(batch_size,))
    # indices = [i for i in range(batch_size)]
    x = data[indices, :]  

    # z = sample_z(sameple_size)
    uv = encoder(x) # b, dim, 2
    z = sample_z_from_u(uv, sameple_size) # b, sample_size, dim
    output = decoder(z) # b, sample_size, dim_out
    x = x.unsqueeze(1)
    bshape = list(x.shape)
    bshape[1] = sameple_size
    bshape = tuple(bshape)
    x = x.broadcast_to(bshape) # b, sample_size, dim_out
    # output = output.unsqueeze(0)
    # origin_output = output
    # output = output.broadcast_to((x.shape[0],) + output.shape[1:])
    # loss = compute_loss(x, output, origin_output, z)
    loss = compute_loss_2(x, output, z)
    loss = loss / (batch_size * sameple_size)
    return loss


# data = torch.randn(100, 3)
data = []
data_point = torch.randint(low=0, high=data_point_size, size=(data_size, 1))
data = torch.randn((data_size, data_dim)) / torch.randint(low=1, high=4, size=(1,)) - 0.7
data_point = data_point.broadcast_to(data.shape)
data += data_point * 0.01 # b, data_dim
# data_raw = [0.2, 0.3, 0.5] * 10
# data = torch.tensor(data_raw).reshape(100, -1)
for i in range(2000):
    loss = train_step(data)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"{i}_iteration_loss is :{loss}")

print(f"data is {data}")
encoder.eval()
decoder.eval()
# for i in range(30):
    # data = torch.randn(50, data_dim)
    # data = data.unsqueeze(0)
    # print(f"{i}th data is : ", data)
    # print(f"----{i}th output is : ----", decoder(data))

import matplotlib.pyplot as plt
# data = torch.randn(10, data_dim)
test_size = 2000
# for i in range(20):
data = torch.randn((test_size, data_dim)) - 0.7 + 0.01 * torch.randint(low=0, high=data_point_size, size=(test_size, 1))
z = encoder(data)
z_mean = z[:, :, 0]
z_mean = z_mean.squeeze(-1)
z_mean_origin = z_mean.detach()
z_mean = z_mean.unsqueeze(1)
z_mean = z_mean.broadcast_to(test_size, sameple_size, hidden_dim)
output = decoder(z_mean).squeeze(1)
print("data is ", data)

print("z_mean is ", z_mean)
print("---output is--- ", output)
plt.figure(figsize=(10, 10), dpi=100)
data_np = data.numpy()
plt.scatter(data_np[:, 0], data_np[:,1])
output_np = output.detach().numpy()
plt.scatter(output_np[:, 0], output_np[:,1])
# z_mean_np = z_mean_origin.numpy()
# plt.scatter(z_mean_np[:, 0], z_mean_np[:,1])

plt.show()
