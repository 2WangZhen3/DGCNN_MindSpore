import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F


def knn(x, k):
    inner = -2*ops.matmul(x.transpose([0, 2, 1]), x)
    xx = ops.ReduceSum(keep_dims=True)(x**2, 1)
    pairwise_distance = -xx - inner - xx.transpose([0, 2, 1])
    
    idx = ops.TopK()(pairwise_distance, k)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    # batch_size = x.size(0)
    # num_points = x.size(2)
    batch_size, _, num_points = x.shape
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    # # device = torch.device("cpu")
    # device = x.device

    idx_base = msnp.arange(0, batch_size).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.shape

    x = x.transpose([0, 2, 1])  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    # x = x.view(batch_size, num_points, 1, num_dims).repeat(k, axis=2)
    x = x.view(batch_size, num_points, 1, num_dims)
    # x = ops.repeat_elements(x, rep = k, axis = 2)
    x = msnp.tile(x, (1, 1, k, 1))
    
    # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    feature = ops.Concat(axis=3)((feature-x, x)).transpose([0, 3, 1, 2])
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


# class PointNet(nn.Module):
#     def __init__(self, args, output_channels=40):
#         super(PointNet, self).__init__()
#         self.args = args
#         self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#         self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#         self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
#         self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.bn5 = nn.BatchNorm1d(args.emb_dims)
#         self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout()
#         self.linear2 = nn.Linear(512, output_channels)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.adaptive_max_pool1d(x, 1).squeeze()
#         x = F.relu(self.bn6(self.linear1(x)))
#         x = self.dp1(x)
#         x = self.linear2(x)
#         return x


class DGCNN_cls(nn.Cell):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.bn4 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.bn5 = nn.BatchNorm1d(num_features=args.emb_dims, momentum=0.9)

        self.conv1 = nn.SequentialCell([nn.Conv2d(6, 64, 1, pad_mode='valid', weight_init='normal'),
                                        self.bn1,
                                        nn.LeakyReLU()])
        self.conv2 = nn.SequentialCell([nn.Conv2d(64*2, 64, 1, pad_mode='valid', weight_init='normal'),
                                        self.bn2,
                                        nn.LeakyReLU()])
        self.conv3 = nn.SequentialCell([nn.Conv2d(64*2, 128, 1, pad_mode='valid', weight_init='normal'),
                                        self.bn3,
                                        nn.LeakyReLU()])
        self.conv4 = nn.SequentialCell([nn.Conv2d(128*2, 256, 1, pad_mode='valid', weight_init='normal'),
                                        self.bn4,
                                        nn.LeakyReLU()])
        # self.conv5 = nn.SequentialCell([nn.Conv1d(512, args.emb_dims, 1, pad_mode='valid', weight_init='normal'),
        #                                 self.bn5,
        #                                 nn.LeakyReLU()])
        self.conv5 = nn.SequentialCell([nn.Conv1d(512, args.emb_dims, 1, pad_mode='valid', weight_init='normal'),
                                        nn.LeakyReLU()])

        self.linear1 = nn.Dense(args.emb_dims*2, 512, has_bias=False)
        self.bn6 = nn.BatchNorm1d(num_features=512, momentum=0.9)
        self.dp1 = nn.Dropout(keep_prob=1-args.dropout)
        self.linear2 = nn.Dense(512, 256)
        self.bn7 = nn.BatchNorm1d(num_features=256, momentum=0.9)
        self.dp2 = nn.Dropout(keep_prob=1-args.dropout)
        self.linear3 = nn.Dense(256, output_channels)

        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.leaky_relu = nn.LeakyReLU()

        self.log_softmax = nn.LogSoftmax()

    def construct(self, x):
        batch_size, _, _ = x.shape
        x = x.transpose([0, 2, 1])
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k) 
        x1 = ops.max(x, axis=3)[1]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = ops.max(x, axis=3)[1]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = ops.max(x, axis=3)[1]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = ops.max(x, axis=3)[1]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = ops.Concat(axis=1)((x1, x2, x3, x4))  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        # x1 = self.max_pooling(x).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x1 = ops.max(x, axis=2)[1].view(batch_size, -1)
        x2 = self.avg_pooling(x).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = ops.Concat(axis=1)((x1, x2))              # (batch_size, emb_dims*2)

        x = self.leaky_relu(self.bn6(self.linear1(x)))  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = self.leaky_relu(self.bn7(self.linear2(x)))  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        x = self.log_softmax(x)
        x = ops.ExpandDims()(x, 2)
        return x
    

class NLLLoss(LossBase):
    """NLL loss"""

    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__(reduction)
        self.one_hot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, logits, label):
        """
        construct method
        """
        label_one_hot = self.one_hot(label, F.shape(logits)[-1], F.scalar_to_tensor(1.0), F.scalar_to_tensor(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return loss


if __name__ == "__main__":
    from dataset import ModelNet40
    import mindspore.dataset as ds
    import argparse

    def parse_args():
        """PARAMETERS"""
        parser = argparse.ArgumentParser('MindSpore DGCNN Training Configurations.')
        parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
        parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
        parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
        
        return parser.parse_known_args()[0]
    
    args = parse_args()
    net = DGCNN_cls(args)
    train_ds_generator = ModelNet40(num_points=1024, data_dir='/home/root/code_demo/dgcnn.pytorch/data/', partition='train')
    train_ds = ds.GeneratorDataset(train_ds_generator, ["data", "label"],
                                   num_parallel_workers=1,
                                   shuffle=True)
    train_ds = train_ds.batch(batch_size=8,
                              drop_remainder=True,
                              num_parallel_workers=1)
    for batch, (data, label) in enumerate(train_ds.create_tuple_iterator()):
        pred = net.construct(data)
        # net_loss = NLLLoss()
        # loss = net_loss.construct(pred, label)
        net_loss = nn.NLLLoss()
        loss = net_loss(pred, label)
        debug = 0
