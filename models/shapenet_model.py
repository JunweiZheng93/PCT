import torch
from torch import nn
from utils import ops


class EdgeConv(nn.Module):
    # TODO: hard code baseline hyper parameter after running
    def __init__(self, K=32, emb_cov1_channel1=3, emb_cov1_channel2=64, emb_cov2_channel1=64, emb_cov2_channel2=64,
                 local_op1_channel1=128, local_op1_channel2=128, local_op2_channel1=256, local_op2_channel2=128):
        super(EdgeConv, self).__init__()
        self.K = K
        self.conv1 = nn.Conv1d(emb_cov1_channel1, emb_cov1_channel2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(emb_cov2_channel1, emb_cov2_channel2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(emb_cov1_channel2)
        self.bn2 = nn.BatchNorm1d(emb_cov2_channel2)
        self.relu = nn.ReLU()
        self.local_op1 = LocalOperation(local_op1_channel1, local_op1_channel2)
        self.local_op2 = LocalOperation(local_op2_channel1, local_op2_channel2)

    def forward(self, x):  # x.shape == (B, N, C=3)
        # TODO: if it is better to calculate distance in feature space, not in Euclidean space
        # xyz.shape == (B, N, C=3)
        xyz = x[...]
        # x.shape == (B, C=3, N)
        x = x.permute(0, 2, 1)
        # x.shape == (B, C=64, N)
        x = self.relu(self.bn1(self.conv1(x)))
        # x.shape == (B, C=64, N)
        x = self.relu(self.bn2(self.conv2(x)))
        # x.shape == (B, N, C=64)
        x = x.permute(0, 2, 1)
        # x.shape == (B, N, K=32, C=128)
        x = ops.group(x, xyz, self.K)
        # x.shape == (B, N, C=128)
        x = self.local_op1(x)
        # x.shape == (B, N, K=32, C=256)
        x = ops.group(x, xyz, self.K)
        # x.shape == (B, N, C=128)
        x = self.local_op2(x)
        return x


class LocalOperation(nn.Module):
    def __init__(self, channel1=128, channel2=128):
        super(LocalOperation, self).__init__()
        self.conv1 = nn.Conv1d(channel1, channel2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channel2, channel2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channel2)
        self.bn2 = nn.BatchNorm1d(channel2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape == (B, N, K=32, C=128)
        B, N, K, C = x.shape
        # x.shape == (B, N, C=128, K=32)
        x = x.permute(0, 1, 3, 2)
        # x.shape == (B*N, C=128, K=32)
        x = x.view(-1, C, K)
        # x.shape == (B*N, C=256, K=32)
        x = self.relu(self.bn1(self.conv1(x)))
        # x.shape == (B*N, C=256, K=32)
        x = self.relu(self.bn2(self.conv2(x)))
        # x.shape == (B*N, C=256)
        x = torch.max(x, dim=2)[0]
        # x.shape == (B, N, C=256)
        x = x.view(B, N, -1)
        return x


class Point2NeighborAttention(nn.Module):
    def __init__(self, K=32, emb_cov1_channel1=3, emb_cov1_channel2=64, emb_cov2_channel1=64, emb_cov2_channel2=128,
                 qkv_channels=128):

        super(Point2NeighborAttention, self).__init__()
        self.K = K
        self.conv1 = nn.Conv1d(emb_cov1_channel1, emb_cov1_channel2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(emb_cov2_channel1, emb_cov2_channel2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(emb_cov1_channel2)
        self.bn2 = nn.BatchNorm1d(emb_cov2_channel2)
        self.relu = nn.ReLU()
        self.ca1 = CrossAttention(qkv_channels)
        self.ca2 = CrossAttention(qkv_channels)

    def forward(self, x):  # x.shape == (B, N, C=3)
        # TODO: if it is better to calculate distance in feature space, not in Euclidean space
        # xyz.shape == (B, N, C=3)
        xyz = x[...]
        # x.shape == (B, C=3, N)
        x = x.permute(0, 2, 1)
        # x.shape == (B, C=64, N)
        x = self.relu(self.bn1(self.conv1(x)))
        # x.shape == (B, C=128, N)
        x = self.relu(self.bn2(self.conv2(x)))
        # x.shape == (B, N, C=128)
        x = x.permute(0, 2, 1)
        # x.shape == (B, N, C=128)    grouped_points.shape == (B, N, K=32, C=128)
        x, grouped_points = ops.group_for_point2neighbor_attention(x, xyz, self.K)
        # x.shape == (B, C=128, N)
        x = x.permute(0, 2, 1)
        # kv.shape == (B*N, C=128, K=32)
        kv = grouped_points.view(-1, self.K, grouped_points.shape[3]).permute(0, 2, 1)
        # x.shape == (B, C=128, N)
        x = self.ca1(x, kv)
        # x.shape == (B, N, C=128)
        x = x.permute(0, 2, 1)
        # x.shape == (B, N, C=128)    grouped_points.shape == (B, N, K=32, C=128)
        x, grouped_points = ops.group_for_point2neighbor_attention(x, xyz, self.K)
        # x.shape == (B, C=128, N)
        x = x.permute(0, 2, 1)
        # kv.shape == (B*N, C=128, K=32)
        kv = grouped_points.view(-1, self.K, grouped_points.shape[3]).permute(0, 2, 1)
        # x.shape == (B, C=128, N)
        x = self.ca2(x, kv).permute(0, 2, 1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, qkv_channels=128):
        super(CrossAttention, self).__init__()
        self.q_conv = nn.Conv1d(qkv_channels, qkv_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(qkv_channels, qkv_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(qkv_channels, qkv_channels, 1, bias=False)
        self.ff_conv = nn.Conv1d(qkv_channels, qkv_channels, 1, bias=False)
        self.ff_bn = nn.BatchNorm1d(qkv_channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, kv):
        # x.shape == (B, C=128, N)   kv.shape == (B*N, C=128, K=32)
        B, C, N = x.shape
        # q.shape == (B, N, 1, C=128)
        q = self.q_conv(x).permute(0, 2, 1)[:, :, None, :]
        # k.shape ==  (B, N, C=128, K=32)
        k = self.k_conv(kv).view(B, N, -1, kv.shape[2])
        # v.shape ==  (B, N, C=128, K=32)
        v = self.v_conv(kv).view(B, N, -1, kv.shape[2])
        # energy.shape == (B, N, 1, K=32)
        energy = q @ k
        # attention.shape == (B, N, 1, K=32)
        scale_factor = torch.sqrt(torch.Tensor([C]))
        attention = self.softmax(energy / scale_factor)
        # x_r.shape == (B, C=128, N)
        x_r = (attention @ v.permute(0, 1, 3, 2)).squeeze().permute(0, 2, 1)
        # x_r.shape == (B, C=128, N)  TODO: need to add kv?
        x_r = self.act(self.ff_bn(self.ff_conv(x - x_r)))
        x = x + x_r
        return x


class Point2PointAttention(nn.Module):
    def __init__(self, channels=128):
        super(Point2PointAttention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.ff_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x.shape == (B, C, N)
        # x_q.shape == (B, N, C)
        x_q = self.q_conv(x).permute(0, 2, 1)
        # x_k.shape == (B, C, N)
        x_k = self.k_conv(x)
        # x_v.shape == (B, C, N)
        x_v = self.v_conv(x)
        # energy.shape == (B, N, N)
        energy = x_q @ x_k
        # attention.shape == (B, N, N)
        scale_factor = torch.sqrt(torch.Tensor([x.shape[1]]))
        attention = self.softmax(energy / scale_factor)
        # x_r.shape == (B, C, N)
        x_r = (attention @ x_v.permute(0, 2, 1)).permute(0, 2, 1)
        # x_r.shape == (B, C, N)
        x_r = self.act(self.bn(self.ff_conv(x - x_r)))
        # x.shape == (B, C, N)
        x = x + x_r
        return x


class ShapeNetModel(nn.Module):
    def __init__(self, K=32, conv_channels=128,
                 emb_cov1_channel1=3, emb_cov1_channel2=64, emb_cov2_channel1=64, emb_cov2_channel2=64,
                 local_op1_channel1=128, local_op1_channel2=128, local_op2_channel1=256, local_op2_channel2=128,
                 point2point_attention_enable=True, qkv_channel_p2point=128, point2neighbor_attention_enable=True,
                 qkv_channel_p2neighbor=128, emb_conv2_channel2_p2neighbor=128):

        super(ShapeNetModel, self).__init__()
        self.conv_fuse = nn.Sequential(nn.Conv1d(4 * qkv_channel_p2point, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.category_cov = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        self.conv1 = nn.Conv1d(1024*3+64, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(256, 50, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        if point2neighbor_attention_enable:
            self.point2neighbor_or_edgeconv = Point2NeighborAttention(K, emb_cov1_channel1, emb_cov1_channel2, emb_cov2_channel1, emb_conv2_channel2_p2neighbor, qkv_channel_p2neighbor)
        else:
            self.point2neighbor_or_edgeconv = EdgeConv(K, emb_cov1_channel1, emb_cov1_channel2, emb_cov2_channel1, emb_cov2_channel2,
                                                       local_op1_channel1, local_op1_channel2, local_op2_channel1, local_op2_channel2)

        if point2point_attention_enable:
            self.point2point_or_conv1 = Point2PointAttention(qkv_channel_p2point)
            self.point2point_or_conv2 = Point2PointAttention(qkv_channel_p2point)
            self.point2point_or_conv3 = Point2PointAttention(qkv_channel_p2point)
            self.point2point_or_conv4 = Point2PointAttention(qkv_channel_p2point)
        else:
            self.point2point_or_conv1 = nn.Conv1d(conv_channels, conv_channels, kernel_size=1, bias=False)
            self.point2point_or_conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=1, bias=False)
            self.point2point_or_conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size=1, bias=False)
            self.point2point_or_conv4 = nn.Conv1d(conv_channels, conv_channels, kernel_size=1, bias=False)

    def forward(self, x, category_id):
        # x.shape == (B, N, C=3)  category_id.shape == (B, 16)
        B, N, C = x.shape
        # x.shape == (B, N, C=128)
        x = self.point2neighbor_or_edgeconv(x)
        # x.shape == (B, C=128, N)
        x = x.permute(0, 2, 1)
        # x1.shape == (B, C=128, N)
        x1 = self.point2point_or_conv1(x)
        # x2.shape == (B, C=128, N)
        x2 = self.point2point_or_conv2(x1)
        # x3.shape == (B, C=128, N)
        x3 = self.point2point_or_conv3(x2)
        # x4.shape == (B, C=128, N)
        x4 = self.point2point_or_conv4(x3)
        # x.shape == (B, C=512, N)
        x = torch.concat((x1, x2, x3, x4), dim=1)
        # x.shape == (B, C=1024, N)
        x = self.conv_fuse(x)
        # x_max.shape == (B, C=1024)
        x_max = torch.max(x, 2)[0]
        # x_avg.shape == (B, C=1024)
        x_avg = torch.mean(x, 2)
        # x_max_feature.shape == (B, C=1024, N)
        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        # x_avg_feature.shape == (B, C=1024, N)
        x_avg_feature = x_avg.unsqueeze(-1).repeat(1, 1, N)
        # category_id.shape == (B, 16, 1)
        category_id = category_id.view(B, 16, 1)
        # category_feature.shape == (B, 64, N)
        category_feature = self.category_cov(category_id).repeat(1, 1, N)
        # x_global_feature.shape == (B, 2*1024+64, N)
        x_global_feature = torch.concat((x_max_feature, x_avg_feature, category_feature), 1)
        # x.shape == (B, 3*1024+64, N)
        x = torch.concat((x, x_global_feature), 1)
        # x.shape == (B, 512, N)
        x = self.relu(self.bn1(self.conv1(x)))
        # x.shape == (B, 512, N)
        x = self.dropout(x)
        # x.shape == (B, 256, N)
        x = self.relu(self.bn2(self.conv2(x)))
        # x.shape == (B, 50, N)
        x = self.conv3(x)
        # x.shape == (B, N, 50)
        x = x.permute(0, 2, 1)
        return x
