import torch
from torch import nn
from utils import ops
import math


class EdgeConvBlock(nn.Module):
    def __init__(self, K=(32, 32, 32), xyz_or_feature=('feature', 'feature', 'feature'), feature_or_diff=('diff', 'diff', 'diff'),
                 conv1_channel_in=(3*2, 64*2, 64*2), conv1_channel_out=(64, 64, 64), conv2_channel_in=(64, 64, 64), conv2_channel_out=(64, 64, 64)):
        super(EdgeConvBlock, self).__init__()
        self.edgeconv_list = nn.ModuleList([EdgeConv(k, x_or_f, f_or_d, conv1_in, conv1_out, conv2_in, conv2_out)
                                            for k, x_or_f, f_or_d, conv1_in, conv1_out, conv2_in, conv2_out
                                            in zip(K, xyz_or_feature, feature_or_diff, conv1_channel_in, conv1_channel_out, conv2_channel_in, conv2_channel_out)])

    def forward(self, x):
        x_list = []
        for edgeconv in self.edgeconv_list:
            x = edgeconv(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        return x, x_list


class EdgeConv(nn.Module):
    def __init__(self, K=32, xyz_or_feature='feature', feature_or_diff='diff',
                 conv1_channel_in=6, conv1_channel_out=64, conv2_channel_in=64, conv2_channel_out=64):

        super(EdgeConv, self).__init__()
        self.K = K
        self.xyz_or_feature = xyz_or_feature
        self.feature_or_diff = feature_or_diff

        self.conv1 = nn.Sequential(nn.Conv2d(conv1_channel_in, conv1_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv1_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv2_channel_in, conv2_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv2_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        xyz = x[...]  # TODO: xyz_or_feature='xyz' is not correct when you stack multiple EdgeConv
        # x.shape == (B, C, N)   xyz.shape == (B, C, N)
        x, _ = ops.group(x, xyz, self.K, self.xyz_or_feature, self.feature_or_diff, cross_attention=False)
        # x.shape == (B, 2C, N, K)
        x = self.conv1(x)
        # x.shape == (B, C, N, K)
        x = self.conv2(x)
        # x.shape == (B, C, N, K)
        x = x.max(dim=-1, keepdim=False)[0]
        # x.shape == (B, C, N)
        return x


class Embedding(nn.Module):
    def __init__(self, emb_in, emb_out):
        super(Embedding, self).__init__()
        self.embedding = nn.Conv1d(emb_in, emb_out, 1, bias=False)

    def forward(self, x):
        # x.shape == (B, C, N)
        x = self.embedding(x)
        # x.shape == (B, C, N)
        return x


class Point2NeighborAttentionBlock(nn.Module):
    def __init__(self, K=(32, 32, 32), xyz_or_feature=('feature', 'feature', 'feature'),
                 feature_or_diff=('diff', 'diff', 'diff'), qkv_channels=(64, 64, 64),
                 ff_conv1_channels_in=(64, 64, 64), ff_conv1_channels_out=(128, 128, 128),
                 ff_conv2_channels_in=(128, 128, 128), ff_conv2_channels_out=(64, 64, 64), num_heads=(8, 8, 8)):
        super(Point2NeighborAttentionBlock, self).__init__()
        self.point2neighbor_list = nn.ModuleList([Point2NeighborAttention(k, x_or_f, f_or_d, qkv_channel, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out, heads)
                                                  for k, x_or_f, f_or_d, qkv_channel, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out, heads
                                                  in zip(K, xyz_or_feature, feature_or_diff, qkv_channels, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out, num_heads)])

    def forward(self, x):
        x_list = []
        for point2neighbor in self.point2neighbor_list:
            x = point2neighbor(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        return x, x_list


class Point2NeighborAttention(nn.Module):
    def __init__(self, K=32, xyz_or_feature='feature', feature_or_diff='diff', qkv_channels=64,
                 ff_conv1_channels_in=64, ff_conv1_channels_out=128,
                 ff_conv2_channels_in=128, ff_conv2_channels_out=64, num_heads=8):

        super(Point2NeighborAttention, self).__init__()
        self.K = K
        self.xyz_or_feature = xyz_or_feature
        self.feature_or_diff = feature_or_diff

        self.ca = CrossAttention(qkv_channels, ff_conv1_channels_in, ff_conv1_channels_out,
                                 ff_conv2_channels_in, ff_conv2_channels_out, num_heads)

    def forward(self, x):
        xyz = x[...]  # TODO: xyz_or_feature='xyz' is not correct when you stack multiple Point2NeighborAttention
        # x.shape == (B, C, N)
        x, neighbors = ops.group(x, xyz, self.K, self.xyz_or_feature, self.feature_or_diff, cross_attention=True)
        # x.shape == (B, C, N, 1)    neighbors.shape == (B, C, N, K)
        x = self.ca(x, neighbors)
        # x.shape == (B, C, N)
        return x


class CrossAttention(nn.Module):
    def __init__(self, qkv_channels=64, ff_conv1_channels_in=64, ff_conv1_channels_out=128,
                 ff_conv2_channels_in=128, ff_conv2_channels_out=64, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        if qkv_channels % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        else:
            self.depth = int(qkv_channels / num_heads)

        self.q_conv = nn.Conv2d(qkv_channels, qkv_channels, 1, bias=False)
        self.k_conv = nn.Conv2d(qkv_channels, qkv_channels, 1, bias=False)
        self.v_conv = nn.Conv2d(qkv_channels, qkv_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(qkv_channels)
        self.bn2 = nn.BatchNorm1d(qkv_channels)

    def forward(self, x, neighbors):
        # x.shape == (B, C, N, 1)
        q = self.q_conv(x)
        # q.shape == (B, C, N, 1)
        q = self.split_heads(q)
        # q.shape == (B, H, N, 1, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N, K)
        k = self.split_heads(k)
        # k.shape == (B, H, N, k, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N, K)
        v = self.split_heads(v)
        # v.shape == (B, H, N, k, D)
        k = k.permute(0, 1, 2, 4, 3)
        # k.shape == (B, H, N, D, K)
        energy = q @ k
        # energy.shape == (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, 1, K)
        x_r = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # x_r.shape == (B, N, H, D)
        x_r = x_r.reshape(x_r.shape[0], x_r.shape[1], -1).permute(0, 2, 1)
        # x_r.shape == (B, C, N)
        x = self.bn1(x[:, :, :, 0] + x_r)
        # x.shape == (B, C, N)
        x_r = self.ff(x)
        # x_r.shape == (B, C, N)
        x = self.bn2(x + x_r)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], self.num_heads, self.depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x


class Point2PointAttentionBlock(nn.Module):
    def __init__(self, use_embedding=True, embedding_channels_in=64*3, embedding_channels_out=1024, qkv_channels=(1024,),
                 ff_conv1_channels_in=(1024,), ff_conv1_channels_out=(512,),
                 ff_conv2_channels_in=(512,), ff_conv2_channels_out=(1024,)):
        super(Point2PointAttentionBlock, self).__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Conv1d(embedding_channels_in, embedding_channels_out, 1, bias=False)
        self.point2point_list = nn.ModuleList([Point2PointAttention(qkv_channel, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out)
                                 for qkv_channel, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out
                                 in zip(qkv_channels, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)])

    def forward(self, x):
        if self.use_embedding:
            # x.shape == (B, C, N)
            x = self.embedding(x)
            # x.shape == (B, C, N)
        for point2point in self.point2point_list:
            x = point2point(x)
            # x.shape == (B, C, N)
        return x


class Point2PointAttention(nn.Module):
    def __init__(self, qkv_channels=64,
                 ff_conv1_channels_in=64, ff_conv1_channels_out=128,
                 ff_conv2_channels_in=128, ff_conv2_channels_out=64):
        super(Point2PointAttention, self).__init__()

        self.q_conv = nn.Conv1d(qkv_channels, qkv_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(qkv_channels, qkv_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(qkv_channels, qkv_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(qkv_channels)
        self.bn2 = nn.BatchNorm1d(qkv_channels)

    def forward(self, x):
        # x.shape == (B, C, N)
        q = self.q_conv(x).permute(0, 2, 1)
        # q.shape == (B, N, C)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        energy = q @ k
        # energy.shape == (B, N, N)
        scale_factor = math.sqrt(x.shape[1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, N, N)
        x_r = (attention @ v.permute(0, 2, 1)).permute(0, 2, 1)
        # x_r.shape == (B, C, N)
        x = self.bn1(x + x_r)
        # x.shape == (B, C, N)
        x_r = self.ff(x)
        # x_r.shape == (B, C, N)
        x = self.bn2(x + x_r)
        # x.shape == (B, C, N)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channels_in=(64*3,), channels_out=(1024,)):
        super(ConvBlock, self).__init__()
        self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv1d(channel_in, channel_out, 1, bias=False), nn.BatchNorm1d(channel_out), nn.LeakyReLU(negative_slope=0.2))
                                        for channel_in, channel_out in zip(channels_in, channels_out)])

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        return x


class ShapeNetModel(nn.Module):
    def __init__(self, embed_in, embed_out, p2neighbor_enable, p2neighbor_K, p2neighbor_x_or_f, p2neighbor_f_or_d, p2neighbor_qkv_channels,
                 p2neighbor_num_heads, p2neighbor_ff_conv1_in, p2neighbor_ff_conv1_out, p2neighbor_ff_conv2_in,
                 p2neighbor_ff_conv2_out, edgeconv_K, edgeconv_x_or_f, edgeconv_f_or_d, edgeconv_conv1_in, edgeconv_conv1_out,
                 edgeconv_conv2_in, edgeconv_conv2_out, p2point_enable, p2point_use_embedding, p2point_embed_in, p2point_embed_out,
                 p2point_qkv_channels, p2point_ff_conv1_in, p2point_ff_conv1_out, p2point_ff_conv2_in, p2point_ff_conv2_out,
                 conv_block_channels_in, conv_block_channels_out):

        super(ShapeNetModel, self).__init__()
        self.p2neighbor_enable = p2neighbor_enable

        if p2neighbor_enable:
            self.embedding = Embedding(embed_in, embed_out)
            self.point2neighbor_or_edgeconv = Point2NeighborAttentionBlock(p2neighbor_K, p2neighbor_x_or_f, p2neighbor_f_or_d,
                                                                           p2neighbor_qkv_channels, p2neighbor_ff_conv1_in,
                                                                           p2neighbor_ff_conv1_out, p2neighbor_ff_conv2_in, p2neighbor_ff_conv2_out, p2neighbor_num_heads)
            x_cat_channels = sum(p2neighbor_ff_conv2_out)
        else:
            self.point2neighbor_or_edgeconv = EdgeConvBlock(edgeconv_K, edgeconv_x_or_f, edgeconv_f_or_d, edgeconv_conv1_in, edgeconv_conv1_out, edgeconv_conv2_in, edgeconv_conv2_out)
            x_cat_channels = sum(edgeconv_conv2_out)

        if p2point_enable:
            self.point2point_or_conv = Point2PointAttentionBlock(p2point_use_embedding, p2point_embed_in, p2point_embed_out, p2point_qkv_channels, p2point_ff_conv1_in,
                                                                 p2point_ff_conv1_out, p2point_ff_conv2_in, p2point_ff_conv2_out)
            point2point_or_conv_output_channels = p2point_ff_conv2_out[-1]
        else:
            self.point2point_or_conv = ConvBlock(conv_block_channels_in, conv_block_channels_out)
            point2point_or_conv_output_channels = conv_block_channels_out[-1]

        self.conv1 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(x_cat_channels+point2point_or_conv_output_channels+64, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Conv1d(128, 50, kernel_size=1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x, category_id):
        # x.shape == (B, C=3, N)  category_id.shape == (B, 16, 1)
        B, C, N = x.shape
        # x.shape == (B, C=3, N)
        if self.p2neighbor_enable:
            x = self.embedding(x)
            # x.shape == (B, C, N)
        x, x_list = self.point2neighbor_or_edgeconv(x)
        # x.shape == (B, C, N)    torch.cat(x_list, dim=1).shape == (B, C1, N)
        x = self.point2point_or_conv(x)
        # x.shape == (B, C, N)
        x = x.max(dim=-1, keepdim=True)[0]
        # x.shape == (B, C, 1)
        category_id = self.conv1(category_id)
        # category_id.shape == (B, 64, 1)
        x = torch.cat([x, category_id], dim=1)
        # x.shape === (B, C+64, 1)
        x = x.repeat(1, 1, N)
        # x.shape == (B, C+64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)
        # x.shape == (B, C1+C+64, N)
        x = self.conv2(x)
        # x.shape == (B, 256, N)
        x = self.dp1(x)
        # x.shape == (B, 256, N)
        x = self.conv3(x)
        # x.shape == (B, 256, N)
        x = self.dp2(x)
        # x.shape == (B, 256, N)
        x = self.conv4(x)
        # x.shape == (B, 128, N)
        x = self.conv5(x)
        # x.shape == (B, 50, N)
        return x
