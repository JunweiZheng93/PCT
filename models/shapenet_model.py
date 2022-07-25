import torch
from torch import nn
from utils import ops
import math


class EdgeConvBlock(nn.Module):
    def __init__(self, K=(32, 32, 32), neighbor_selection_method=('feature', 'feature', 'feature'), group_type=('center_diff', 'center_diff', 'center_diff'),
                 conv1_channel_in=(3*2, 64*2, 64*2), conv1_channel_out=(64, 64, 64), conv2_channel_in=(64, 64, 64), conv2_channel_out=(64, 64, 64), pooling=('max', 'max', 'max')):
        super(EdgeConvBlock, self).__init__()
        self.edgeconv_list = nn.ModuleList([EdgeConv(k, method, g_type, conv1_in, conv1_out, conv2_in, conv2_out, pool)
                                            for k, method, g_type, conv1_in, conv1_out, conv2_in, conv2_out, pool
                                            in zip(K, neighbor_selection_method, group_type, conv1_channel_in, conv1_channel_out, conv2_channel_in, conv2_channel_out, pooling)])

    def forward(self, x, coordinate):
        x_list = []
        for edgeconv in self.edgeconv_list:
            x = edgeconv(x, coordinate)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        return x, x_list


class EdgeConv(nn.Module):
    def __init__(self, K=32, neighbor_selection_method='feature', group_type='center_diff',
                 conv1_channel_in=6, conv1_channel_out=64, conv2_channel_in=64, conv2_channel_out=64, pooling='max'):

        super(EdgeConv, self).__init__()
        self.K = K
        self.neighbor_selection_method = neighbor_selection_method
        self.group_type = group_type
        self.pooling = pooling

        self.conv1 = nn.Sequential(nn.Conv2d(conv1_channel_in, conv1_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv1_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv2_channel_in, conv2_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv2_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, coordinate):
        # x.shape == (B, C, N)   coordinate.shape == (B, C, N)
        x = ops.group(x, coordinate, self.K, 0, self.neighbor_selection_method, self.group_type)
        # x.shape == (B, 2C, N, K) or (B, C, N, K)
        x = self.conv1(x)
        # x.shape == (B, C, N, K)
        x = self.conv2(x)
        # x.shape == (B, C, N, K)
        if self.pooling == 'max':
            x = x.max(dim=-1, keepdim=False)[0]
        elif self.pooling == 'average':
            x = x.mean(dim=-1, keepdim=False)
        else:
            raise ValueError('pooling should be max or average')
        # x.shape == (B, C, N)
        return x


class LinearEmbedding(nn.Module):
    def __init__(self, emb_in, emb_out):
        super(LinearEmbedding, self).__init__()
        self.embedding = nn.Conv1d(emb_in, emb_out, 1, bias=False)

    def forward(self, x):
        # x.shape == (B, C, N)
        x = self.embedding(x)
        # x.shape == (B, C, N)
        return x


class Point2PointEmbedding(nn.Module):
    def __init__(self, q_in, q_out, k_in, k_out, v_in, v_out, num_heads):
        super(Point2PointEmbedding, self).__init__()
        self.sa = SelfAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads)

    def forward(self, x):
        # x.shape == (B, C, N)
        x = self.sa(x)
        # x.shape == (B, C, N)
        return x


class Neighbor2PointEmbedding(nn.Module):
    def __init__(self, K=32, neighbor_selection_method='coordinate', group_type='diff', q_in=3, q_out=64, k_in=3, k_out=64, v_in=3, v_out=64, num_heads=8):
        super(Neighbor2PointEmbedding, self).__init__()
        self.K = K
        self.neighbor_selection_method = neighbor_selection_method
        self.group_type = group_type
        self.embedding = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads)

    def forward(self, x, coordinate):
        # x.shape == (B, C, N)
        neighbors = ops.group(x, coordinate, self.K, 0, self.neighbor_selection_method, self.group_type)
        # x.shape == (B, C, N)    neighbors.shape == (B, C, N, K)
        x = self.embedding(x, neighbors)
        # x.shape == (B, C, N)
        return x


class Neighbor2PointAttentionBlock(nn.Module):
    def __init__(self, scale=(0, 0, 0), shared_ca=(False, False, False), concat_ms_inputs=(False, False, False),
                 mlp_or_ca=('mlp', 'mlp', 'mlp'), K=(32, 32, 32), neighbor_selection_method=('feature', 'feature', 'feature'),
                 group_type=('diff', 'diff', 'diff'), q_in=(64, 64, 64), q_out=(64, 64, 64),
                 k_in=(64, 64, 64), k_out=(64, 64, 64), v_in=(64, 64, 64), v_out=(64, 64, 64), num_heads=(8, 8, 8),
                 ff_conv1_channels_in=(64, 64, 64), ff_conv1_channels_out=(128, 128, 128),
                 ff_conv2_channels_in=(128, 128, 128), ff_conv2_channels_out=(64, 64, 64)):
        super(Neighbor2PointAttentionBlock, self).__init__()
        self.neighbor2point_list = nn.ModuleList([Neighbor2PointAttention(s, shared, concat, m_or_ca, k, method, g_type, q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out)
                                                  for s, shared, concat, m_or_ca, k, method, g_type, q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out
                                                  in zip(scale, shared_ca, concat_ms_inputs, mlp_or_ca, K, neighbor_selection_method, group_type, q_in, q_out, k_in, k_out, v_in, v_out, num_heads, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)])

    def forward(self, x, coordinate):
        x_list = []
        for neighbor2point in self.neighbor2point_list:
            x = neighbor2point(x, coordinate)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        return x, x_list


class Neighbor2PointAttention(nn.Module):
    def __init__(self, scale=0, shared_ca=False, concat_ms_inputs=False, mlp_or_ca='mlp',
                 K=32, neighbor_selection_method='feature', group_type='diff',
                 q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8,
                 ff_conv1_channels_in=64, ff_conv1_channels_out=128,
                 ff_conv2_channels_in=128, ff_conv2_channels_out=64):

        super(Neighbor2PointAttention, self).__init__()
        self.scale = scale
        self.K = K
        self.neighbor_selection_method = neighbor_selection_method
        self.group_type = group_type

        if scale == 0:
            self.ca_reslink = CrossAttentionResLink(q_in, q_out, k_in, k_out, v_in, v_out, num_heads, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)
        else:
            self.ca_ms = CrossAttentionMS(scale, shared_ca, concat_ms_inputs, mlp_or_ca, q_in, q_out, k_in, k_out, v_in, v_out, num_heads, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)

    def forward(self, x, coordinate):
        if self.scale == 0:
            # x.shape == (B, C, N)
            neighbors = ops.group(x, coordinate, self.K, 0, self.neighbor_selection_method, self.group_type)
            # x.shape == (B, C, N)    neighbors.shape == (B, C, N, K)
            x = self.ca_reslink(x, neighbors)
            # x.shape == (B, C, N)
        else:
            neighbor_list = []
            for i in range(self.scale+1):
                # x.shape == (B, C, N)
                neighbors = ops.group(x, coordinate, self.K, i, self.neighbor_selection_method, self.group_type)
                # neighbors.shape == (B, C, N, K)
                neighbor_list.append(neighbors)
            x = self.ca_ms(x, neighbor_list)
            # x.shape == (B, C, N)
        return x


class CrossAttention(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8):
        super(CrossAttention, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, neighbors):
        # x.shape == (B, C, N)
        x = x[:, :, :, None]
        # x.shape == (B, C, N, 1)  neighbors.shape == (B, C, N, K)
        q = self.q_conv(x)
        # q.shape == (B, C, N, 1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, 1, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N, K)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, K, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N, K)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, K, D)
        k = k.permute(0, 1, 2, 4, 3)
        # k.shape == (B, H, N, D, K)
        energy = q @ k
        # energy.shape == (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, 1, K)
        x = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # x.shape == (B, N, H, D)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x


class CrossAttentionResLink(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8, ff_conv1_channels_in=64,
                 ff_conv1_channels_out=128, ff_conv2_channels_in=128, ff_conv2_channels_out=64):
        super(CrossAttentionResLink, self).__init__()
        # check input values
        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')

        self.ca = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads)
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)
        
    def forward(self, x, neighbors):
        # x.shape == (B, C, N)  neighbors.shape == (B, C, N, K)
        x_out = self.ca(x, neighbors)
        # x_out.shape == (B, C, N)
        x = self.bn1(x + x_out)
        # x.shape == (B, C, N)
        x_out = self.ff(x)
        # x_out.shape == (B, C, N)
        x = self.bn2(x + x_out)
        # x.shape == (B, C, N)
        return x


class CrossAttentionMS(nn.Module):
    def __init__(self, scale, shared_ca, concat_ms_inputs, mlp_or_ca, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8, ff_conv1_channels_in=64,
                 ff_conv1_channels_out=128, ff_conv2_channels_in=128, ff_conv2_channels_out=64):
        super(CrossAttentionMS, self).__init__()
        self.shared_ca = shared_ca
        self.concat_ms_inputs = concat_ms_inputs
        self.mlp_or_ca = mlp_or_ca
        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')
        if concat_ms_inputs:
            if not shared_ca:
                raise ValueError('shared_ca must be true when concat_ms_inputs is true')

        if shared_ca:
            self.ca = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads)
        else:
            self.ca_list = nn.ModuleList([CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads) for _ in range(scale+1)])
        if not concat_ms_inputs:
            if mlp_or_ca == 'mlp':
                self.linear = nn.Conv1d(v_out*(scale+1), q_in, 1, bias=False)
            elif mlp_or_ca == 'ca':
                self.ca_aggregation = CrossAttention(q_in, q_out, k_in, k_out, v_in, v_out, num_heads)
            else:
                raise ValueError(f'mlp_or_ca should be mlp or ca, but got {mlp_or_ca}')
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, x, neighbor_list):
        if self.concat_ms_inputs:
            neighbors = torch.concat(neighbor_list, dim=1)
            x_out = self.ca(x, neighbors)
        else:
            x_output_list = []
            if self.shared_ca:
                for neighbors in neighbor_list:
                    # x.shape == (B, C, N)  neighbors.shape == (B, C, N, K)
                    x_out = self.ca(x, neighbors)
                    # x_out.shape == (B, C, N)
                    x_output_list.append(x_out)
            else:
                for neighbors, ca in zip(neighbor_list, self.ca_list):
                    # x.shape == (B, C, N)  neighbors.shape == (B, C, N, K)
                    x_out = ca(x, neighbors)
                    # x_out.shape == (B, C, N)
                    x_output_list.append(x_out)
            if self.mlp_or_ca == 'mlp':
                x_out = torch.concat(x_output_list, dim=1)
                # x_out.shape == (B, C, N)
                x_out = self.linear(x_out)
                # x_out.shape == (B, C, N)
            else:
                neighbors = torch.stack(x_output_list, dim=-1)
                # x.shape == (B, C, N)   neighbors.shape == (B, C, N, K=scale+1)
                x_out = self.ca_aggregation(x, neighbors)
                # x_out.shape == (B, C, N)
        # x_out.shape == (B, C, N)
        x = self.bn1(x + x_out)
        # x.shape == (B, C, N)
        x_out = self.ff(x)
        # x_out.shape == (B, C, N)
        x = self.bn2(x + x_out)
        # x.shape == (B, C, N)
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


class SelfAttention(nn.Module):
    def __init__(self, q_in, q_out, k_in, k_out, v_in, v_out, num_heads):
        super(SelfAttention, self).__init__()
        # check input values
        if q_in != k_in or q_in != v_in or k_in != v_in:
            raise ValueError(f'q_in, k_in and v_in should be the same! Got q_in:{q_in}, k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(x.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, N)
        x = (attention @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # x.shape == (B, N, H, D)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
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
    def __init__(self, which_emb, linear_emb_in, linear_emb_out, n2p_emb_K, n2p_emb_neighbor_selection_method, n2p_emb_group_type,
                 n2p_emb_q_in, n2p_emb_q_out, n2p_emb_k_in, n2p_emb_k_out, n2p_emb_v_in, n2p_emb_v_out, n2p_emb_num_heads,
                 p2p_emb_q_in, p2p_emb_q_out, p2p_emb_k_in, p2p_emb_k_out, p2p_emb_v_in, p2p_emb_v_out, p2p_emb_num_heads,
                 egdeconv_emb_K, egdeconv_emb_neighbor_selection_method, egdeconv_emb_group_type, egdeconv_emb_conv1_in,
                 egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out, edgeconv_emb_pooling,
                 neighbor2point_enable, neighbor2point_scale, neighbor2point_shared_ca, neighbor2point_concat_ms_inputs, neighbor2point_mlp_or_ca,
                 neighbor2point_K, neighbor2point_neighbor_selection_method, neighbor2point_group_type, neighbor2point_q_in,
                 neighbor2point_q_out, neighbor2point_k_in, neighbor2point_k_out, neighbor2point_v_in, neighbor2point_v_out, neighbor2point_num_heads,
                 neighbor2point_ff_conv1_in, neighbor2point_ff_conv1_out, neighbor2point_ff_conv2_in, neighbor2point_ff_conv2_out,
                 edgeconv_K, edgeconv_neighbor_selection_method, edgeconv_group_type, edgeconv_conv1_in, edgeconv_conv1_out,
                 edgeconv_conv2_in, edgeconv_conv2_out, edgeconv_pooling, p2point_enable, p2point_use_embedding, p2point_embed_in, p2point_embed_out,
                 p2point_qkv_channels, p2point_ff_conv1_in, p2point_ff_conv1_out, p2point_ff_conv2_in, p2point_ff_conv2_out,
                 conv_block_channels_in, conv_block_channels_out):

        super(ShapeNetModel, self).__init__()
        self.neighbor2point_enable = neighbor2point_enable
        self.which_emb = which_emb

        if neighbor2point_enable:
            if which_emb == 'linear':
                self.embedding = LinearEmbedding(linear_emb_in, linear_emb_out)
            elif which_emb == 'n2p':
                self.embedding = Neighbor2PointEmbedding(n2p_emb_K, n2p_emb_neighbor_selection_method, n2p_emb_group_type, n2p_emb_q_in, n2p_emb_q_out, n2p_emb_k_in, n2p_emb_k_out, n2p_emb_v_in, n2p_emb_v_out, n2p_emb_num_heads)
            elif which_emb == 'p2p':
                self.embedding = Point2PointEmbedding(p2p_emb_q_in, p2p_emb_q_out, p2p_emb_k_in, p2p_emb_k_out, p2p_emb_v_in, p2p_emb_v_out, p2p_emb_num_heads)
            elif which_emb == 'edgeconv':
                self.embedding = EdgeConv(egdeconv_emb_K, egdeconv_emb_neighbor_selection_method, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out, edgeconv_emb_pooling)
            else:
                raise ValueError(f'which_emb should be linear, n2p or p2p. Got {which_emb}')
            self.neighbor2point_or_edgeconv = Neighbor2PointAttentionBlock(neighbor2point_scale, neighbor2point_shared_ca, neighbor2point_concat_ms_inputs, neighbor2point_mlp_or_ca,
                                                                           neighbor2point_K, neighbor2point_neighbor_selection_method, neighbor2point_group_type,
                                                                           neighbor2point_q_in, neighbor2point_q_out, neighbor2point_k_in, neighbor2point_k_out,
                                                                           neighbor2point_v_in, neighbor2point_v_out, neighbor2point_num_heads, neighbor2point_ff_conv1_in,
                                                                           neighbor2point_ff_conv1_out, neighbor2point_ff_conv2_in, neighbor2point_ff_conv2_out)
            x_cat_channels = sum(neighbor2point_ff_conv2_out)
        else:
            self.neighbor2point_or_edgeconv = EdgeConvBlock(edgeconv_K, edgeconv_neighbor_selection_method, edgeconv_group_type, edgeconv_conv1_in, edgeconv_conv1_out, edgeconv_conv2_in, edgeconv_conv2_out, edgeconv_pooling)
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
        coordinate = x[...].detach()
        # x.shape == (B, C=3, N)  category_id.shape == (B, 16, 1)
        B, C, N = x.shape
        # x.shape == (B, C=3, N)
        if self.neighbor2point_enable:
            if self.which_emb == 'n2p' or self.which_emb == 'edgeconv':
                x = self.embedding(x, coordinate)
            else:
                x = self.embedding(x)
            # x.shape == (B, C, N)
        x, x_list = self.neighbor2point_or_edgeconv(x, coordinate)
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
