import torch


def square_distance(src, dst):
    """
    Input:
        src: source points, [B, M, C]
        dst: target points, [B, N, C]
    Output:
        dist: point-wise square distance, [B, M, N]
    """
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, M, K]
    Return:
        new_points: indexed points data, [B, M, K, C]
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def group(pcd, xyz, K):
    # dists.shape == (B, N, N)
    dists = square_distance(xyz, xyz)
    # idx.shape == (B, N, K)
    idx = dists.argsort()[:, :, :K]
    # grouped_points.shape == (B, N, K, C)
    grouped_points = index_points(pcd, idx)
    # diff.shape == (B, N, K, C)
    diff = grouped_points - pcd[:, :, None, :]
    # x.shape == (B, N, K, 2C)
    pcd = torch.cat([diff, pcd[:, :, None, :].repeat(1, 1, K, 1)], dim=-1)
    return pcd


def group_for_point2neighbor_attention(pcd, xyz, K):
    # dists.shape == (B, N, N)
    dists = square_distance(xyz, xyz)  # TODO: try another way to group K points, e.g. diff
    # idx.shape == (B, N, K)
    idx = dists.argsort()[:, :, :K]
    # grouped_points.shape == (B, N, K, C)
    grouped_points = index_points(pcd, idx)
    return pcd, grouped_points
