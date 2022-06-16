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


def group(pcd, xyz, K, xyz_or_feature, feature_or_diff, cross_attention):
    pcd_clone = pcd[..., None]  # pcd_clone.shape == (B, C, N, 1)
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    xyz = xyz.permute(0, 2, 1)  # xyz.shape == (B, N, C)

    if xyz_or_feature == 'xyz':
        dists = square_distance(xyz, xyz)  # dists.shape == (B, N, N)
    elif xyz_or_feature == 'feature':
        dists = square_distance(pcd, pcd)  # dists.shape == (B, N, N)
    else:
        raise ValueError(f'xyz_or_feature should be "xyz" or "feature", but got {xyz_or_feature}')

    idx = dists.argsort()[:, :, :K]  # idx.shape == (B, N, K)
    neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)

    if feature_or_diff == 'xyz':
        pcd = torch.cat([neighbors, pcd[:, :, None, :].repeat(1, 1, K, 1)], dim=-1).permute(0, 3, 1, 2)  # pcd.shape == (B, 2C, N, K)
        neighbors = neighbors.permute(0, 3, 1, 2)  # neighbors.shape == (B, C, N, K)
    elif feature_or_diff == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        pcd = torch.cat([diff, pcd[:, :, None, :].repeat(1, 1, K, 1)], dim=-1).permute(0, 3, 1, 2)  # pcd.shape == (B, 2C, N, K)
        neighbors = diff.permute(0, 3, 1, 2)  # diff.shape == (B, C, N, K)
    else:
        raise ValueError(f'feature_or_diff should be "feature" or "diff", but got {feature_or_diff}')

    # the group function is used for cross attention input or edgeconv input
    if cross_attention:
        pcd = pcd_clone  # pcd.shape == (B, C, N, 1)
    return pcd, neighbors
