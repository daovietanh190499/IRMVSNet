import torch
import numpy as np
import torch.nn.functional as F

def get_homographies(intrinsics, extrinsics, depths, scale=4):
    M = len(depths)
    # define source K, R, t
    src_Ks = intrinsics
    src_Ks = intrinsics / scale # scaled intrinsic for feature domain
    src_Ks[:, 2, 2] = 1
    src_Rs = extrinsics[:, :3, :3]
    src_ts = extrinsics[:, :3, 3:]
    src_Ks = src_Ks.unsqueeze(1)
    src_Rs = src_Rs.unsqueeze(1)
    src_ts = src_ts.unsqueeze(1)
    src_Rts = src_Rs.transpose(2, 3)
    src_Cs = -src_Rts.matmul(src_ts)
    src_KIs = torch.inverse(src_Ks)

    # define ref K, R, t
    ref_K = src_Ks[:1]
    ref_R = src_Rs[:1]
    ref_t = src_ts[:1]
    ref_Rt = src_Rts[:1]
    ref_KI = src_KIs[:1]
    ref_C = src_Cs[:1]

    fronto_direction = ref_R[:, :, 2:3, :3] # N x 1 x 1 x 3
    rel_C = src_Cs[1:] - ref_C # N x 1 x 3 x 1

    # compute h
    depth_mat = depths.view(1, M, 1, 1)
    trans_mat = torch.eye(3, device=intrinsics.device).view(1, 1, 3, 3) - rel_C.matmul(fronto_direction) / depth_mat
    return src_Ks[1:].matmul(src_Rs[1:]).matmul(trans_mat).matmul(ref_Rt).matmul(ref_KI)

def homography_transform(feat, homography, current_pixel_coords):
    B, C_H, H, W = feat.size()
    pcoords = homography.bmm(current_pixel_coords)
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)
    X_norm = 2 * (X / Z) / (W - 1) - 1  # Normalized
    Y_norm = 2 * (Y / Z) / (H - 1) - 1  #
    pixel_coords = torch.stack([X_norm, Y_norm], dim=2).view(B, H, W, 2)  # [B, H, W, 2]
    targetimg_fea_t = F.grid_sample(feat, pixel_coords, align_corners=True)
    return targetimg_fea_t

def get_pixel_coords(feat):
    B, CH, H, W = feat.size()
    i_range = torch.arange(0, H).view(1, H, 1).expand(1, H, W).type_as(feat)  # [1, H, W]
    j_range = torch.arange(0, W).view(1, 1, W).expand(1, H, W).type_as(feat)  # [1, H, W]
    ones = torch.ones(1, H, W).type_as(feat)
    pixel_coords_ = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
    current_pixel_coords = pixel_coords_[:, :, :H, :W].expand(B, 3, H, W).contiguous().view(B, 3, -1)
    return current_pixel_coords

def init_pixel_coords(H, W):
    i_range = torch.arange(0, H).view(1, H, 1).expand(1, H, W).type(torch.float)  # [1, H, W]
    j_range = torch.arange(0, W).view(1, 1, W).expand(1, H, W).type(torch.float)  # [1, H, W]
    ones = torch.ones(1, H, W).type(torch.float)
    pixel_coords_ = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
    current_pixel_coords = pixel_coords_[:, :, :H, :W].expand(1, 3, H, W).contiguous().view(1, 3, -1)
    return current_pixel_coords