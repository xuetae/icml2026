import torch
from torchvision.ops import roi_align


def frequency_consistency_loss(
    feats0,
    fg_mask,
    pred_bboxes,
    target_bboxes,
    stride_tensor,
    stride0,
    beta=0.05,
    roi_size=16,
    lam=1.0,
):
    """
    Core L_freq implementation extracted from the project loss logic.

    Args:
        feats0: highest-resolution feature map, shape [B, C, H, W]
        fg_mask: positive mask, shape [B, N]
        pred_bboxes: predicted boxes in stride units, [B, N, 4]
        target_bboxes: target boxes in stride units, [B, N, 4]
        stride_tensor: anchor strides, [N, 1]
        stride0: stride for feats0 scale (e.g., self.stride[0])
        beta: loss weight in total loss (paper uses 0.05)
        roi_size: pooled ROI size (default 16)
        lam: frequency radial weight slope in w(r)=1+lam*r
    """
    if fg_mask.sum() == 0:
        return feats0.new_tensor(0.0)

    pos_pred_bboxes = pred_bboxes[fg_mask]
    pos_target_bboxes = target_bboxes[fg_mask]

    bsz, num_anchors = fg_mask.shape
    stride_expanded = stride_tensor.squeeze(-1).unsqueeze(0).expand(bsz, num_anchors)
    pos_strides = stride_expanded[fg_mask].unsqueeze(-1)  # [num_pos, 1]

    pos_pred_bboxes = pos_pred_bboxes * pos_strides / stride0
    pos_target_bboxes = pos_target_bboxes * pos_strides / stride0

    batch_idx = torch.where(fg_mask)[0]
    pred_rois = torch.cat([batch_idx.unsqueeze(1).float(), pos_pred_bboxes], dim=1)
    target_rois = torch.cat([batch_idx.unsqueeze(1).float(), pos_target_bboxes], dim=1)

    p_i = roi_align(feats0, pred_rois, output_size=(roi_size, roi_size), spatial_scale=1.0)
    t_i = roi_align(feats0, target_rois, output_size=(roi_size, roi_size), spatial_scale=1.0)

    f_p = torch.fft.fft2(p_i, norm="ortho")
    f_t = torch.fft.fft2(t_i, norm="ortho")

    h = roi_size
    w = roi_size
    freq_y = torch.fft.fftfreq(h, d=1.0).view(-1, 1).repeat(1, w)
    freq_x = torch.fft.fftfreq(w, d=1.0).view(1, -1).repeat(h, 1)
    r = torch.sqrt(freq_x**2 + freq_y**2).to(feats0.device)
    w_mat = (1 + lam * r).view(1, 1, h, w)

    diff_mag = torch.abs(f_p - f_t)
    l_freq = (w_mat * diff_mag).pow(2).mean()

    return beta * l_freq
