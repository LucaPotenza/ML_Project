# Loss Function

def contrastive_loss(sketch_emb, view_emb, sketch_peer_emb, view_peer_emb,
                     sketch_labels, view_labels,
                     Cpos=0.2, Cneg=10.0, alpha=1.0, sw=0.5, vw=0.5,
                     mask_pos=None, mask_neg=None):
    """
    Complete loss with intra-modal regularization.

    Args:
        sketch_emb: main sketch embeddings (batch, code_len)
        view_emb: main view embeddings (batch, code_len)
        sketch_peer_emb: peer sketch embeddings (batch, code_len)
        view_peer_emb: peer view embeddings (batch, code_len)
        sketch_labels: sketch labels (batch,)
        view_labels: view labels (batch,)
        Cpos, Cneg: positive/negative loss weights
        alpha: exp() parameter in the negative loss
        sw, vw: intra-modal sketch/view regularization weights
        mask_pos, mask_neg: masks for stochastic sampling
    """
    batch_size = sketch_emb.size(0)
    device = sketch_emb.device

    # Matrix indicating same class (0) or different class (1) for all combinations of i and j
    # y_mat[i,j] = 1 if sketch_i and view_j have different classes
    y_mat = (sketch_labels.unsqueeze(1) != view_labels.unsqueeze(0)).float()

    # Sampling masks (if not provided, use all pairs)
    # Masks allow for stochastic sampling:
    # you can choose to sample only a subset of negatives to prevent cost_neg
    # from dominating cost_pos, making training more stable.
    if mask_pos is None:
        mask_pos = torch.ones_like(y_mat)
    if mask_neg is None:
        mask_neg = torch.ones_like(y_mat)

    # Count positive and negative pairs
    numpos = torch.sum((1 - y_mat) * mask_pos)
    numneg = torch.sum(y_mat * mask_neg)

    # ===== CROSS-MODAL LOSS (sketch-view) =====
    # L1 distance between sketch and view
    l1norm = torch.cdist(sketch_emb, view_emb, p=1)  # (batch, batch)
    diff = l1norm ** 2

    # Positive loss: minimizes distance for the same class
    dpos = Cpos * diff * (1 - y_mat)

    # Negative loss: maximizes distance for different classes using exp
    dneg = Cneg * torch.exp(-alpha / Cneg * l1norm) * y_mat

    # ===== INTRA-MODAL SKETCH LOSS (sketch-sketch_peer) =====
    l1norm_sp = torch.cdist(sketch_emb, sketch_peer_emb, p=1)
    diff_sp = l1norm_sp ** 2
    dpos_sp = sw * Cpos * diff_sp * (1 - y_mat)
    dneg_sp = Cneg * torch.exp(-alpha / Cneg * l1norm_sp) * y_mat

    # ===== INTRA-MODAL VIEW LOSS (view-view_peer) =====
    l1norm_vp = torch.cdist(view_emb, view_peer_emb, p=1)
    diff_vp = l1norm_vp ** 2
    dpos_vp = vw * Cpos * diff_vp * (1 - y_mat)
    dneg_vp = Cneg * torch.exp(-alpha / Cneg * l1norm_vp) * y_mat

    # ===== TOTAL LOSS =====
    cost_pos = torch.sum((dpos + dpos_sp + dpos_vp) * mask_pos)
    cost_neg = torch.sum((dneg + dneg_sp + dneg_vp) * mask_neg)

    # Normalize by the number of pairs
    if numpos > 0:
        cost_pos = cost_pos / numpos
    if numneg > 0:
        cost_neg = cost_neg / numneg

    cost = cost_pos + cost_neg

    # Metrics for monitoring
    posdiff = torch.sum(l1norm * (1 - y_mat) * mask_pos) / (numpos + 1e-8)
    negdiff = torch.sum(l1norm * y_mat * mask_neg) / (numneg + 1e-8)

    metrics = {
        'cost': cost.item(),
        'cost_pos': cost_pos.item(),
        'cost_neg': cost_neg.item(),
        'dpos': torch.sum(dpos * mask_pos).item(),
        'dneg': torch.sum(dneg * mask_neg).item(),
        'dpos_sp': torch.sum(dpos_sp * mask_pos).item(),
        'dneg_sp': torch.sum(dneg_sp * mask_neg).item(),
        'dpos_vp': torch.sum(dpos_vp * mask_pos).item(),
        'dneg_vp': torch.sum(dneg_vp * mask_neg).item(),
        'posdiff': posdiff.item(),
        'negdiff': negdiff.item(),
        'numpos': numpos.item(),
        'numneg': numneg.item()
    }

    return cost, metrics