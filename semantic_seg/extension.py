# coding=utf-8
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict


def prototype_seed_indices(score, fea, weak_label, num_sp_list,
                           confidence_th=0.85, prototype_margin=0.05,
                           max_per_class=3, ignore_label=255):
    """Seed fully unlabeled scene graphs without consulting their ground truth."""
    weak_label = weak_label.reshape(-1)
    if score.shape[0] != weak_label.shape[0] or sum(num_sp_list) != score.shape[0]:
        raise ValueError("scene sizes, scores, and weak labels must align")
    if max_per_class < 0:
        raise ValueError("max_per_class must be non-negative")
    labeled_mask = weak_label != ignore_label
    if not torch.any(labeled_mask) or max_per_class == 0:
        empty_idx = torch.empty(0, dtype=torch.long, device=score.device)
        empty_label = torch.empty(0, dtype=torch.long, device=score.device)
        return empty_idx, empty_label

    num_classes = score.shape[1]
    normalized_features = F.normalize(fea.detach(), p=2, dim=1)
    prototypes = []
    available = []
    for class_id in range(num_classes):
        class_mask = labeled_mask & (weak_label == class_id)
        available.append(bool(torch.any(class_mask)))
        if available[-1]:
            prototype = normalized_features[class_mask].mean(dim=0, keepdim=True)
            prototypes.append(F.normalize(prototype, p=2, dim=1))
        else:
            prototypes.append(torch.zeros_like(normalized_features[:1]))
    prototypes = torch.cat(prototypes, dim=0)
    similarity = torch.mm(normalized_features, prototypes.t())
    available_mask = torch.tensor(available, dtype=torch.bool, device=score.device)
    similarity[:, ~available_mask] = -2.0

    probabilities = F.softmax(score.detach(), dim=1)
    confidence, prediction = probabilities.max(dim=1)
    top_similarity, top_class = similarity.max(dim=1)
    if num_classes > 1:
        second_similarity = similarity.topk(2, dim=1).values[:, 1]
    else:
        second_similarity = torch.full_like(top_similarity, -1.0)
    eligible = (
        (confidence >= confidence_th)
        & (prediction == top_class)
        & ((top_similarity - second_similarity) >= prototype_margin)
        & (~labeled_mask)
    )

    selected_indices = []
    selected_labels = []
    scene_start = 0
    for scene_size in num_sp_list:
        scene_end = scene_start + int(scene_size)
        if not torch.any(labeled_mask[scene_start:scene_end]):
            for class_id in range(num_classes):
                local = torch.nonzero(
                    eligible[scene_start:scene_end]
                    & (prediction[scene_start:scene_end] == class_id),
                    as_tuple=False,
                ).reshape(-1)
                if local.numel() > 0:
                    candidate = local + scene_start
                    ranking = confidence[candidate] + 0.1 * top_similarity[candidate]
                    keep = min(max_per_class, candidate.numel())
                    chosen = candidate[torch.topk(ranking, keep).indices]
                    selected_indices.append(chosen)
                    selected_labels.append(torch.full(
                        (keep,), class_id, dtype=torch.long, device=score.device))
        scene_start = scene_end

    if not selected_indices:
        empty_idx = torch.empty(0, dtype=torch.long, device=score.device)
        empty_label = torch.empty(0, dtype=torch.long, device=score.device)
        return empty_idx, empty_label
    return torch.cat(selected_indices), torch.cat(selected_labels)


def extension_accum2(input, score, fea, weak_label, edges, th=0.7, undirected=True, ext_max=80):
    # input: Nsp*c*n
    # score: Nsp*num_classes
    # fea: Nsp*c
    # weak_label: Nsp

    weak_label = weak_label.reshape(-1) # Nsp

    score_soft = F.softmax(score, dim=-1) # Nsp*num_classes
    Nsp = input.shape[0]

    mask_label = (weak_label < 255).reshape(-1) # Nsp
    mask_unlabel = (weak_label == 255).reshape(-1) # Nsp
    labeled_idx = torch.nonzero(mask_label, as_tuple=False).reshape(-1).long()
    unlabel_idx = torch.nonzero(mask_unlabel, as_tuple=False).reshape(-1).long()

    candidates = defaultdict(list)
    for i in range(edges.shape[0]):
        x = edges[i, 0].item()
        y = edges[i, 1].item()
        if x != y:
            candidates[x].append(y)
            if undirected:
                candidates[y].append(x)


    weak_label2 = []
    score2 = []
    extend_idx = []
    for i in range(labeled_idx.shape[0]):
        sp_idx = labeled_idx[i] # idx of the labeled sp
        sp_wl = weak_label[sp_idx]

        neighbors = candidates[sp_idx.item()] 
        neighbor_score = -1
        neighbor_ext = -1
        for nei in neighbors:
            if weak_label[nei] == 255:
                score_vec_point = score_soft[nei] # 13
                s, p = torch.max(score_vec_point, 0)
                if (p==sp_wl) & (s>neighbor_score):
                    neighbor_ext = nei 
                    neighbor_score = s

        if (neighbor_score>th) & (neighbor_ext not in extend_idx) & (neighbor_ext not in labeled_idx):
            weak_label2.append(sp_wl.unsqueeze(0))
            score2.append(score[neighbor_ext, :].unsqueeze(0))
            extend_idx.append(neighbor_ext)

    if len(extend_idx)>0:
        weak_label2 = torch.cat(weak_label2)
        score2 = torch.cat(score2, 0)
    extend_idx = torch.tensor(extend_idx)


    if extend_idx.shape[0]>ext_max:
        pred_v, _ = torch.max(score2, 1)
        _, indices = torch.sort(pred_v, 0, descending=True)

        weak_label2_sample = weak_label2[indices[:ext_max]]
        score2_sample = score2[indices[:ext_max], :]
        extend_idx_sample = extend_idx[indices[:ext_max]]
    else:
        weak_label2_sample = weak_label2
        score2_sample = score2
        extend_idx_sample = extend_idx

    score1 = score[mask_label, :]
    weak_label1 = weak_label[mask_label]
    
    return score1, weak_label1, score2_sample, weak_label2_sample, extend_idx_sample, labeled_idx
