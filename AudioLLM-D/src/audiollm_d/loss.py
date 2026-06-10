"""Focal / tiered token-weighted loss for disfluency-aware fine-tuning.

The training objective is causal-LM cross-entropy over the assistant transcription.
On top of that we apply two corrections aimed at the clinically relevant disfluency
tokens, which are rare relative to ordinary words:

  * a per-token alpha weight (``weight_by_id``) so frequency-tiered disfluencies can
    be upweighted relative to common words and to each other, and
  * the focal ``(1 - p)^gamma`` factor that down-weights already-confident tokens so
    gradient mass concentrates on the hard, still-misclassified ones.

With ``gamma == 0`` and a uniform (all-1.0 / empty) weight map this reduces *exactly*
to mean cross-entropy, preserving the base objective. That identity is unit-tested and
is the safety check called for in the loss upgrade.
"""

from __future__ import annotations

from collections.abc import Mapping


def disfluency_focal_loss(
    logits,
    labels,
    *,
    weight_by_id: Mapping[int, float] | None = None,
    gamma: float = 0.0,
    focal_ids=None,
    ignore_index: int = -100,
):
    """Causal-LM focal loss with per-token alpha weights.

    Args:
        logits: ``[batch, seq, vocab]`` model logits (un-shifted).
        labels: ``[batch, seq]`` target ids (un-shifted); ``ignore_index`` positions
            (prompt + padding) are excluded from the loss.
        weight_by_id: optional ``{token_id: alpha}`` map. Targets whose id is present
            get that alpha; everything else gets ``1.0``.
        gamma: focal focusing parameter. ``0`` gives plain (weighted) cross-entropy.
        focal_ids: optional iterable of token ids that the ``(1-p)^gamma`` focal
            factor is *restricted* to. When ``None`` (default) the focal factor applies
            to every target — the original v3 behaviour. When provided, only targets
            whose id is in this set are focal-modulated; all other targets keep plain
            (alpha-weighted) cross-entropy. This is the v3.5 "focal on the rare brackets
            only, flat loss on fillers" knob. Has no effect when ``gamma == 0``.
        ignore_index: label value to skip (matches the collator's masked prompt id).

    Returns:
        Scalar loss tensor, averaged over non-ignored target positions.
    """
    import torch
    import torch.nn.functional as F

    # Standard causal-LM shift: predict token t+1 from positions up to t.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    ce = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = flat_labels.ne(ignore_index)

    alpha = torch.ones_like(ce)
    if weight_by_id:
        ids = torch.tensor(sorted(weight_by_id), dtype=torch.long, device=flat_labels.device)
        weights = torch.tensor(
            [float(weight_by_id[int(i)]) for i in ids],
            dtype=ce.dtype,
            device=ce.device,
        )
        match = flat_labels[..., None] == ids  # [N, K]
        token_alpha = (match.to(ce.dtype) * weights).sum(dim=-1)
        alpha = torch.where(match.any(dim=-1), token_alpha, alpha)

    factor = alpha
    if gamma:
        # p_t = exp(-CE) is the softmax probability of the true class.
        p = torch.exp(-ce)
        focal = (1.0 - p).clamp_min(0.0).pow(gamma)
        if focal_ids:
            # Restrict the focal modulation to the listed token ids (v3.5): every
            # other target falls back to focal=1.0, i.e. plain alpha-weighted CE.
            fids = torch.tensor(sorted(focal_ids), dtype=torch.long, device=flat_labels.device)
            is_focal = (flat_labels[..., None] == fids).any(dim=-1)
            focal = torch.where(is_focal, focal, torch.ones_like(focal))
        factor = alpha * focal

    return (ce * factor * valid).sum() / valid.sum().clamp_min(1)
