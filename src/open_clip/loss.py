import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def cosine_similarity(x, y, _):
    return x @ y.T


def nsphere_arc(x, y, _):
    return -torch.acos(x @ y.T)


def euclidean_distance(x, y, _):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    squared = (x - y) ** 2
    return -squared.sum(-1).sqrt()


def _exponential_map(x, curvature):
    c_sqrt_norm = torch.sqrt(curvature) * x.norm(dim=1, keepdim=True)
    x_space = torch.sinh(c_sqrt_norm) / c_sqrt_norm * x
    x_time = torch.sqrt(curvature.reciprocal() + (x_space ** 2).sum(-1))
    return x_space, x_time


def lorentzian_distance(x, y, curvature):
    # FP32 for exponential map and losses for numerical stability,
    # per https://arxiv.org/abs/2304.09172
    x, y, curvature = x.double(), y.double(), curvature.double()
    x_space, x_time = _exponential_map(x, curvature)
    y_space, y_time = _exponential_map(y, curvature)
    return -torch.rsqrt(curvature) * torch.acosh(-curvature * (x_space @ y_space.T - torch.outer(x_time, y_time)))


METRICS = {
    'clip': cosine_similarity,
    'elliptic': nsphere_arc,
    'euclidean': euclidean_distance,
    'hyperbolic': lorentzian_distance,
}


def euclidean_entailment(x, y, _, K):
    # https://arxiv.org/abs/1804.01882
    y_x = y - x
    x_norm = x.norm(dim=1)
    ext = torch.acos((y_x * x).sum(-1) / (y_x.norm(dim=1) * x_norm))
    if not K:
        return ext.mean()
    aper = torch.asin(torch.clamp(K / x_norm, max=1.))
    return torch.clamp(ext - aper, min=0.).mean()


def hyperbolic_entailment(x, y, curvature, K):
    # FP32 for exponential map and losses for numerical stability,
    # per https://arxiv.org/abs/2304.09172
    x, y, curvature = x.double(), y.double(), curvature.double()
    x_space, x_time = _exponential_map(x, curvature)
    x_space_norm = x_space.norm(dim=1)
    y_space, y_time = _exponential_map(y, curvature)
    l = (x_space * y_space).sum(-1) - x_time * y_time
    c_l = curvature * l
    ext = torch.acos((y_time + x_time * c_l) / (x_space_norm * torch.sqrt(c_l ** 2 - 1.)))
    if not K:
        return ext.mean()
    aper = torch.asin(torch.clamp(2 * K / (torch.sqrt(curvature) * x_space_norm), max=1.))
    return torch.clamp(ext - aper, min=0.).mean()


_ENTAILMENT = {
    'euclidean': euclidean_entailment,
    'hyperbolic': hyperbolic_entailment,    
}


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            geometry='clip',
            entailment_weight=0.0,
            K=0.1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.metric = METRICS[geometry]
        if geometry in _ENTAILMENT:
            self.entailment = _ENTAILMENT[geometry]
            self.entailment_weight = entailment_weight
            self.K = K
        else:
            self.entailment_weight = 0.0
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, curvature):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * self.metric(image_features, all_text_features, curvature)
                logits_per_text = logit_scale * self.metric(text_features, all_image_features, curvature)
            else:
                logits_per_image = logit_scale * self.metric(all_image_features, all_text_features, curvature)
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * self.metric(image_features, text_features, curvature)
            logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, curvature, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale, curvature)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        output = {"contrastive_loss": contrastive_loss}

        if self.entailment_weight:
            entailment_loss = self.entailment_weight * self.entailment(
                text_features, image_features, curvature, self.K)
            output["entailment_loss"] = entailment_loss
            total_loss = contrastive_loss + entailment_loss

        return output if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss
