"""Reusable ELECTRA + ScalarMix model components for DANN experiments."""

import math
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Parameter, ParameterList
from transformers import AutoModel


class ScalarMix(nn.Module):
    """Learn a weighted mixture over all hidden states from a transformer."""

    def __init__(self, mixture_size: int, trainable: bool = True) -> None:
        super().__init__()
        self.scalar_parameters = ParameterList(
            [Parameter(torch.zeros(1), requires_grad=trainable) for _ in range(mixture_size)]
        )
        self.gamma = Parameter(torch.ones(1), requires_grad=trainable)

    def forward(self, tensors: List[Tensor]) -> Tensor:
        weights = torch.nn.functional.softmax(
            torch.cat([p for p in self.scalar_parameters]), dim=0
        )
        weights = torch.split(weights, 1)
        return self.gamma * sum(weight * tensor for weight, tensor in zip(weights, tensors))


def grl_lambda_schedule(progress: float, max_lambda: float = 1.0) -> float:
    """DANN schedule from Ganin et al.; progress should be in [0, 1]."""

    progress = float(min(1.0, max(0.0, progress)))
    return float(max_lambda) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


class GradientReversalFunction(torch.autograd.Function):
    """Forward identity, backward gradient reversal."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def apply_gradient_reversal(x: Tensor, lambda_: float) -> Tensor:
    return GradientReversalFunction.apply(x, float(lambda_))


def build_domain2id(sources: List[str]) -> Tuple[Dict[str, int], int]:
    unique = sorted(set(str(source) for source in sources))
    return {source: idx for idx, source in enumerate(unique)}, len(unique)


class MLPHead(nn.Module):
    """Small feed-forward classification head used by task and domain branches."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DifficultyClassifierHead(MLPHead):
    def __init__(self, in_dim: int, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__(in_dim=in_dim, out_dim=num_classes, dropout=dropout)


class DomainClassifierHead(MLPHead):
    def __init__(self, in_dim: int, num_domains: int, dropout: float = 0.1) -> None:
        super().__init__(in_dim=in_dim, out_dim=num_domains, dropout=dropout)


class ElectraScalarMixClassifier(nn.Module):
    """ELECTRA + ScalarMix + mean pooling + 3-class difficulty head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        n_layers = int(self.encoder.config.num_hidden_layers) + 1
        self.scalar_mix = ScalarMix(n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def encode_pooled(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        mixed = self.dropout(self.scalar_mix(list(out.hidden_states)))
        mask = attention_mask.unsqueeze(-1).float()
        summed = (mixed * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.head(self.encode_pooled(input_ids, attention_mask))


class ElectraScalarMixDANN(nn.Module):
    """Text-only ELECTRA DANN model with difficulty and domain heads."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 3,
        num_domains: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        n_layers = int(self.encoder.config.num_hidden_layers) + 1
        self.scalar_mix = ScalarMix(n_layers)
        self.dropout = nn.Dropout(dropout)
        self.difficulty_head = DifficultyClassifierHead(hidden, num_classes, dropout)
        self.domain_head = DomainClassifierHead(hidden, num_domains, dropout)

    def encode_pooled(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        mixed = self.dropout(self.scalar_mix(list(out.hidden_states)))
        mask = attention_mask.unsqueeze(-1).float()
        summed = (mixed * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        grl_lambda: float,
    ) -> Tuple[Tensor, Tensor]:
        pooled = self.encode_pooled(input_ids, attention_mask)
        diff_logits = self.difficulty_head(pooled)
        dom_logits = self.domain_head(apply_gradient_reversal(pooled, grl_lambda))
        return diff_logits, dom_logits

    def difficulty_logits_only(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.difficulty_head(self.encode_pooled(input_ids, attention_mask))

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder_except_top(self, n_layers: int) -> None:
        self.freeze_encoder()
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            for layer in self.encoder.encoder.layer[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        for param in self.scalar_mix.parameters():
            param.requires_grad = True


class ElectraStaticDANN(nn.Module):
    """ELECTRA DANN model that fuses pooled text embeddings with static features."""

    def __init__(
        self,
        model_name: str,
        static_dim: int,
        num_classes: int = 3,
        num_domains: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        n_layers = int(self.encoder.config.num_hidden_layers) + 1
        self.scalar_mix = ScalarMix(n_layers)
        self.dropout = nn.Dropout(dropout)
        fused_dim = hidden + int(static_dim)
        self.difficulty_head = MLPHead(fused_dim, num_classes, dropout)
        self.domain_head = MLPHead(fused_dim, num_domains, dropout)

    def encode_pooled(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        mixed = self.dropout(self.scalar_mix(list(out.hidden_states)))
        mask = attention_mask.unsqueeze(-1).float()
        summed = (mixed * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1e-9)

    def fused(self, input_ids: Tensor, attention_mask: Tensor, static_features: Tensor) -> Tensor:
        pooled = self.encode_pooled(input_ids, attention_mask)
        return torch.cat([pooled, static_features.float()], dim=1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        static_features: Tensor,
        grl_lambda: float,
    ) -> Tuple[Tensor, Tensor]:
        fused = self.fused(input_ids, attention_mask, static_features)
        diff_logits = self.difficulty_head(fused)
        dom_logits = self.domain_head(apply_gradient_reversal(fused, grl_lambda))
        return diff_logits, dom_logits

    def difficulty_logits_only(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        static_features: Tensor,
    ) -> Tensor:
        return self.difficulty_head(self.fused(input_ids, attention_mask, static_features))

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder_except_top(self, n_layers: int) -> None:
        self.freeze_encoder()
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            for layer in self.encoder.encoder.layer[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        for param in self.scalar_mix.parameters():
            param.requires_grad = True


def combined_loss(
    diff_logits: Tensor,
    dom_logits: Tensor,
    labels: Tensor,
    domains: Tensor,
    domain_loss_alpha: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Difficulty loss plus weighted domain-classifier loss."""

    ce_task = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ce_dom = nn.CrossEntropyLoss()
    diff_loss = ce_task(diff_logits, labels)
    dom_loss = ce_dom(dom_logits, domains)
    total = diff_loss + float(domain_loss_alpha) * dom_loss
    return total, diff_loss, dom_loss
