import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import deque

# --- MoCo Encoder Wrapper ---
class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class MoCoEncoder(nn.Module):
    def __init__(self, base_encoder='resnet18', out_dim=128):
        super().__init__()
        encoder = getattr(models, base_encoder)(pretrained=False)
        self.backbone = nn.Sequential(*list(encoder.children())[:-1])  # Remove FC layer
        dim_mlp = encoder.fc.in_features
        self.head = MLPHead(dim_mlp, out_dim)

    def forward(self, x):
        feat = self.backbone(x).squeeze()
        return F.normalize(self.head(feat), dim=1)


# --- MoCo v2 Model ---
class MoCo(nn.Module):
    def __init__(self, base_encoder='resnet18', dim=128, K=4096, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        # encoders
        self.encoder_q = MoCoEncoder(base_encoder, dim)
        self.encoder_k = MoCoEncoder(base_encoder, dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        assert self.K % batch_size == 0  # for simplicity

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)

        # logits: [B, 1+K]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)

        return logits, labels



