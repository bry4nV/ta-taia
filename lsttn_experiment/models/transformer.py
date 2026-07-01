import torch
import torch.nn as nn


class MaskedSubseriesTransformer(nn.Module):
    """MST con encoder visible y cabeza de reconstrucción contextual."""

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        long_len: int,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
    ):
        super().__init__()
        if long_len % patch_size:
            raise ValueError("long_len debe ser divisible entre patch_size")
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = long_len // patch_size
        self.mask_ratio = mask_ratio

        self.patch_embedding = nn.Linear(patch_size, d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # El paper usa una capa Transformer como cabeza autosupervisada.
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.reconstruction_head = nn.Linear(d_model, patch_size)

    def _patchify(self, inputs):
        batch, length, nodes = inputs.shape
        patches = inputs.permute(0, 2, 1).reshape(
            batch * nodes, self.num_patches, self.patch_size
        )
        return patches, batch, nodes

    def encode(self, inputs):
        patches, batch, nodes = self._patchify(inputs)
        tokens = self.patch_embedding(patches) + self.position_embedding
        encoded = self.encoder(self.input_dropout(tokens))
        return encoded.reshape(batch, nodes, self.num_patches, self.d_model)

    def encode_subseries(self, inputs, position: int = 0):
        """Codifica una hora externa (p. ej. referencia semanal)."""
        batch, length, nodes = inputs.shape
        if length != self.patch_size:
            raise ValueError("La subserie externa debe contener exactamente un patch")
        patch = inputs.permute(0, 2, 1).reshape(batch * nodes, 1, self.patch_size)
        token = self.patch_embedding(patch) + self.position_embedding[:, position : position + 1]
        return self.encoder(token).reshape(batch, nodes, self.d_model)

    def pretrain_forward(self, inputs):
        patches, _, _ = self._patchify(inputs)
        device = patches.device
        number_masked = int(self.num_patches * self.mask_ratio)
        permutation = torch.randperm(self.num_patches, device=device)
        masked = permutation[:number_masked].sort().values
        visible = permutation[number_masked:].sort().values

        visible_tokens = self.patch_embedding(patches[:, visible])
        visible_tokens = visible_tokens + self.position_embedding[:, visible]
        visible_hidden = self.encoder(self.input_dropout(visible_tokens))

        # Se reconstruye la secuencia completa antes de decodificar. Así cada
        # máscara puede atender a todos los tokens visibles.
        full_hidden = self.mask_token.expand(patches.size(0), self.num_patches, -1).clone()
        full_hidden = full_hidden + self.position_embedding
        full_hidden[:, visible] = visible_hidden
        decoded = self.decoder(full_hidden)

        reconstruction = self.reconstruction_head(decoded[:, masked])
        labels = patches[:, masked]
        return reconstruction, labels

    def forward(self, inputs, mode: str = "inference"):
        if mode == "pretrain":
            return self.pretrain_forward(inputs)
        if mode == "inference":
            return self.encode(inputs)
        raise ValueError(f"Modo desconocido: {mode}")
