import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


# ---------------------------------------------------------
# Old autoencoder model kept here only if you still need it
# ---------------------------------------------------------
class ViTEncoderDecoder(nn.Module):
    def __init__(self, vit_model_name="vit_base_patch16_224", image_size=224):
        super().__init__()

        self.encoder = create_model(vit_model_name, pretrained=True)
        self.encoder.reset_classifier(0)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.Upsample(size=(image_size, image_size), mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        tokens = self.encoder.forward_features(x)  # (B, 197, 768)
        patch_tokens = tokens[:, 1:, :]            # remove CLS token

        batch_size, num_patches, hidden_dim = patch_tokens.shape
        h = w = int(num_patches ** 0.5)

        feat_map = patch_tokens.permute(0, 2, 1).contiguous().view(batch_size, hidden_dim, h, w)
        reconstructed = self.decoder(feat_map)
        return reconstructed


def freeze_vit_layers(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.blocks[-1].parameters():
        param.requires_grad = True


# ---------------------------------------------------------
# SSL model
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimSiamViT(nn.Module):
    """
    Encoder + projector + predictor
    """
    def __init__(self, vit_model_name="vit_base_patch16_224", feat_dim=768, proj_dim=256):
        super().__init__()

        self.encoder = create_model(vit_model_name, pretrained=True)
        self.encoder.reset_classifier(0)

        self.projector = MLP(feat_dim, 2048, proj_dim)
        self.predictor = MLP(proj_dim, 512, proj_dim)

    def encode(self, x):
        tokens = self.encoder.forward_features(x)   # (B, 197, 768)
        patch_tokens = tokens[:, 1:, :]             # (B, 196, 768)
        feat = patch_tokens.mean(dim=1)             # (B, 768)
        return feat

    def forward(self, x1, x2):
        f1 = self.encode(x1)
        f2 = self.encode(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

    @torch.no_grad()
    def embed(self, x, normalize=True):
        feat = self.encode(x)
        return F.normalize(feat, dim=1) if normalize else feat