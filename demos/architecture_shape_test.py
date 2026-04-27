#!/usr/bin/env python3
import torch
import torch.nn as nn


# =========================================================
# MODEL
# =========================================================
class WaveletHybridNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ---------------- RGB ----------------
        self.rgb_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.rgb_bn1   = nn.BatchNorm2d(32)

        self.rgb_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.rgb_bn2   = nn.BatchNorm2d(64)

        self.rgb_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.rgb_bn3   = nn.BatchNorm2d(128)

        self.rgb_refine1 = nn.Conv2d(128, 128, 3, padding=1)
        self.rgb_refine2 = nn.Conv2d(128, 128, 3, padding=1)

        # ---------------- WAVELET ----------------
        self.wav_conv_l1_1 = nn.Conv2d(12, 64, 3, padding=1)
        self.wav_conv_l1_2 = nn.Conv2d(64, 128, 3, padding=1)

        self.wav_conv_l2_1 = nn.Conv2d(12, 64, 3, padding=1)
        self.wav_conv_l2_2 = nn.Conv2d(64, 128, 3, padding=1)

        self.wav_refine1 = nn.Conv2d(128, 128, 3, padding=1)
        self.wav_refine2 = nn.Conv2d(128, 128, 3, padding=1)

        self.wav_proj = nn.Conv2d(128, 128, 1)

        # ---------------- ATTENTION ----------------
        self.level_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        self.final_attn = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # ---------------- CLASSIFIER ----------------
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, x_rgb, wav):
        B = x_rgb.size(0)

        # ---------------- RGB ----------------
        rgb = self.pool(self.relu(self.rgb_bn1(self.rgb_conv1(x_rgb))))
        rgb = self.pool(self.relu(self.rgb_bn2(self.rgb_conv2(rgb))))
        rgb = self.relu(self.rgb_bn3(self.rgb_conv3(rgb)))

        # ---------------- WAVELET LEVEL 1 ----------------
        L1 = torch.cat([wav["LL1"], wav["LH1"], wav["HL1"], wav["HH1"]], dim=1)
        L1 = self.relu(self.wav_conv_l1_1(L1))
        L1 = self.relu(self.wav_conv_l1_2(L1))
        L1 = self.pool(L1)

        # ---------------- WAVELET LEVEL 2 ----------------
        L2 = torch.cat([wav["LL2"], wav["LH2"], wav["HL2"], wav["HH2"]], dim=1)
        L2 = self.relu(self.wav_conv_l2_1(L2))
        L2 = self.relu(self.wav_conv_l2_2(L2))

        # ---------------- LEVEL ATTENTION ----------------
        L_cat = torch.cat([L1, L2], dim=1)  # (B, 256, H/4, W/4)

        w = self.level_attn(L_cat)  # (B, 2)
        w1 = w[:, 0].view(B, 1, 1, 1)
        w2 = w[:, 1].view(B, 1, 1, 1)

        W = w1 * L1 + w2 * L2

        # ---------------- RESIDUAL FUSION ----------------
        W_proj = self.wav_proj(W)
        rgb = rgb + W_proj

        # ---------------- REFINEMENT ----------------
        rgb = self.relu(self.rgb_refine1(rgb))
        rgb = self.relu(self.rgb_refine2(rgb))

        W = self.relu(self.wav_refine1(W))
        W = self.relu(self.wav_refine2(W))

        # ---------------- GLOBAL POOL ----------------
        rgb_vec = self.global_pool(rgb).view(B, -1)
        wav_vec = self.global_pool(W).view(B, -1)

        # ---------------- FINAL ATTENTION ----------------
        f = torch.cat([rgb_vec, wav_vec], dim=1)

        w_final = self.final_attn(f)
        f = w_final[:, 0:1] * rgb_vec + w_final[:, 1:2] * wav_vec

        # ---------------- CLASSIFIER ----------------
        return self.classifier(f)


# =========================================================
# DUMMY DATA GENERATION
# =========================================================
def create_dummy_data(B=2, H=128, W=128):
    x_rgb = torch.randn(B, 3, H, W)

    wav = {
        "LL1": torch.randn(B, 3, H//2, W//2),
        "LH1": torch.randn(B, 3, H//2, W//2),
        "HL1": torch.randn(B, 3, H//2, W//2),
        "HH1": torch.randn(B, 3, H//2, W//2),

        "LL2": torch.randn(B, 3, H//4, W//4),
        "LH2": torch.randn(B, 3, H//4, W//4),
        "HL2": torch.randn(B, 3, H//4, W//4),
        "HH2": torch.randn(B, 3, H//4, W//4),
    }

    return x_rgb, wav


# =========================================================
# FORWARD TEST
# =========================================================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WaveletHybridNet().to(device)
    model.eval()

    x_rgb, wav = create_dummy_data()

    x_rgb = x_rgb.to(device)
    wav = {k: v.to(device) for k, v in wav.items()}

    with torch.no_grad():
        out = model(x_rgb, wav)

    print("\n===== FORWARD TEST =====")
    print("Output shape:", out.shape)
    print("Output logits:", out)
