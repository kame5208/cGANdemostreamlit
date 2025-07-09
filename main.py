import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator モデル定義
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# モデルのロード（weights_only=False）
latent_dim = 10
n_classes = 10
generator = torch.load("GANgenerator.pth", map_location=device, weights_only=False)
generator.eval()

# Streamlit UI
st.title("Conditional GAN MNIST Generator")
st.write("任意の数字 (0-9) を選んで画像を生成します。")

target_label = st.number_input("生成したい数字 (0-9)", min_value=0, max_value=9, step=1)
n_rows = st.slider("生成する画像の枚数", 1, 10, 5)

if st.button("画像を生成"):
    label_tensor = torch.tensor([target_label], dtype=torch.long, device=device)
    gen_labels = label_tensor.repeat(n_rows)
    z = torch.randn(n_rows, latent_dim, device=device)

    with torch.no_grad():
        gen_imgs = generator(z, gen_labels)
        gen_imgs = (gen_imgs + 1) / 2  # [-1,1] → [0,1]

    # 表示用画像を作成
    fig, axs = plt.subplots(1, n_rows, figsize=(2 * n_rows, 2))
    if n_rows == 1:
        axs = [axs]
    for i in range(n_rows):
        img = gen_imgs[i].cpu().squeeze().numpy()
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Label {target_label}")
    st.pyplot(fig)
