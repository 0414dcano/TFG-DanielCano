import os
import django
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from io import BytesIO
import torch.nn.functional as F

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TFG.settings")
django.setup()

from detector.models import ModeloGuardado


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category='bottle', split='train', transform=None):
        self.transform = transform
        self.img_paths = []
        base_path = os.path.join(root_dir, category, split)
        if split == 'train':
            normal_dir = os.path.join(base_path, 'good')
            self.img_paths = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.png')]
        else:
            raise NotImplementedError("Solo train implementado")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        return self.features(x)


class RDModel(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.encoder = FeatureExtractor()
        # Decoder simple: 3 conv transpose layers to reconstruct features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, feature_dim, 3, stride=1, padding=1),
        )

    def forward(self, x):
        feat = self.encoder(x)  # B x C x H x W, por ejemplo [B, 512, 8, 8]
        recon = self.decoder(feat)  # Actualmente [B, 512, 32, 32]

        # Redimensionar recon a la forma de feat
        recon = F.interpolate(recon, size=feat.shape[2:], mode='bilinear', align_corners=False)
        return feat, recon


def train_rd(dataset_path, category, epochs=20, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = MVTecDataset(dataset_path, category=category, split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = RDModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            feat, recon = model(imgs)
            loss = criterion(recon, feat.detach())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.6f}")

    # Guardar modelo en DB
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    binario = buffer.getvalue()

    obj, created = ModeloGuardado.objects.update_or_create(
        nombre=f"reverse_distillation_{category}",
        defaults={'modelo_binario': binario}
    )

    print(f"Modelo Reverse Distillation guardado en DB. Created: {created}")


if __name__ == "__main__":
    dataset_path = "C:/Users/daniel/Downloads/TFG DanielCano/TFG DanielCano/TFG/data/mvtec"
    category = "bottle"
    train_rd(dataset_path, category)
