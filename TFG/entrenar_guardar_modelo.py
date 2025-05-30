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
import matplotlib.pyplot as plt

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
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # Hasta conv5_x sin avgpool ni fc

    def forward(self, x):
        return self.features(x)


class RDModel(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.encoder = FeatureExtractor()
        for param in self.encoder.parameters():
            param.requires_grad = False  # congelar encoder
        self.encoder.eval()  # fija modo eval para batchnorm y dropout

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, feature_dim, 3, stride=1, padding=1),
        )

    def forward(self, x):
        with torch.no_grad():
            feat = self.encoder(x)
        recon = self.decoder(feat)
        return feat, recon


def normalize_tensor(x):
    mean = x.mean(dim=(2,3), keepdim=True)
    std = x.std(dim=(2,3), keepdim=True) + 1e-6
    return (x - mean) / std


def train_rd(dataset_path, category, epochs=30, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = MVTecDataset(dataset_path, category=category, split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = RDModel().to(device)
    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-4)  # solo decoder
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            feat, recon = model(imgs)

            feat_norm = normalize_tensor(feat)
            recon_norm = normalize_tensor(recon)

            loss = criterion(recon_norm, feat_norm)
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


def load_model_from_db(category, device):
    obj = ModeloGuardado.objects.get(nombre=f"reverse_distillation_{category}")
    state_dict = torch.load(BytesIO(obj.modelo_binario), map_location=device)
    model = RDModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer_and_heatmap(model, pil_img, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat, recon = model(img_tensor)

    # Calcular error por pixel entre feat y recon (en feature space)
    error_map = torch.mean((feat - recon) ** 2, dim=1, keepdim=True)  # [1, 1, H, W]

    # Upsample a tamaño imagen
    error_map = F.interpolate(error_map, size=img_tensor.shape[2:], mode='bilinear', align_corners=False)
    error_map = error_map.squeeze().cpu().numpy()

    # Normalizar para visualizar (0-1)
    error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-6)

    return error_map


def show_image_and_heatmap(img_pil, heatmap):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img_pil)
    plt.title("Imagen original")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_pil)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Heatmap anomalías")
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    dataset_path = r"C:/Users/daniel/Downloads/data/mvtec"
    category = "bottle"

    # Entrenar
    train_rd(dataset_path, category)

    # Cargar y probar inferencia con una imagen de ejemplo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_db(category, device)

    # Cargar imagen ejemplo (debe estar en tu dataset o una imagen nueva)
    example_img_path = os.path.join(dataset_path, category, 'train', 'good', os.listdir(os.path.join(dataset_path, category, 'train', 'good'))[0])
    pil_img = Image.open(example_img_path).convert('RGB')

    heatmap = infer_and_heatmap(model, pil_img, device)
    show_image_and_heatmap(pil_img, heatmap)
