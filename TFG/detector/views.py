from django.shortcuts import render, redirect
from .models import Conversacion, Mensaje, ModeloGuardado
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.http import HttpResponse
from django.template.loader import render_to_string

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import cv2
import traceback

# Transforms para MVTEC AD (imagen tamaño 256x256 y normalización imagen ImageNet)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet mean
                         std=[0.229, 0.224, 0.225])   # Imagenet std
])

# Backbone para extracción de features (ResNet18)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # hasta penúltima capa conv
        
    def forward(self, x):
        return self.features(x)

def inferir_imagen(imagen_django_file):
    try:
        # Obtener último modelo guardado
        modelo_guardado = ModeloGuardado.objects.order_by("-fecha_guardado").first()
        if not modelo_guardado:
            return "Modelo no disponible.", None

        # Cargar modelo Reverse Distillation guardado (solo state_dict)
        buffer = BytesIO(modelo_guardado.modelo_binario)
        state_dict = torch.load(buffer, map_location='cpu')

        # Definir modelo RD (igual que en entrenamiento)
        class FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                backbone = models.resnet18(pretrained=False)
                self.features = nn.Sequential(*list(backbone.children())[:-2])
            def forward(self, x):
                return self.features(x)

        class RDModel(nn.Module):
            def __init__(self, feature_dim=512):
                super().__init__() 
                self.encoder = FeatureExtractor()
                # Decoder ajustado para que mantenga las dimensiones espaciales (sin upsampling)
                self.decoder = nn.Sequential(
                nn.ConvTranspose2d(feature_dim, 256, kernel_size=3, stride=1, padding=1),  # mantiene HxW
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, feature_dim, kernel_size=3, stride=1, padding=1),
            )

            def forward(self, x):
                feat = self.encoder(x)  # B x C x H x W (normalmente [B,512,8,8])
                recon = self.decoder(feat)  # Reconstrucción del mismo tamaño
                return feat, recon


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RDModel().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        imagen = Image.open(imagen_django_file).convert("RGB")
        entrada = transform(imagen).unsqueeze(0).to(device)

        with torch.no_grad():
            feat, recon = model(entrada)  # [B, C, H, W]
            # Calculamos diferencia por pixel (MSE)
            diff = (feat - recon).pow(2).mean(dim=1).squeeze(0)  # [H, W]

            # Normalizamos a [0, 1]
            heatmap = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

            threshold = 0.5
            pred_mask = (heatmap > threshold).float()

        # Convertir máscara a numpy uint8 para contornos
        pred_mask_np = (pred_mask.cpu().numpy() * 255).astype(np.uint8)

        # Encontrar contornos con OpenCV
        contours, _ = cv2.findContours(pred_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Redimensionar imagen original a 256x256 para escalar bbox
        imagen_np = np.array(imagen.resize((256, 256)))

        scale_x = imagen_np.shape[1] / heatmap.shape[1]
        scale_y = imagen_np.shape[0] / heatmap.shape[0]

        # Dibujar bounding boxes rojas sobre la imagen
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1 = int(x * scale_x), int(y * scale_y)
            x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
            cv2.rectangle(imagen_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

        img_bbox = Image.fromarray(imagen_np)
        buffer = BytesIO()
        img_bbox.save(buffer, format="PNG")
        base64_img_bbox = base64.b64encode(buffer.getvalue()).decode()
        bbox_img = f"data:image/png;base64,{base64_img_bbox}"

        return "Anomalía detectada", bbox_img

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error completo en inferencia:\n", traceback_str)
        return f"Error en inferencia: {e}", None


@login_required
def chat_view(request):
    user = request.user

    conversacion = Conversacion.objects.filter(usuario=user).order_by('-fecha').first()
    if not conversacion:
        conversacion = Conversacion.objects.create(usuario=user)

    respuesta = None
    bbox_img = None
    uploaded_file_url = None

    if request.method == 'POST' and request.FILES.get('imagen'):
        imagen = request.FILES['imagen']

        mensaje = Mensaje.objects.create(conversacion=conversacion, imagen=imagen, respuesta='')

        respuesta, bbox_img = inferir_imagen(mensaje.imagen)

        mensaje.respuesta = respuesta
        mensaje.save()

        uploaded_file_url = mensaje.imagen.url

    mensajes = conversacion.mensajes.order_by('fecha')

    # Si la petición es AJAX, devuelve sólo la parte del chat (fragmento HTML)
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        
        # Renderiza la plantilla completa
        html_completo = render_to_string('detector/chat.html', {
            'conversacion': conversacion,
            'mensajes': mensajes,
            'respuesta': respuesta,
            'uploaded_file_url': uploaded_file_url,
            'bbox_img': bbox_img,
        }, request=request)
        # Aquí devuelve la plantilla completa y en JS extraemos la parte que nos interesa
        return HttpResponse(html_completo)


    # Si no es AJAX, renderiza la página completa
    return render(request, 'detector/chat.html', {
        'conversacion': conversacion,
        'mensajes': mensajes,
        'respuesta': respuesta,
        'uploaded_file_url': uploaded_file_url,
        'bbox_img': bbox_img,
    })
