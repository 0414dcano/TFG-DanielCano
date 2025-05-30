import subprocess
import sys
import shutil
import os


def has_nvidia_gpu():
    return shutil.which("nvidia-smi") is not None


def install_torch():
    if has_nvidia_gpu():
        print("✅ NVIDIA GPU detectada. Instalando PyTorch con soporte CUDA...")
        torch_version = "torch==2.2.2"
        torchvision_version = "torchvision==0.17.2"
        index_url = "https://download.pytorch.org/whl/cu118"
    else:
        print("⚠️ No se detectó GPU NVIDIA. Instalando PyTorch para CPU...")
        torch_version = "torch==2.2.2"
        torchvision_version = "torchvision==0.17.2"
        index_url = "https://download.pytorch.org/whl/cpu"

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        torch_version, torchvision_version,
        "--index-url", index_url
    ])


def install_requirements():
    requirements_file = "./TFG/requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ No se encontró {requirements_file}")
        sys.exit(1)

    print(f"📦 Instalando dependencias desde {requirements_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])


def ajuste_BDD():
    subprocess.check_call([sys.executable, "./TFG/manage.py", "makemigrations"])
    subprocess.check_call([sys.executable, "./TFG/manage.py", "migrate"])


def creacion_modelo():
    subprocess.check_call([sys.executable, "./TFG/entrenar_guardar_modelo.py"])


if __name__ == "__main__":
    install_torch()
    install_requirements()
    print("✅ Instalación completa. Ahora se va a proceder a ajustar la BBDD")
    ajuste_BDD()
    print("✅ BBDD ajustada correctamente. Se va a proceder a crear el modelo necesario")
    creacion_modelo()
    print("✅ Modelo Creado. Iniciando servidor local")
    subprocess.check_call([sys.executable, "./TFG/manage.py", "runserver"])
