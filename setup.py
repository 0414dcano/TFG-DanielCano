import subprocess
import sys
import shutil
import os

def has_nvidia_gpu():
    return shutil.which("nvidia-smi") is not None

def install_torch():
    index_url = None
    if has_nvidia_gpu():
        print("✅ NVIDIA GPU detectada. Instalando PyTorch con soporte CUDA...")
        torch_version = "torch==2.7.0"
        torchvision_version = "torchvision==0.22.0"
        
    else:
        print("⚠️ No se detectó GPU NVIDIA. Instalando PyTorch para CPU...")
        torch_version = "torch==2.7.0+cpu"
        torchvision_version = "torchvision==0.22.0+cpu"
        index_url = "https://download.pytorch.org/whl/cpu"
    
    if index_url is None:
        subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        torch_version, torchvision_version,
        ])
       
    else:
        subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        torch_version, torchvision_version,
        "--index-url", index_url
    	])

def install_requirements():
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ No se encontró {requirements_file}")
        sys.exit(1)

    print(f"📦 Instalando dependencias desde {requirements_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])

if __name__ == "__main__":
    install_torch()
    install_requirements()
    print("✅ Instalación completa.")

