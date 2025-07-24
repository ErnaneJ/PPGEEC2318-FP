import os
import zipfile
import requests
from pathlib import Path
from io import BytesIO

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification"
DEST_DIR = Path("dataset/intel-image-classification")
ZIP_NAME = "intel-image-classification.zip"

def download_zip(url: str) -> BytesIO:
    print("📥 Baixando o dataset...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Erro ao baixar dataset. Código: {response.status_code}")
    return BytesIO(response.content)

def extract_zip(zip_bytes: BytesIO, dest_path: Path):
    print("📦 Extraindo o conteúdo...")
    with zipfile.ZipFile(zip_bytes) as zip_ref:
        zip_ref.extractall(dest_path)
    print(f"✅ Extração concluída em: {dest_path.resolve()}")

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        zip_bytes = download_zip(DATASET_URL)
        extract_zip(zip_bytes, DEST_DIR)
    except Exception as e:
        print(f"❌ Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
