### 1. venv 시작

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
deactivate
```


```bash
# cuda version 확인
nvidia-smi
# 12.6 이므로
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```