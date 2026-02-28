 ### 1. 의존성 설치
```bash
pip install -r .\requirement.txt
```

```bash
# cuda version 확인
nvidia-smi
# 12.6 이므로
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```