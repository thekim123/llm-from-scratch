import json
import os
from pathlib import Path

import torch
import tensorflow as tf

from practice.ch05.gpt_download import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt

MODEL_PATH = '../../gpt2/124M'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_gpt2(model_dir):
    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    ROOT = Path(__file__).resolve().parents[2]  # 필요하면 parents[1]로 루트 맞추기
    path = ROOT / "gpt2" / "124M" / "hparams.json"
    setting = json.loads(path.read_text(encoding="utf-8"))
    para = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, setting)
    return setting, para


if __name__ == '__main__':
    url = (
        "https://raw.githubusercontent.com/rickiepark/"
        "llm-from-scratch/main/ch05/"
        "01_main-chapter-code/gpt_download.py"
    )

    filename = url.split("/")[-1]
    # urllib.request.urlretrieve(url, filename)
    #
    # settings, params = download_and_load_gpt2(
    #     model_size="124M",
    #     models_dir="gpt2",
    # )

    file_path = '../../gpt2/124M'
    settings, params = load_gpt2(file_path)
    print('settings: ', settings)
    print('params: ', params.keys())
