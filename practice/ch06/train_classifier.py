import torch

from gpt_download import download_and_load_gpt2
from practice.ch04.gpt_model import GPTModel
from practice.ch06.classification_dataloader import SpamDataset
from practice.ch06.classification_fine_tuning import classify_review
from practice.util.token_util import get_tokenizer

if __name__ == '__main__':
    model_state_dict = torch.load("review_classifier.pth")
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}
    }
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir='gpt2')
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(model_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = SpamDataset(
        csv_file="train.csv",
        tokenizer=get_tokenizer(),
    )
    txt1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award"
    )
    print(classify_review(txt1, model, get_tokenizer(), device, max_length=train_dataset.max_length))

    txt2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(txt2, model, get_tokenizer(), device, max_length=train_dataset.max_length))
