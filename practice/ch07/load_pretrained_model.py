import re

import torch
import time
from practice.ch04.gpt_model import GPTModel, generate
from practice.ch05.calculate_loss import calc_loss_loader

from practice.ch05.load_gpt import load_weights_into_gpt
from practice.ch05.pretrain_simple import train_model_simple
from practice.ch07.dataloader import get_dataloaders
from practice.ch07.dataset_download import format_input, get_datas
from practice.ch07.gpt_download import download_and_load_gpt2
from practice.util.graph_util import plot_losses
from practice.util.token_util import text_to_token_ids, get_tokenizer, token_ids_to_text

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def load_model():
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    return model


if __name__ == '__main__':
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    torch.manual_seed(123)
    train_data, test_data, val_data = get_datas()
    input_text = format_input(val_data[0])
    print(input_text)

    token_ids = generate(
        input_model=model,
        idx=text_to_token_ids(input_text, tokenizer=get_tokenizer()),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer=get_tokenizer())
    response_text = generated_text[len(input_text):].strip()
    print(response_text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(123)
    train_loader, val_loader, test_loader = get_dataloaders(train_data, test_data, val_data)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(train_loss, val_loss)

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 2

    train_losses, val_losses, token_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, num_epochs=num_epochs,
        device=device, eval_freq=5, eval_iter=5, start_context=format_input(val_data[0]),
        tokenizer=get_tokenizer()
    )
    end_time = time.time()
    execution_time_min = (end_time - start_time) / 60
    print(f"execution time: {execution_time_min:.2f}min")

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    torch.save(model.state_dict(), file_name)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, token_seen, train_losses, val_losses)
