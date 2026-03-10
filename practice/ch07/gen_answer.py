from tqdm.auto import tqdm

from practice.ch04.gpt_model import generate
from practice.ch07.dataset_download import get_datas, format_input

import torch
import json

from practice.ch07.load_pretrained_model import load_model, BASE_CONFIG
from practice.util.token_util import text_to_token_ids, get_tokenizer, token_ids_to_text

if __name__ == "__main__":
    train_data, test_data, val_data = get_datas()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    model.load_state_dict(torch.load('gpt2-medium355M-sft.pth'))
    model.to(device)
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
            input_model=model,
            idx=text_to_token_ids(input_text, tokenizer=get_tokenizer()).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer=get_tokenizer())
        response_text = (
            generated_text[len(input_text):]
            .replace(" ", "")
            .strip()
        )
        test_data[i]["model_response"] = response_text

    with open("instruction-data-with-response.json", "w") as f:
        json.dump(test_data, f, indent=4)

    print(test_data[0])
