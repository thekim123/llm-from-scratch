import urllib.request
import os
import json

import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response: \n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, item):
        return self.encoded_texts[item]

    def __len__(self):
        return len(self.data)


def custom_collate_draft_1(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    input_list = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
                new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        input_list.append(inputs)
        return input_list


def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    input_list, target_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        needed_pad_token = [pad_token_id] * (batch_max_length - len(new_item))
        padded = (
                new_item + needed_pad_token
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        input_list.append(inputs)
        target_list.append(targets)

    input_tensor = torch.stack(input_list).to(device)
    target_tensor = torch.stack(target_list).to(device)
    return input_tensor, target_tensor


def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    input_list, target_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
                new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        input_list.append(inputs)
        target_list.append(targets)

    input_tensor = torch.stack(input_list).to(device)
    target_tensor = torch.stack(target_list).to(device)
    return input_tensor, target_tensor

    return None


def download_and_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request. "
        f"\n\n### Instructions: \n {entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    )
    return instruction_text + input_text


def get_datas():
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rickiepark/llm-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_file(file_path, url)
    print(f'sample count: {len(data)}')

    # 데이터셋 포멧 출력 확인 코드
    # model_input = format_input(data[999])
    # desired_response = f"\n\n### Response: \n{data[999]['output']}"
    # print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    return train_data, test_data, val_data


if __name__ == "__main__":
    file_path = "gpt_download.py"
    # url = (
    #     "https://raw.githubusercontent.com/rickiepark/llm-from-scratch"
    #     "/main/ch07/01_main-chapter-code/instruction-data.json"
    # )
    url = (
        "https://raw.githubusercontent.com/rickiepark/llm-from-scratch"
        "/main/ch05/01_main-chapter-code/gpt_download.py"
    )

    data = download_and_file(file_path, url)
    print(f'sample count: {len(data)}')

    # 데이터셋 포멧 출력 확인 코드
    # model_input = format_input(data[999])
    # desired_response = f"\n\n### Response: \n{data[999]['output']}"
    # print(model_input + desired_response)
    #
    # train_portion = int(len(data) * 0.85)
    # test_portion = int(len(data) * 0.1)
    # val_portion = len(data) - train_portion - test_portion
    #
    # train_data = data[:train_portion]
    # test_data = data[train_portion:train_portion + test_portion]
    # val_data = data[train_portion + test_portion:]
    # # print(f'train count: {len(train_data)}')
    # # print(f'test count: {len(test_data)}')
    # # print(f'val count: {len(val_data)}')
    #
    # input_1 = [0, 1, 2, 3, 4]
    # input_2 = [5, 6]
    # input_3 = [7, 8, 9]
    # batch = (input_1, input_2, input_3)
    # inputs, targets = custom_collate_fn(batch)
    # # print(inputs)
    # # print(targets)
    #
    # logits_1 = torch.tensor(
    #     [[-1.0, 1.0],
    #      [-0.5, 1.5]]
    # )
    # target_1 = torch.tensor([0, 1])
    # loss_1 = torch.nn.functional.cross_entropy(logits_1, target_1)
    # print(loss_1)
    #
    # logits_2 = torch.tensor(
    #     [[-1.0, 1.0],
    #      [-0.5, 1.5],
    #      [-0.5, 1.5]]
    # )
    # target_2 = torch.tensor([0, 1, 1])
    # loss_2 = torch.nn.functional.cross_entropy(logits_2, target_2)
    # print(loss_2)
    #
    # target_3 = torch.tensor([0, 1, -100])
    # loss_3 = torch.nn.functional.cross_entropy(logits_2, target_3)
    # print(loss_3)
    # print("loss1 == loss3: ", loss_1 == loss_3)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
