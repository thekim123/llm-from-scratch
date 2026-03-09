from functools import partial

import torch
from torch.utils.data import DataLoader

from practice.ch07.dataset_download import custom_collate_fn, InstructionDataset, get_datas
from practice.util.token_util import get_tokenizer


def get_dataloaders(train_data, test_data, val_data):
    num_workers = 0
    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )
    train_dataset = InstructionDataset(train_data, tokenizer=get_tokenizer())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer=get_tokenizer())
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer=get_tokenizer())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_data, test_data, val_data = get_datas()
    train_dataset = InstructionDataset(train_data, tokenizer=get_tokenizer())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer=get_tokenizer())
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer=get_tokenizer())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    print("train dataloader: ")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)
