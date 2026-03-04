import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import _T_co

from practice.util.token_util import get_tokenizer


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index) -> _T_co:
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def get_loader():
    tokenizer = get_tokenizer()
    # endoftext =50256
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    train_dataset = SpamDataset(
        csv_file="train.csv",
        tokenizer=tokenizer,
    )
    # print(train_dataset.max_length)

    validation_dataset = SpamDataset(
        csv_file="validation.csv",
        tokenizer=tokenizer,
    )
    test_dataset = SpamDataset(
        csv_file="test.csv",
        tokenizer=tokenizer,
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers,  drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    return train_loader, validation_loader, test_loader

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    # endoftext =50256
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    train_dataset = SpamDataset(
        csv_file="train.csv",
        tokenizer=tokenizer,
    )
    print(train_dataset.max_length)

    validation_dataset = SpamDataset(
        csv_file="validation.csv",
        tokenizer=tokenizer,
    )
    test_dataset = SpamDataset(
        csv_file="test.csv",
        tokenizer=tokenizer,
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers,  drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

    # for input_batch, target_batch in train_loader:
    #     print("입력 배치 차원: ", input_batch.shape)
    #     print("레이블 배치 차원: ", target_batch.shape)
    print(f"{len(train_loader)}개 훈련 배치")
    print(f"{len(validation_loader)}개 검증 배치")
    print(f"{len(test_loader)}개 테슽 배치")
