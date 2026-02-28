import torch
from torch.utils.data import Dataset, DataLoader

from dataset import create_dataloader_v1


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.feature = X
        self.labels = y

    def __getitem__(self, item):
        one_x = self.feature[item]
        one_y = self.labels[item]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

if __name__ == "__main__":
    X_train= torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    y_train = torch.tensor([0,0,0,1,1])
    X_test = torch.tensor([
        [-0.8,2.8],
        [2.6,-1.6]
    ])
    y_test = torch.tensor([0,1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    print(len(train_ds))

    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    for idx, (x, y) in enumerate(train_loader):
        print(f"batch {idx+1}: ", x, y)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    print('')
    for idx, (x, y) in enumerate(train_loader):
        print(f"batch {idx+1}: ", x, y)

    vocab_size=6
    output_dim=3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))
    input_ids = torch.tensor([2,3,5,1])
    print(embedding_layer(input_ids))

    vocab_size=50257
    output_dim=256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    with open("../the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("token id: \n", inputs)
    print("\ninput size: \n", inputs.shape)