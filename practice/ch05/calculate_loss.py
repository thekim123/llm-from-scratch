from dataset import create_dataloader_v1
from practice.ch04.gpt_config import GPT_CONFIG_124M
from practice.ch04.gpt_model import GPTModel
from practice.util.token_util import get_tokenizer
import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def get_dummy_train_data():
    file_path = "../../the-verdict.txt"
    with open(file_path) as f:
        lines = f.read()
    total_characters = len(lines)
    tokenizer = get_tokenizer()
    total_token = len(tokenizer.encode(lines))
    print('total characters:', total_characters)
    print('total token:', total_token)
    print()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(lines))
    train_data, test_data = lines[:split_idx], lines[split_idx:]
    return train_data, test_data

if __name__ == '__main__':
    train_data, test_data = get_dummy_train_data()

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        test_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print('훈련 데이터 로더: ')
    for x, y in train_loader:
        print(x.shape, y.shape)

    print('\ntest data loader: ')
    for x, y in val_loader:
        print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print('train_loss:', train_loss)
    print('val_loss:', val_loss)
