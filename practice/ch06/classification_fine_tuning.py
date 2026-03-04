from matplotlib import pyplot as plt

from practice.ch04.gpt_model import GPTModel, generate_text_simple
from practice.ch05.gpt_download import download_and_load_gpt2
from practice.ch05.load_gpt import load_weights_into_gpt
from practice.ch06.classification_dataloader import get_loader, SpamDataset
from practice.util.token_util import text_to_token_ids, get_tokenizer, token_ids_to_text
import torch
import time


def train_classifier_simple(
        model, train_loader, val_loader,
        optimizer, device, num_epochs,
        eval_freq, eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    example_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            example_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"epoch {epoch + 1} (Step {global_step:06d}): "
                      f"train_loss: {train_loss:.4f}, "
                      f"val_loss: {val_loss:.4f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"train accuracy: {train_accuracy:.4f}, |  ", end="")
        print(f"val accuracy: {val_accuracy:.4f}")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, example_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def plot_values(epochs_seen, example_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(example_seen, train_values, alpha=0)
    ax2.set_xlabel("Example seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions = 0
    num_examples = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predict_labels = torch.argmax(logits, dim=-1)

            num_examples += predict_labels.shape[0]
            correct_predictions += (
                (predict_labels == target_batch).sum().item()
            )

        else:
            break

    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if(len(data_loader) == 0):
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256
):
    model.eval()
    input_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    supported_context_length = model.pos_emb.weight.shape[0]
    if max_length is None:
        max_length = supported_context_length
    max_length = min(max_length, supported_context_length)
    input_ids = input_ids[:max_length]

    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
    predicted_label = torch.argmax(logits, dim=-1).item()
    print("predicted_label:", predicted_label)
    return "spam" if predicted_label == 1 else "not spam"



if __name__ == '__main__':
    INPUT_PROMPT = "Every effort moves"
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
    load_weights_into_gpt(model, params)
    model.eval()

    text1 = "Every effort moves you"
    token_ids = generate_text_simple(
        input_model=model,
        idx=text_to_token_ids(text1, tokenizer=get_tokenizer()),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"],
    )
    print(token_ids_to_text(token_ids, tokenizer=get_tokenizer()))

    text2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        "'You are a winner you have been specially"
        "selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(
        input_model=model,
        idx=text_to_token_ids(text2, tokenizer=get_tokenizer()),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"],
    )
    print(token_ids_to_text(token_ids, tokenizer=get_tokenizer()))

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
    )

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    inputs = get_tokenizer().encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print(inputs)
    print(inputs.shape)

    with torch.no_grad():
        outputs = model(inputs)
    print(outputs.shape)
    print(outputs)

    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    print(label.item())
    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("class label: ", label.item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(123)

    train_loader, val_loader, test_loader = get_loader()
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(train_accuracy, val_accuracy, test_accuracy)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
    print(train_loss, val_loss, test_loss)

    strat_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_loader()
    # eval_freq, eval_iter
    train_losses, val_losses, train_accs, val_accs, example_seen = \
        train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=50,
                                eval_iter=5)

    end_time = time.time()
    execution_time_minutes = (end_time - strat_time) / 60
    print(f"execution time: {execution_time_minutes:.2f} minutes")

    epochs_tensor = torch.linspace(0, example_seen, len(train_losses))
    example_seen_tensor = torch.linspace(0, example_seen, len(train_losses))
    plot_values(epochs_tensor, example_seen_tensor, train_losses, val_losses)

    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
    print(train_accuracy, val_accuracy, test_accuracy)
    torch.save(model.state_dict(), "review_classifier.pth")

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

    txt3 = (
        "Mila, age23, blonde, new in UK. I look sex with UK guys. if u like fun with me. Text MTALK to 69866.18 . 30pp/txt 1st 5free. £1.50 increments. Help08718728876"
    )
    print(classify_review(txt3, model, get_tokenizer(), device, max_length=train_dataset.max_length))


