import torch

from dataset import create_dataloader_v1
from practice.ch04.gpt_config import GPT_CONFIG_124M
from practice.ch04.gpt_model import generate_text_simple, GPTModel, generate
from practice.ch05.calculate_loss import calc_loss_batch, calc_loss_loader, get_dummy_train_data
from practice.util.graph_util import plot_losses
from practice.util.token_util import text_to_token_ids, token_ids_to_text, get_tokenizer


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter,
                       start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(
                    f"epoch {epoch + 1} (Step {global_step:06d}): "
                    f"train_loss: {train_loss: .3f}, "
                    f"val_loss: {val_loss: .3f}, "
                )

                generate_and_print_sample(
                    model, tokenizer, device, start_context,
                )
    return train_losses, val_losses, track_token_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        model.train()
        return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            input_model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace('\n', ' '))
        model.train()


if __name__ == '__main__':
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 20
    tokenizer = get_tokenizer()

    train_data, test_data = get_dummy_train_data()
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=64,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        test_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=64,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    prompt = 'if she had not dragged him down,'
    train_losses, val_losses, token_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=prompt, tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, token_seen, train_losses, val_losses)

    model.to("cpu")
    model.eval()
    tokenizer = get_tokenizer()
    token_ids = generate(
        input_model=model,
        idx=text_to_token_ids(prompt, tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=None,
        temperature=0.0
    )
    print("output text: \n", token_ids_to_text(token_ids, tokenizer))

    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
        "model_and_optimizer.pth"
    )

    checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
