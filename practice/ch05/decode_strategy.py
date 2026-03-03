import torch
from matplotlib import pyplot as plt

from practice.ch04.gpt_config import GPT_CONFIG_124M
from practice.ch04.gpt_model import generate, GPTModel
from practice.util.token_util import text_to_token_ids, get_tokenizer, token_ids_to_text


def softmax_with_temperature(logits, temperature):
    scared_logits = logits / temperature
    return torch.softmax(scared_logits, dim=0)

def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1_000)]
    sample_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sample_ids):
        print(f"{freq} x {inverse_vocab[i]}")


if __name__ == "__main__":
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}
    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print(inverse_vocab[next_token_id])

    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])
    print_sampled_tokens(probas)

    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x+i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')
    ax.set_xticks(x)
    ax.set_ylabel('Probability')
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()

    top_k=3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print('top-k logits: ', top_logits)
    print('top-k pos: ', top_pos)

    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float('-inf')),
        other=next_token_logits
    )
    print('new logits: ', new_logits)

    top_k_probas = torch.softmax(new_logits, dim=0)
    print('top-k probas: ', top_k_probas)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cpu")
    model.to(device)
    token_ids= generate(
        input_model=model,
        idx=text_to_token_ids('Every effort moves you', tokenizer=get_tokenizer()),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M['context_length'],
        top_k=25,
        temperature=1.4
    )
    print('output text: \n', token_ids_to_text(token_ids, tokenizer=get_tokenizer()))