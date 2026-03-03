import tiktoken
import torch

from practice.ch04.gpt_config import GPT_CONFIG_124M
from practice.ch04.gpt_model import generate_text_simple, GPTModel

tokenizer=tiktoken.get_encoding('gpt2')
def get_tokenizer():
    return tokenizer


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


if __name__ == '__main__':
    start_context = "Every effort moves you"

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    token_ids = generate_text_simple(
        input_model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print('test: ', token_ids_to_text(token_ids, tokenizer))

    # 5.1.2 텍스트 생성 손실 계산하기
    # 임의의 텍스트 토큰을 넣고 모델에서 찾은 후 변환함
    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    # print(probas.shape)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # print("token id:\n", token_ids)
    #
    # print(f"first sample target: {token_ids_to_text(targets[0], tokenizer)}")
    # print(f"output of first sample: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    # 소프트맥스 함수에 의한 평가
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print('text 1: ', target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print('text 2: ', target_probas_2)

    log_probas = torch.log(torch.cat([target_probas_1, target_probas_2]))
    print('log_probas: ', log_probas)

    avg_log_probas = torch.mean(log_probas)
    print('avg_log_probas: ', avg_log_probas)
    neg_avg_log_probas = torch.neg(avg_log_probas)
    print('neg_avg_log_probas: ', neg_avg_log_probas)
    print('logits size: ', logits.shape)
    print('target size: ', targets.shape)

    logits_flat = logits.flatten(0,1)
    targets_flat = targets.flatten()
    print('logits_flat: ', logits_flat.shape)
    print('targets_flat: ', targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print('loss: ', loss)