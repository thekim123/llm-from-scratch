import tiktoken

# 이게 머하는 거냐면 LLM 훈련을 위한 입력-타깃 쌍을 만드는거임
if __name__ == '__main__':
    with open('the-verdict.txt') as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding('gpt2')
    enc_text = tokenizer.encode(raw_text)
    # 5145
    print(len(enc_text))

    enc_sample = enc_text[50:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    # x: [290, 4920, 2241, 287]
    # y:         [4920, 2241, 287, 257]
    print(f"x: {x}")
    print(f"y:         {y}")


    """
    [290] ----> 4920
    [290, 4920] ----> 2241
    [290, 4920, 2241] ----> 287
    [290, 4920, 2241, 287] ----> 257
    """
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->",  desired)

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

