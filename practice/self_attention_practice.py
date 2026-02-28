import time

import torch

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

"""
- 셀프 어텐션에서는 입력시퀀스에 있는 각 원소 xi에 대한 문맥 벡터 zi를 계산하는 것이 목표입니다.
- 문맥 벡터는 정보가 풍부한 임베딩 벡터로 생각할 수 있습니다.

- 문맥 벡터의 목적은 입력 시퀀스에 있는 다른 모든 원소의 정보를 통합해 이 시퀀스에 있는 각 원소의 표현을 풍부하게 만드는 것입니다.
- 이것이 llm의 핵심이며, 문장에서 다른 단어 사이의 관계와 관련성을 이해하기 위해 LLM에 반드시 필요합니다.
- 나중에 LLM이 이런 문맥 벡터를 학습할 수 있도록 훈련 가능한 가중치를 추가하겠습니다.
"""
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # Your (x^2)
    [0.57, 0.85, 0.64], # Your (x^3)
    [0.22, 0.58, 0.33], # Your (x^4)
    [0.77, 0.25, 0.10], # Your (x^5)
    [0.05, 0.80, 0.55], # Your (x^6)
])

"""
셀프 어텐션을 구하는 첫 단계는 어텐션 점수라고 부르는 중간값 w를 계산하는 것입니다.
"""
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

"""
다음 단계에서는 어텐션 점수를 정규화합니다. 
"""
attn_weight2_tmp = attn_scores_2 / attn_scores_2.sum()
print("attn_weight: ",attn_weight2_tmp)

"""
소프트맥스 함수를 사용하여 정규화하는 것이 더 일반적이고 권장된다.
이 방법이 극한 값을 더 잘 다루며 훈련과정에 유용한 gradient 속성을 가지고 있다.
어텐션 가중치가 항상 양수가 되도록 보장한다.
이렇게 하면 출력을 확률이나 상대적인 중요도로 해석할 수 있고 가중치가 높을수록 중요도가 높다.
"""
# 실전에서는 토치 라이브러리씀
attn_weights2_naive = softmax_naive(attn_scores_2)
print("attn_weights2_naive: ",attn_weights2_naive)
print("sum: ", attn_weights2_naive.sum())

attn_weight2 = torch.softmax(attn_scores_2, dim=0)
print("attn_weight2: ",attn_weight2)

query = inputs[1]
content_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    content_vec_2 += attn_weight2[i] * x_i
print('con_vec2: ',content_vec_2)

start = time.time()
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
end = time.time()
print(f"걸린 시간: {end - start:.96f}초")

start = time.time()
attn_scores = inputs @ inputs.T
print(attn_scores)
end = time.time()
print(f"걸린 시간: {end - start:.96f}초")

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print(row_2_sum)

all_context_Vecs = attn_weights @ inputs
print(all_context_Vecs)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

print('')
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)

query2 = x_2 @ W_query
key2 = x_2 @ W_key
value2 = x_2 @ W_value
print(query2)

keys = inputs @ W_key
values = inputs @ W_value
print(f"keys.shape: ", keys.shape)
print(f"values.shape: ", values.shape)

keys_2 = keys[1]
attn_score_22 = query2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query2 @ keys.T
print(attn_scores_2)

print()
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim=-1)
print(attn_weights_2)

print()
content_vec_22 = attn_weights_2 @ values
print(content_vec_22)
