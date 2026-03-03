import tiktoken
import torch

from torch import nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def forward(x):
    return x


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class DummyLayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


def get_dummy_batch():
    tik_tokenizer = tiktoken.get_encoding('gpt2')
    result_batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    result_batch.append(torch.tensor(tik_tokenizer.encode(txt1)))
    result_batch.append(torch.tensor(tik_tokenizer.encode(txt2)))
    result_batch = torch.stack(result_batch, dim=0)
    return result_batch


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding('gpt2')
    batch = get_dummy_batch()

    # 4.2 층 정규화로 활성화 정규화하기
    # ReLU 정규화
    torch.manual_seed(123)
    batch_example = torch.rand(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print("layer_out: ", out)

    # 평균과 분산
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("mean: ", mean)
    print("var: ", var)

    # 얻은 층 출력에 층 정규화 적용
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("normed total mean: ", out_norm)
    print("mean: ", mean)
    print("var: ", var)

    torch.set_printoptions(sci_mode=False)
    print("no sci mean: ", mean)
    print("no sci var: ", var)

    # LayerNorm 클래스 사용 예시
    batch_example = torch.rand(2, 5)
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("mean: ", mean)
    print("var: ", var)
