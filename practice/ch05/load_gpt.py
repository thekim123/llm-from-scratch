import torch
import numpy as np

from practice.ch04.gpt_config import GPT_CONFIG_124M
from practice.ch04.gpt_model import generate, GPTModel
from practice.ch05.gpt2_download import load_gpt2, MODEL_PATH, DEVICE
from practice.util.token_util import text_to_token_ids, get_tokenizer, token_ids_to_text

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}
}


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError("Shape mismatch: {} vs {}".format(left.shape, right.shape))
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params['blocks'])):
        trf_block = gpt.trf_blocks[b]
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        trf_block.att.W_query.weight = assign(
            trf_block.att.W_query.weight, q_w.T)
        trf_block.att.W_key.weight = assign(
            trf_block.att.W_key.weight, k_w.T)
        trf_block.att.W_value.weight = assign(
            trf_block.att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        trf_block.att.W_query.bias = assign(
            trf_block.att.W_query.bias, q_b)
        trf_block.att.W_key.bias = assign(
            trf_block.att.W_key.bias, k_b)
        trf_block.att.W_value.bias = assign(
            trf_block.att.W_value.bias, v_b)

        trf_block.att.out_proj.weight = assign(
            trf_block.att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        trf_block.att.out_proj.bias = assign(
            trf_block.att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        ff_layer = trf_block.ff
        ff_layer.layers[0].weight = assign(
            ff_layer.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        ff_layer.layers[0].bias = assign(
            ff_layer.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        ff_layer.layers[2].weight = assign(
            ff_layer.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        ff_layer.layers[2].bias = assign(
            ff_layer.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        trf_block.norm1.scale = assign(
            trf_block.norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        trf_block.norm1.shift = assign(
            trf_block.norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        trf_block.norm2.scale = assign(
            trf_block.norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        trf_block.norm2.shift = assign(
            trf_block.norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.scale, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


if __name__ == '__main__':
    model_name = 'gpt2-small (124M)'
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({'qkv_bias': True})
    gpt_model = GPTModel(NEW_CONFIG)
    gpt_model.eval()

    settings, params = load_gpt2(MODEL_PATH)
    load_weights_into_gpt(gpt_model, params)

    gpt_model.to(DEVICE)

    torch.manual_seed(123)
    token_ids = generate(
        input_model=gpt_model,
        idx=text_to_token_ids("hello gpt who are you?", tokenizer=get_tokenizer()).to(DEVICE),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=1,
        temperature=0,
    )
    print('output text: ', token_ids_to_text(token_ids, tokenizer=get_tokenizer()))
