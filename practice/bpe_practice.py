from importlib.metadata import version
import tiktoken

# 0.12.0
print("tiktoken version: ", version('tiktoken'))

tokenizer = tiktoken.get_encoding('gpt2')
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)
integers = tokenizer.encode(
    text,
    allowed_special={"<|endoftext|>"},
)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
print(integers)

strings = tokenizer.decode(integers)
# Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.
print(strings)
