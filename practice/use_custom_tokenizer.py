# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import re

from simple_tokenizer import SimpleTokenizerV1, SimpleTokenizerV2


def read_book(file_name):
    with open(file_name, 'r', encoding="utf-8") as f:
        raw_text = f.read()
    # print("total characters: ", len(raw_text))
    # print(raw_text[:99])
    pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    pre_processed = [item.strip() for item in pre_processed if item.strip()]
    # print(pre_processed[:30])
    return pre_processed

def convert_to_token_id(pre_processed):
    all_words = sorted(set(pre_processed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_words)
    #print(vocab_size)
    vocab = {token:integer for integer, token in enumerate(all_words)}
    # for i, item in enumerate(vocab.items()):
    #     print(item)
    #     if i >=50:
    #         break
    return vocab

if __name__ == '__main__':
    text = read_book('../the-verdict.txt')
    # text = 'Hello world. this, is-- a test?'
    # result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    # result = [item for item in result if item.strip()]
    # print(result)

    vocab = convert_to_token_id(text)
    tokenizer = SimpleTokenizerV1(vocab)
    sample = """"It's the last he painted, you know,"
                    Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(sample)
    # print(ids)
    # print(tokenizer.decode(ids))

    # 에러가 터진다. 왜냐하면 저 소설에 Hello가 나오지 않아서.
    test1 = 'Hello, do you like tea?'
    # print(tokenizer.encode(test1))

    test2 = 'In the sunlit terraces of the palace.'
    test = "<|endoftext|> ".join((test1, test2))
    # print(test)

    tokenizer = SimpleTokenizerV2(vocab)
    print(tokenizer.encode(test))

