# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import re

def read_book(file_name):
    with open(file_name, 'r', encoding="utf-8") as f:
        raw_text = f.read()
    # print("total characters: ", len(raw_text))
    # print(raw_text[:99])
    pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    print(pre_processed[:30])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_book('the-verdict.txt')
    # text = 'Hello world. this, is-- a test?'
    # result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    # result = [item for item in result if item.strip()]
    # print(result)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
