from cgitb import reset
from pathlib import Path
import pandas as pd
import os
import urllib.request
import zipfile


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print('data_file_path already exist: ', data_file_path)
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, 'wb') as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
        original_file_path = Path(extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, data_file_path)
        print('file download is complete. path: ', data_file_path)


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced_df["Label"] = balanced_df["Label"].map({"spam": 1, "ham": 0})
    return balanced_df


def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


if __name__ == '__main__':
    req_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    req_zip_path = "sms_spam_collection.zip"
    req_extracted_path = "sms_spam_collection"
    req_data_file_path = Path(req_extracted_path) / "SMSSpamCollection.tsv"
    # download_and_unzip_spam_data(req_url, req_zip_path, req_extracted_path, req_data_file_path)

    df = pd.read_csv(req_data_file_path, sep='\t', header=None, names=["Label", "Text"])
    # print(df)
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    train_df, validation_df, test_df = random_split(balanced_df, train_frac=0.7, validation_frac=0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
