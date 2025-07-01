
import pandas as pd
import requests
import zipfile
import io
import os
def download_and_extract_imdb_dataset():
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    response = requests.get(url)
    if response.status_code == 200:
        # 确保存储目录存在
        if not os.path.exists('aclImdb'):
            os.makedirs('aclImdb')
        # 保存压缩文件
        with open('aclImdb_v1.tar.gz', 'wb') as f:
            f.write(response.content)
        # 解压文件
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall()
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

def LoadIBMData():
    if not os.path.exists('aclImdb'):
        download_and_extract_imdb_dataset()
    data = []
    for label in ['pos', 'neg']:
        path = f'aclImdb/train/{label}'
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                data.append({'text': text, 'label': 1 if label == 'pos' else 0})
    return pd.DataFrame(data)

def ProcessData(data):
    data = data["text"]
    data = data.str.lower()
    data = data.str.replace(r"[^a-z0-9']", " ", regex=True)
    data = data.str.replace(r"'[a-z0-9]*", " ", regex=True)
    data = data.str.replace(r"[' ']+", " ", regex=True)
    data = data.str.split(" ")
    return data


def CreateWordList(data):
    word_list = {}
    line = data.shape[0]
    for i in range(line):
        for word in data[i]:
            if word not in word_list:
                word_list[word] = 1
            elif word in word_list:
                word_list[word] += 1
    if '' in word_list:
        del word_list['']
    return word_list

def SortedWordList(word_list):
    sorted_word_list = {}
    sorted_word_count = sorted(word_list.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_word_count:
        sorted_word_list[word] = count
    return sorted_word_list


def SelectedWordList(sorted_word_list, k):
    selected_word_list = {}
    for key, value in sorted_word_list.items():
        if value >= k:
            selected_word_list[key] = value
    return selected_word_list


def FinalWordList(selected_word_list):
    final_word_list = {}
    num = len(selected_word_list.keys())
    for i in range(num):
        final_word_list[int(i)] = list(selected_word_list.keys())[i]
    return final_word_list

def TextToSequence(data, final_word_list):
    vocab = {word: int(idx) for idx, word in final_word_list.items()}
    index = data.shape[0]
    sequences = []
    for i in range(index):
        sequence = []
        for word in list(data[i]):
            if word in final_word_list.values():
                sequence.append(vocab[word])
            else:
                sequence.append(-1)
        sequences.append(sequence)
    return sequences
def save_results(sorted_word_list, selected_word_list, final_word_list, sequences):
    with open("SortedWordList.txt", "w", encoding="utf-8") as f:
        for word, count in sorted_word_list.items():
            f.write(f"{word}: {count}\n")

    with open("SelectedWordList.txt", "w", encoding="utf-8") as f:
        for word, count in selected_word_list.items():
            f.write(f"{word}: {count}\n")

    with open("FinalWordList.txt", "w", encoding="utf-8") as f:
        for idx, word in final_word_list.items():
            f.write(f"{idx}: {word}\n")

    with open("ProcessedIBM.txt", "w", encoding="utf-8") as f:
        for sequence in sequences:
            f.write(" ".join(map(str, sequence)) + "\n")


data = LoadIBMData()
data = ProcessData(data)
word_list = CreateWordList(data)
sorted_word_list = SortedWordList(word_list)
selected_word_list = SelectedWordList(sorted_word_list, 100)
final_word_list = FinalWordList(selected_word_list)
sequences = TextToSequence(data, final_word_list)

save_results(sorted_word_list, selected_word_list, final_word_list, sequences)

print("排序后的词表:")
for i, (word, count) in enumerate(sorted_word_list.items()):
    if i < 10:
        print(f"{word}: {count}")
    else:
        break

print("\n筛选后的词表:")
for i, (word, count) in enumerate(selected_word_list.items()):
    if i < 10:
        print(f"{word}: {count}")
    else:
        break

print("\n最终编号词表:")
for i, (idx, word) in enumerate(final_word_list.items()):
    if i < 10:
        print(f"{idx}: {word}")
    else:
        break
    