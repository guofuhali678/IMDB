"""
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
from scipy import stats #可视化频率前N单词用
class ImdbDataset(Dataset):
    def __init__(self, folder_path, category='train'):
        super().__init__()
        self.category = category
        self.folder_path = Path(folder_path)
        self.load()

    def load(self):
        path = self.folder_path / self.category
        pos_path = path / 'pos'
        neg_path = path / 'neg'
        self.texts = []
        self.labels = []
        try:
            for file_path in tqdm(pos_path.glob('*.txt'), desc='load:pos'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.texts.append(file.read())
                    self.labels.append(1)

            for file_path in tqdm(neg_path.glob('*.txt'), desc='load:neg'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.texts.append(file.read())
                    self.labels.append(0)
        except FileNotFoundError:
            print(f"路径 {pos_path} 或 {neg_path} 不存在，请检查路径。")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        return self.texts[id], self.labels[id]

def main():
    word_set = set()
    # 传入数据集根目录
    imdb_train = ImdbDataset(r'F:\点春季学习\aclImdb', 'train')
    imdb_test = ImdbDataset(r'F:\点春季学习\aclImdb', 'test')
    train_replacement = []
    test_replacement = []

    data_train = DataLoader(imdb_train, batch_size=32, shuffle=True)
    data_test = DataLoader(imdb_test, batch_size=32, shuffle=True)

    # 构建词汇表
    for texts, _ in tqdm(data_train, desc='train'):
        for text in texts:
            word_set.update(re.findall(r'\w+', text.lower()))

    for texts, _ in tqdm(data_test, desc='test'):
        for text in texts:
            word_set.update(re.findall(r'\w+', text.lower()))

    word_dict = {word: num for num, word in enumerate(word_set)}

    try:
        with open('vocab_dict.json', 'w', encoding='utf-8') as output:
            json.dump(word_dict, output, indent=4, ensure_ascii=False)
        print("词汇表已保存到 vocab_dict.json")
    except Exception as e:
        print(f"保存词汇表时出错: {e}")

    # 替换文本为索引
    for texts, _ in tqdm(data_train, desc='replacement:train'):
        for text in texts:
            train_replacement.append([word_dict[word] for word in re.findall(r'\w+', text.lower())])

    for texts, _ in tqdm(data_test, desc='replacement:test'):
        for text in texts:
            test_replacement.append([word_dict[word] for word in re.findall(r'\w+', text.lower())])

    replacement = [train_replacement, test_replacement]

    try:
        with open('replaced_texts.json', 'w', encoding='utf-8') as output:
            json.dump(replacement, output, indent=4, ensure_ascii=False)
        print("替换后的文本已保存到 replaced_texts.json")
    except Exception as e:
        print(f"保存替换后的文本时出错: {e}")
          
if __name__ == '__main__':
    main()

    
    """
#test
"""
import torch
from torch.utils.data import Dataset, DataLoader
class SquareDataset(Dataset):
    def __init__(self, a=0, b=1):
        super().__init__()
        assert a <= b
        self.a = a
        self.b = b

    def __len__(self):
        return self.b - self.a + 1

    def __getitem__(self, index):
        assert self.a <= index <= self.b
        return index, index ** 2

data_train = SquareDataset(a=1, b=64)
data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
print(len(data_train))

# 输出整个数据集
print("整个数据集内容：")
for index in range(data_train.a, data_train.b + 1):
    x, y = data_train[index]
    print(f"样本: x = {x}, x² = {y}")
    """
# 通过Dataset和Dataloader包装数据




#系统制定代码
"""
import torch
from torch.utils.data import Dataset,DataLoader
import chardet
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
import numpy as np
import json


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


class CSVMovieDataset(Dataset):
    def __init__(self,csvfilename,wordlist=None,maxlength=500):
        
        with open(csvfilename,"rb") as f:
            raw_data=f.read()       #二进制形式读取数据      
            detected_encoding=chardet.detect(raw_data)["encoding"]      #输入二进制形式数据并得到编码格式
        
        self.data=pd.read_csv(csvfilename,encoding=detected_encoding)       #以特定编码形式读取数据
        
        self.wordlist=wordlist      #词表
        
        self.maxlength=maxlength        #数据最大长度
        
        if self.wordlist==None:
            self.wordlist=self.buildwordlist()      
            
        self.texts,self.labels=self.processtext()
            
                
    def buildwordlist(self):        #
        wordlist={"UNK":0}      #0号词条表示unknown的单词(即不重要或不常在评论中出现的单词)
        
        stop_words=set(stopwords.words("english"))   
        stemmer=SnowballStemmer("english")
        
        for text in self.data["text"]:
            text=re.sub("[^a-zA-Z]"," ",text)      #将非字母转换为空格
            text=text.lower().strip()       #小写化并去除首尾的空格
            words=word_tokenize(text)       #自动分词
            words=[stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]       #将非停用词词干化
            for word in words:
                if word not in wordlist:
                    wordlist[word]=len(wordlist)
                    
        return wordlist
    

    def processtext(self):      #依据词表将评论以数字矩阵形式替换
        labels=torch.tensor(self.data["polarity"].values,dtype=torch.long)
        
        stop_words=set(stopwords.words("english"))
        stemmer=SnowballStemmer("english")
        
        texts=[]
        
        for text in self.data["text"]:
            text=re.sub("[^a-zA-Z]"," ",text)
            text=text.lower().strip()
            words=word_tokenize(text)
            words=[stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]
            indices=[self.wordlist.get(word,0) for word in words]
            
            if len(indices)>self.maxlength:     #截断数据
                indices=indices[:self.maxlength]
            elif len(indices)<self.maxlength:       #补充无效数据
                indices=indices+[0]*(self.maxlength-len(indices))
            
            texts.append(torch.tensor(indices,dtype=torch.long))
            
        return texts,labels
    
    
    def __len__(self):      
        return len(self.data)       #返回行数
    
    def __getitem__(self,idx):
        return self.texts[idx],self.labels[idx]     #返回特定行数据
    
    
def SaveToJson(dataloader):     #保存为json格式(数据是乱序版)
    data_list=[]
    
    for batch_idx,(texts,labels) in enumerate(dataloader):
        batch_texts=np.array(texts).tolist()
        batch_labels=np.array(labels).tolist()
        for text,label in zip(batch_texts,batch_labels):
            data={"text":text,"label":label}
            data_list.append(data)
            
    with open("data.json","w",encoding="utf-8") as json_file:
        json.dump(data_list,json_file,indent=4)
            
    
def main():
    csvfilename="imdb_tr.csv"
    
    dataset=CSVMovieDataset(csvfilename)
    
    dataloader=DataLoader(dataset,batch_size=128,shuffle=True,num_workers=4)        #128个数据为一次,数据乱序重组
    
    for batch_idx,(texts,labels) in enumerate(dataloader):      #迭代
        print(f"Batch{batch_idx+1}")
        print(f"TextsShape:{texts.shape}")
        print(f"LabelsShape:{labels.shape}")

    SaveToJson(dataloader)

if __name__=="__main__":
    main()
    """


#可视化：
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
class ImdbDataset(Dataset):
    def __init__(self, folder_path, category='train'):
        super().__init__()
        self.category = category
        self.folder_path = Path(folder_path)
        self.load()

    def load(self):
        path = self.folder_path / self.category
        pos_path = path / 'pos'
        neg_path = path / 'neg'
        self.texts = []
        self.labels = []
        try:
            for file_path in tqdm(pos_path.glob('*.txt'), desc='load:pos'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.texts.append(file.read())
                    self.labels.append(1)

            for file_path in tqdm(neg_path.glob('*.txt'), desc='load:neg'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.texts.append(file.read())
                    self.labels.append(0)
        except FileNotFoundError:
            print(f"路径 {pos_path} 或 {neg_path} 不存在，请检查路径。")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        return self.texts[id], self.labels[id]


def main():
    word_set = set()
    imdb_train = ImdbDataset(r'F:\点春季学习\aclImdb', 'train')
    imdb_test = ImdbDataset(r'F:\点春季学习\aclImdb', 'test')
    train_replacement = []
    test_replacement = []

    data_train = DataLoader(imdb_train, batch_size=32, shuffle=True)
    data_test = DataLoader(imdb_test, batch_size=32, shuffle=True)

    # 构建词汇表并统计文本长度
    train_text_lengths = []
    test_text_lengths = []

    # 训练集处理
    for texts, _ in tqdm(data_train, desc='train'):
        for text in texts:
            word_set.update(re.findall(r'\w+', text.lower()))
            train_text_lengths.append(len(re.findall(r'\w+', text.lower())))

    # 测试集处理
    for texts, _ in tqdm(data_test, desc='test'):
        for text in texts:
            word_set.update(re.findall(r'\w+', text.lower()))
            test_text_lengths.append(len(re.findall(r'\w+', text.lower())))

    word_dict = {word: num for num, word in enumerate(word_set)}

    try:
        with open('vocab_dict.json', 'w', encoding='utf-8') as output:
            json.dump(word_dict, output, indent=4, ensure_ascii=False)
        print("词汇表已保存到 vocab_dict.json")
    except Exception as e:
        print(f"保存词汇表时出错: {e}")

    # 替换文本为索引
    for texts, _ in tqdm(data_train, desc='replacement:train'):
        for text in texts:
            train_replacement.append([word_dict[word] for word in re.findall(r'\w+', text.lower())])

    for texts, _ in tqdm(data_test, desc='replacement:test'):
        for text in texts:
            test_replacement.append([word_dict[word] for word in re.findall(r'\w+', text.lower())])

    replacement = [train_replacement, test_replacement]

    try:
        with open('replaced_texts.json', 'w', encoding='utf-8') as output:
            json.dump(replacement, output, indent=4, ensure_ascii=False)
        print("替换后的文本已保存到 replaced_texts.json")
    except Exception as e:
        print(f"保存替换后的文本时出错: {e}")

    # 计算文本长度统计量（均值、中位数、众数）
    train_mean = np.mean(train_text_lengths)
    train_median = np.median(train_text_lengths)
    train_mode = stats.mode(train_text_lengths, keepdims=True).mode[0]

    test_mean = np.mean(test_text_lengths)
    test_median = np.median(test_text_lengths)
    test_mode = stats.mode(test_text_lengths, keepdims=True).mode[0]

    print(f"训练集平均文本长度：{train_mean:.2f}")
    print(f"训练集文本长度中位数：{train_median:.2f}")
    print(f"训练集文本长度众数：{train_mode:.2f}")

    print(f"测试集平均文本长度：{test_mean:.2f}")
    print(f"测试集文本长度中位数：{test_median:.2f}")
    print(f"测试集文本长度众数：{test_mode:.2f}")
#.mean,.median,stats.mode(),分别用来计数均值，中位数，众数。
    # 可视化
    plt.figure(figsize=(12, 6))

    # 子图1：词汇表长度分布
    plt.subplot(1, 2, 1)
    plt.bar(['Vocabulary Size'], [len(word_dict)], color='skyblue')
    plt.ylabel('Count')
    plt.title('Vocabulary Size')
    plt.xticks(rotation=45)

    # 子图2：文本长度分布（单词数）
    plt.subplot(1, 2, 2)
    plt.hist([train_text_lengths, test_text_lengths], bins=20, label=['Train', 'Test'], alpha=0.7)
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('text_analysis.png')
    plt.show()


if __name__ == '__main__':
    main()
    