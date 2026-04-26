import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import random
import jieba
from gensim.models import word2vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 数据预处理
try:
    f1 = open('corpus.txt', encoding='utf-8')
    f2 = open('fenci_result.txt', "w", encoding='utf-8')
    lines = f1.readlines()
    for line in lines:
        line = line.replace('\n', '').strip()
        if not line:
            continue
        set_list = jieba.lcut(line)
        f2.write(' '.join(set_list) + '\n')
    f1.close()
    f2.close()
    print("语料库处理完成")
except FileNotFoundError as e:
    print(e)

# 加载与解析模板文件
f3 = open("templet.txt", encoding="utf-8")
str = ""
for line in f3.readlines():
    str += line
content = json.loads(str)
f3.close()
print("模板文件加载完成")

# 训练word2vec模型
sentence = word2vec.Text8Corpus("fenci_result.txt")
model = word2vec.Word2Vec(sentence)
model.save("fenci_model.model")

# 寻找最大相似度回答
def answer(input, content, model):
    titles = [item['title'] for item in content]
    embeddings = model.encode([input] + titles, convert_to_numpy=True)
    input_embedding = embeddings[0:1]
    title_embeddings = embeddings[1:]
    similarities = cosine_similarity(input_embedding, title_embeddings)
    similarity_index = np.argmax(similarities)
    similarity_max = similarities[similarity_index]

    threshold = 0.7
    if similarity_max < threshold:
        return {"title": "无", "reply": "抱歉，我不太明白您的意思"}

    reply_list = content[similarity_index]['reply']
    reply = random.choice(reply_list)
    return {
        "title": content[similarity_index]['title'],
        "reply": reply
    }

# 运行
def main():
    try:
        print("正在加载预训练模型...")
        # 【关键修复3】使用本地加载 + 强制镜像下载
        model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            trust_remote_code=True
        )
        print("模型加载完成！")
        print("\n--- 聊天机器人已启动 (输入 'quit' 退出) ---")
        while True:
            input_str = input("用户：")
            if input_str.lower() == 'quit':
                print("机器人：再见！")
                break

            result = answer(input_str, content, model)
            print(f"匹配到问题: {result['title']} \n回答： {result['reply']}\n")
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == '__main__':
    main()