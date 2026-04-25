import json
import random
import math
from os import replace
import jieba
from gensim.models import word2vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#数据预处理
try:
    f1 = open('corpus.txt', encoding='utf-8')
    f2 = open('fenci_result.txt',"w", encoding='utf-8')
    lines = f1.readlines()
    for line in lines:
        line = line.replace('\n', '').replace('','')
        if not line:continue
        set_list = jieba.lcut(line)
        f2.write(' '.join(set_list) + '\n')
    f1.close()
    f2.close()
    print("语料库处理完成")
except FileNotFoundError as e:
    print(e)

#加载与解析模板文件
f3 = open("templet.txt",encoding="utf-8")
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

#寻找最大相似度回答
def answer(input, content, model):
    """
    使用 BERT 模型寻找与用户输入最相似的回答
    :param input: 用户输入的字符串
    :param content: 包含问题和回答的列表
    :param model: 加载好的 SentenceTransformer 模型
    :return: 包含 title 和 reply 的字典
    """
    # 1. 准备所有需要比较的文本（知识库中的所有问题）
    titles = [item['title'] for item in content]

    # 2. 使用模型将文本转换为向量
    # model.encode 可以一次处理一个列表，效率很高
    embeddings = model.encode([input] + titles, convert_to_numpy=True)

    # 第一个向量是用户输入的向量
    input_embedding = embeddings[0:1]
    # 剩下的向量是知识库问题的向量
    title_embeddings = embeddings[1:]

    # 3. 计算用户输入与所有知识库问题的余弦相似度
    similarities = cosine_similarity(input_embedding, title_embeddings)

    # 4. 找到相似度最高的问题的索引
    similarity_index = np.argmax(similarities)
    similarity_max = similarities[similarity_index]

    # --- 优化建议：设置一个相似度阈值 ---
    # 如果最高相似度低于这个值，就认为没有匹配到
    threshold = 0.7
    if similarity_max < threshold:
        return {"title": "无", "reply": "抱歉，我不太明白您的意思"}

    # 5. 返回最佳匹配的回答
    if similarity_index != -1:
        reply_list = content[similarity_index]['reply']
        # 随机选择一个回复
        reply = random.choice(reply_list)
        return {
            "title": content[similarity_index]['title'],
            "reply": reply
        }

    # 兜底回复
    return {"title": "无", "reply": "抱歉，我不太明白您的意思"}

# 运行
def main():
    try:
        print("正在加载预训练模型...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("模型加载完成！")
        print("\n--- 聊天机器人已启动 (输入 'quit' 退出) ---")
        while True:
            # 接受用户输入
            input_str = input("用户：")
            if input_str.lower() == 'quit':
                print("机器人：再见！")
                break

            # 寻找匹配的答复
            result = answer(input_str, content, model)
            # 输出结果
            print(f"匹配到问题: {result['title']} \n回答： {result['reply']}\n")
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == '__main__':
    main()