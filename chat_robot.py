import json
import random
import math
from os import replace
import jieba
from gensim.models import word2vec

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
    寻找与用户输入最相似的回答
    :param input: 用户输入的字符串
    :param content: 包含问题和回答的列表
    :param model: 训练好的 Word2Vec 模型
    :return: 包含 title 和 reply 的字典
    """
    # 存储最大相似度
    similarityMax = 0
    # 存储最大相似度问句的下标
    similarityIndex = -1

    # 对用户输入做分词处理
    input_word_arr = list(jieba.cut(input))

    # 遍历规则库
    for i in range(len(content)):
        # 对知识库中的问题进行分词
        title_word_arr = list(jieba.cut(content[i]['title'].replace(" ", "")))

        # 使用try...except语法来做余弦相似度计算，避免因词向量小而引发报错
        try:
            similarity = model.wv.n_similarity(input_word_arr, title_word_arr)
        except Exception:
            similarity = 0

        # 储存当前最大相似度及其下标
        if similarityMax < similarity:
            similarityMax = similarity
            similarityIndex = i

    # 随机取一个回复，如果similarityIndex为-1，则说明未匹配到相似语句
    if similarityIndex != -1:
        reply_list = content[similarityIndex]['reply']
        reply_index = math.floor(random.random() * len(reply_list))
        return {
            "title": content[similarityIndex]['title'],
            "reply": reply_list[reply_index]
        }

    # 未找到匹配项时返回默认回答
    return {"title": "无", "reply": "抱歉，我不太明白您的意思"}

# 运行
def main():
    try:
        model = word2vec.Word2Vec.load("fenci_model.model")
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