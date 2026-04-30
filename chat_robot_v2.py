import json
import random
import jieba
import numpy as np
from gensim.models import FastText

# ===================== 全局配置 =====================
# 停用词表（可自行扩展）
STOP_WORDS = {
    "的", "了", "是", "我", "你", "吗", "呢", "啊", "就", "都", "还", "在", "有",
    "和", "与", "吧", "对", "这", "那", "个", "里", "去", "来", "哦", "呀", "呢",
    "什么", "怎么", "哪里", "多少", "啦", "嘛", "呗"
}

# 相似度阈值（低于这个值 → 不匹配）
SIMILARITY_THRESHOLD = 0.45

# 文件路径配置
CORPUS_PATH = "corpus.txt"
SEG_RESULT_PATH = "fenci_result.txt"
TEMPLATE_PATH = "templet.txt"
FASTTEXT_MODEL_PATH = "fasttext.model"


# ===================== 工具函数 =====================
def cut_words(text, stop_words):
    """分词 + 停用词过滤"""
    words = jieba.lcut(text.strip())
    # 过滤：停用词、空字符、单字
    words = [w for w in words if w not in stop_words and w.strip() and len(w) > 1]
    return words


def seg_corpus():
    """语料预处理：分词并保存"""
    try:
        with open(CORPUS_PATH, encoding='utf-8') as f1, \
                open(SEG_RESULT_PATH, "w", encoding='utf-8') as f2:

            for line in f1:
                line = line.strip()
                if not line:
                    continue
                words = cut_words(line, STOP_WORDS)
                f2.write(' '.join(words) + '\n')

        print("✅ 语料分词 + 停用词过滤完成")
    except FileNotFoundError:
        print(f"❌ 未找到文件：{CORPUS_PATH}")


def train_fasttext():
    """训练 FastText 模型"""
    from gensim.models.word2vec import Text8Corpus
    sentences = Text8Corpus(SEG_RESULT_PATH)
    model = FastText(
        sentences=sentences,
        vector_size=100, # 词向量的维度大小。这个参数决定了每个词将被表示为一个多长的向量。
        window=3, #上下文窗口的大小。它定义了模型在预测一个词时，会考虑其前后多少个词
        min_count=1, #词频阈值。模型会忽略在语料库中出现次数低于这个值的词
        workers=4, #用于训练模型的线程数
        epochs=10 #  训练的迭代轮数
    )
    model.save(FASTTEXT_MODEL_PATH)
    print("✅ FastText 模型训练完成")


def load_model():
    """加载模型（不存在则训练）"""
    try:
        model = FastText.load(FASTTEXT_MODEL_PATH)
        print("✅ 模型加载成功")
        return model
    except:
        print("🔧 模型不存在，开始训练...")
        train_fasttext()
        return FastText.load(FASTTEXT_MODEL_PATH)


def load_template():
    """加载问答模板"""
    with open(TEMPLATE_PATH, encoding='utf-8') as f:
        content = json.load(f)
    print("✅ 问答模板加载完成")
    return content


# ===================== 核心回答函数 =====================
def get_best_answer(user_input, content, model):
    """使用 FastText 计算语义相似度，返回最优回答"""
    # 用户输入分词 + 过滤
    input_words = cut_words(user_input, STOP_WORDS)

    if not input_words:
        return {"title": "无", "reply": "请输入有效内容~"}

    max_sim = 0
    best_idx = -1

    # 遍历所有模板问题
    for i, item in enumerate(content):
        title_words = cut_words(item['title'], STOP_WORDS)
        try:
            sim = model.wv.n_similarity(input_words, title_words)
        except:
            sim = 0

        if sim > max_sim:
            max_sim = sim
            best_idx = i

    # 阈值判断
    if max_sim < SIMILARITY_THRESHOLD or best_idx == -1:
        return {"title": "无", "reply": "抱歉，我不太明白您的意思"}

    # 随机回答
    reply = random.choice(content[best_idx]['reply'])
    return {
        "title": content[best_idx]['title'],
        "reply": reply,
        "score": round(float(max_sim), 2)
    }


# ===================== 主程序 =====================
def main():
    seg_corpus()
    ft_model = load_model()
    content = load_template()

    print("\n--- 🤖 智能聊天机器人已启动（输入 quit 退出）---\n")

    while True:
        user_input = input("用户：").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("机器人：再见！")
            break

        result = get_best_answer(user_input, content, ft_model)
        print(f"机器人：{result['reply']}")
        print(f"【匹配问题：{result['title']} 】\n")


if __name__ == '__main__':
    main()