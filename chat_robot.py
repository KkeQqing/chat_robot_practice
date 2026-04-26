from modelscope.hub.snapshot_download import snapshot_download
import json
import random
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# --- 配置 ---
# 使用 ModelScope 镜像源下载模型
MODEL_ID = 'bert-base-chinese'  # ModelScope 上的模型 ID
LOCAL_MODEL_DIR = './local_bert_model/'  # 模型将被下载并保存到的本地目录
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def download_model():
    """
    从 ModelScope 下载模型到本地目录
    """
    print(f"正在从 ModelScope 下载模型 {MODEL_ID} 到 {LOCAL_MODEL_DIR} ...")
    try:
        # snapshot_download 会自动处理国内镜像加速
        model_dir = snapshot_download(MODEL_ID)
        # 将模型复制或移动到固定目录（这里简化处理，直接使用下载路径或进行复制）
        # 为了简单起见，我们直接使用下载返回的路径，或者你可以用 shutil.copytree 复制到 LOCAL_MODEL_DIR
        # 这里假设我们直接使用下载管理器下载到的目标位置，或者你可以在下载时指定 cache_dir
        # 注意：snapshot_download 默认会下载到 ModelScope 的缓存目录
        # 我们可以通过参数指定下载目录，或者直接使用缓存目录的路径
        return model_dir
    except Exception as e:
        print(f"模型下载失败: {e}")
        # 如果自动下载失败，尝试手动指定路径或检查网络
        # 如果手动下载了模型，可以将 LOCAL_MODEL_DIR 指向你的模型文件夹
        return LOCAL_MODEL_DIR


class BertChatbot:
    def __init__(self, model_dir):
        print(f"正在从本地目录加载模型: {model_dir} ...")
        # 1. 从本地目录加载分词器和模型
        # 注意：这里不再传入模型名称，而是传入本地路径
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)
        self.model.to(DEVICE)
        self.model.eval()  # 设置为评估模式
        print("模型加载完成！")

    def get_sentence_embedding(self, text):
        # 编码文本
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # 将张量移动到与模型相同的设备上
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 提取 [CLS] 向量
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embedding

    # ... (find_best_answer 方法保持不变，代码过长此处省略，使用上文提供的即可)


# --- 主程序 ---
def main():
    # 1. 下载或确认模型在本地
    # 如果是第一次运行，会自动下载；如果已下载过，snapshot_download 会直接使用缓存
    model_path = download_model()

    # 2. 初始化机器人（传入本地模型路径）
    bot = BertChatbot(model_path)

    # 3. 加载知识库
    kb = load_knowledge_base('templet.txt')
    if not kb:
        return

    print("\n--- BERT 智能聊天机器人已启动 ---")
    while True:
        input_str = input("用户：")
        if input_str.lower() == 'quit':
            break
        result = bot.find_best_answer(input_str, kb)
        print(f"匹配: {result['title']} (置信度: {result.get('score', 0):.2f})")
        print(f"回答： {result['reply']}\n")


if __name__ == '__main__':
    main()