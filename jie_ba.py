import jieba

text = "八百标兵奔北坡，炮兵并排北边跑"

result = jieba.cut(text, cut_all=True)
print(list(result))
result = jieba.cut(text, cut_all=False)
print(list(result))
result = jieba.cut_for_search(text)
print(list(result))


word_vector_list = ["我们","来","学习","基于","Python","的","AI","实践","课程"]

def get_word_vector_result(word, vocabulary):
    word_vector_result = []
    for i in vocabulary:
        if i == word:
            word_vector_result.append(1)
        else:
            word_vector_result.append(0)
    return word_vector_result

word1 = "Python"
print(get_word_vector_result(word1,word_vector_list))