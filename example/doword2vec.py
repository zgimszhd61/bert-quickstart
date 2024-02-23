!pip install gensim
from gensim.models import Word2Vec

# 假设我们有一些已经预处理并分词的句子
sentences = [
    ['this', 'is', 'a', 'sample', 'sentence'],
    ['this', 'is', 'another', 'sentence', 'example']
]

# 初始化模型
# size参数表示向量维度
# window参数表示当前词与预测词在一个句子中的最大距离
# min_count表示词频少于min_count的单词会被忽略
# workers表示训练的线程数
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 构建词汇表
model.build_vocab(sentences)

# 训练模型
# total_examples表示句子数
# epochs表示迭代次数
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

# 保存模型
model.save("word2vec.model")

# 加载模型
model = Word2Vec.load("word2vec.model")

# 使用模型
# 获取单词的向量表示
vector = model.wv['sample']

# 找到最相似的词汇
similar_words = model.wv.most_similar('sample')

# 打印结果
print(vector)
for word, similarity in similar_words:
    print(word, similarity)


