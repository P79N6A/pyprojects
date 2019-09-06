import gensim
sentences = [['我', '来自', '中国'], ['她', '是', '美国','人']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1, workers=4)

print(model.wv.__getitem__(u'她'))
sim1 = model.wv.similarity(u'我', u'她')
print(sim1)

word = model.wv.most_similar(u'中国', topn=1)
for item in word:
    print(item)