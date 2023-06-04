import pandas as pd
import string
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation
print(tf.keras.__version__)

# 展示数据
poems_text = pd.read_table('poems1.txt', header=None)
poems_text.columns = ["text"]
print(poems_text.head())

f = open('poems.txt',"r",encoding='utf-8')
poems = []
for line in f.readlines():
  try:
    title, poem = line.strip().split(':')
    poem = poem.replace(' ','')
    poem = poem.replace('\n','')
    poem = poem.replace('，', '')
    poem = poem.replace('。','')
    poems.append(list(poem))
  except ValueError as e:
    pass

print(poems[0][:])
print(len(poems))  # 一共4w3千条诗文
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(poems) # 将文本标签化并生成文本字典
vocab_size=len(tokenizer.word_index)+1 # 字典的长度是token 下标 + 1
poems_digit = tokenizer.texts_to_sequences(poems) # 将文本转化为数字的序列
poems_digit = tf.keras.preprocessing.sequence.pad_sequences(poems_digit,maxlen=50,padding='post')

X=poems_digit[:,:-1]
Y=poems_digit[:,1:]
print(X.shape)
print(Y.shape)

Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size) # 将Y 转化为 one-hot 向量
print(Y.shape)

hidden_size1=128
hidden_size2=64
# 将一些网络层通过.add()堆叠起来，就构成了一个模型
model = Sequential()
# 一个Embedding 层
model.add(Embedding(input_dim=vocab_size,output_dim=hidden_size1,input_length=49,mask_zero=True))
# 一个 LSTM 层
model.add(LSTM(hidden_size2,return_sequences=True))
# 一个全连接层
model.add(Dense(vocab_size))
# 一个softmax归一化函数
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
# 训练10次，每次64个batch，每个batch中traning 和 validation 的比例是4:1
model.fit(X, Y, epochs=10, batch_size=64, validation_split=0.2)

model.save('Poetry_LSTM.h5')

model = tf.keras.models.load_model('Poetry_LSTM.h5')

poem_incomplete = '雨****轩****可****爱****'
poem_index = []
poem_text = ''
for i in range(len(poem_incomplete)):
  current_word = poem_incomplete[i]

  if current_word != '*':
    index = tokenizer.word_index[current_word]

  else:
    x = np.expand_dims(poem_index, axis=0)  # 使用已有的poem_index 内容进行预测
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=49, padding='post')  # 输入内容padding补齐
    y = model.predict(x)[0, i]  # 预测输出结果

    y[0] = 0
    index = y.argmax()
    current_word = tokenizer.index_word[index]  # 将输出结果的概率转化为文本内容

  poem_index.append(index)
  poem_text = poem_text + current_word

poem_text = poem_text[0:]
print(poem_text[0:5])
print(poem_text[5:10])
print(poem_text[10:15])
print(poem_text[15:20])