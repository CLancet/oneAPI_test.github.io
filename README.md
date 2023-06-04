# 借助tensorflow 实现诗词生成器
# oneAPI_test.github.io
# 实现目标
对已有大量古诗词数据通过循环神经网络进行训练，建立一个自动生成古诗词的模型，通过输入每句诗或词的首字，生成完整的一首诗词。

# 实现工具
因为需要循环神经网络模型，本次实现利用到oneAPI工具中的tensorflow。

# 文本生成理论部分：

模型原型：循环神经网络RNN。T时刻的输出能够传递给下一个时间t+1。RNN对文本生成的优势是处理短序列文本时兼具准确和高效。但是RNN并不能处理长序列文本。当文本序列过长时，最初的输入对后续文本生成的影响会越来越小。

针对RNN模型的劣势，做出升级的LSTM模型。LSTM（长短期记忆网络）是一种常用的循环神经网络（RNN）模型，旨在解决传统RNN模型中梯度消失和梯度爆炸的问题。LSTM模型通过引入门控机制来控制信息的流动和保留，从而更好地捕捉序列数据中的长期依赖关系。

LSTM模型中的每个单元包含三个门控单元：输入门、遗忘门和输出门。输入门控制新输入信息的流入，遗忘门控制旧信息的流出，输出门控制输出的信息。

此外还有两个重要的元素，传送带和新输入。

传送带的作用是整合经过遗忘门和输入门处理的过的数据，生成整合后的结果。新输出则是利用tan函数处理数据，防止梯度爆炸。，在与输出门的输出结果相乘得到新的输出即t时刻的输出，也是下一时刻的输入。

LSTM模型通过这些门控单元来控制信息的流动，不同的门有不同的可训练矩阵，从而更好地捕捉序列数据中的长期依赖关系。即能够保持重要信息对后续生成的影响。

LSTM模型在自然语言处理、语音识别、时间序列预测等领域得到了广泛应用。由于其能够处理长序列数据和长期依赖关系，LSTM模型在处理自然语言处理任务中的文本分类、情感分析、机器翻译等方面表现出色。

# 文本生成代码实现：
     #部分包的导入,早期的tensorflow不包含keras API,TensorFlow 2.0 及以上版本中包含了 Keras API，可以直接使用 TensorFlow 中的 Keras API，而不需要额外安装 Keras 库。代码中更新了部分导包过程
     import pandas as pd
     import string
     import numpy as np
     from keras.models import Sequential
     from keras.layers import Embedding, LSTM, Dense, Activation
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
    okenizer = tf.keras.preprocessing.text.Tokenizer()
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
## 将一些网络层通过.add()堆叠起来，就构成了一个模型
    model = Sequential()
## 一个Embedding 层
    model.add(Embedding(input_dim=vocab_size,output_dim=hidden_size1,input_length=49,mask_zero=True))
## 一个 LSTM 层
    model.add(LSTM(hidden_size2,return_sequences=True))
## 一个全连接层
    model.add(Dense(vocab_size))
## 一个softmax归一化函数
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
## 训练10次，每次64个batch，每个batch中traning 和 validation 的比例是4:1
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
本机无法提供足够的运行内存和算力，在oneAPI上解决了内存不够的问题，但是受限于服务器资源限制，后续部分未能完成。
