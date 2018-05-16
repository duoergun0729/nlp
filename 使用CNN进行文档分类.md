# 数据集
Yelp Reviews是Yelp为了学习目的而发布的一个开源数据集。它包含了由数百万用户评论，商业属性和来自多个大都市地区的超过20万张照片。这是一个常用的全球NLP挑战数据集，包含5,200,000条评论，174,000条商业属性。 数据集下载地址为：

	https://www.yelp.com/dataset/download

Yelp Reviews格式分为JSON和SQL两种，以JSON格式为例,其中最重要的review.json,包含评论数据。Yelp Reviews数据量巨大，非常适合验证CNN模型。

# 特征提取

特征提取使用词袋序列模型和词向量。
## 词袋序列模型
词袋序列模型是在词袋模型的基础上发展而来的，相对于词袋模型，词袋序列模型可以反映出单词在句子中的前后关系。keras中通过Tokenizer类实现了词袋序列模型，这个类用来对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示，创建该类时，需要设置词典的最大值。

	tokenizer = Tokenizer(num_words=None)
	
Tokenizer类的示例代码如下：

	from keras.preprocessing.text import Tokenizer
	
	text1='some thing to eat'
	text2='some thing to drink'
	texts=[text1,text2]
	
	tokenizer = Tokenizer(num_words=None) 
	#num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
	tokenizer.fit_on_texts(texts)
	
	# num_words=多少会影响下面的结果，行数=num_words
	print( tokenizer.texts_to_sequences(texts)) 
	#得到词索引[[1, 2, 3, 4], [1, 2, 3, 5]]
	print( tokenizer.texts_to_matrix(texts))  
	# 矩阵化=one_hot
	[[ 0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
	 [ 0.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.]]

在处理Yelp数据集时，把每条评论看成一个词袋序列，且长度固定。超过固定长度的截断，不足的使用0补齐。

	#转换成词袋序列，max_document_length为序列的最大长度
	max_document_length=200
	
	#设置分词最大个数 即词袋的单词个数
	tokenizer = Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(text)
	sequences = tokenizer.texts_to_sequences(text)
	 #截断补齐
	x=pad_sequences(sequences, maxlen=max_document_length)
	
## 词向量模型
词向量模型常见实现形式有word2Vec,fastText和GloVe，本章使用最基础的word2Vec，基于gensim库实现。为了提高性能，使用了预训练好的词向量，即使用Google News dataset数据集训练出的词向量。加载方式为：

	model =
	gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

GloVe预训练好的词向量可以从下列地址下载：

	http://nlp.stanford.edu/projects/glove/


# 衡量指标
使用5折交叉验证，并且考核F1的值，训练轮数为10轮，批处理大小为128。
	
	clf = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=128, verbose=0)
		
	#使用5折交叉验证
	scores = cross_val_score(clf, x, encoded_y, cv=5, scoring='f1_micro')
	# print scores
	print("f1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 单层CNN模型
我们尝试使用单层CNN结构来处理Yelp的分类问题。首先通过一个Embedding层，降维成为50位的向量，然后使用一个核数为250，步长为1的一维CNN层进行处理，接着连接一个池化层。为了防止过拟合，CNN层和全连接层之间随机丢失20%的数据进行训练。需要特别指出的是，Embedding层相当于是临时进行了词向量的计算，把原始的词序列转换成了指定维数的词向量序列。

    #CNN参数
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    #池化
    model.add(GlobalMaxPooling1D())

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
                  
损失函数使用交叉熵categorical_crossentropy，优化算法使用adsm，可视化结果如下。

![使用CNN进行文档分类-图1.png](picture/使用CNN进行文档分类-图1.png)

打印CNN的结构。

	model.summary()

输出的结果如下所示，除了显示模型的结构，还可以显示需要训练的参数信息。

	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	embedding_1 (Embedding)      (None, None, 50)          250000    
	_________________________________________________________________
	conv1d_1 (Conv1D)            (None, None, 250)         37750     
	_________________________________________________________________
	global_max_pooling1d_1 (Glob (None, 250)               0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 2)                 502       
	=================================================================
	Total params: 288,252
	Trainable params: 288,252
	Non-trainable params: 0
	_________________________________________________________________
	
当特征提取使用词袋序列，特征数取5000的前提下，结果如下。

<table>
    <tr>
        <td>数据量</td>
        <td>F1值</td>
    </tr>
    <tr>
        <td>1w</td>
        <td>0.86</td>
    </tr>
    <tr>
        <td>10w</td>
        <td>0.92</td>
    </tr>     
</table>

# 单层CNN+MLP模型
在单层CNN的基础上增加一个隐藏层，便于更好的分析CNN层提取的高级特征，该隐藏层结点数为250。

    #CNN参数
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    #池化
    model.add(GlobalMaxPooling1D())


    #增加一个隐藏层
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    #输出层

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

可视化结果如下。

![使用CNN进行文档分类-图2.png](picture/使用CNN进行文档分类-图2.png)

打印CNN的结构。

	model.summary()

输出的结果如下所示，除了显示模型的结构，还可以显示需要训练的参数信息。

	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	embedding_1 (Embedding)      (None, None, 50)          250000    
	_________________________________________________________________
	conv1d_1 (Conv1D)            (None, None, 250)         37750     
	_________________________________________________________________
	global_max_pooling1d_1 (Glob (None, 250)               0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 250)               62750     
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 250)               0         
	_________________________________________________________________
	activation_1 (Activation)    (None, 250)               0         
	_________________________________________________________________
	dense_2 (Dense)              (None, 2)                 502       
	=================================================================
	Total params: 351,002
	Trainable params: 351,002
	Non-trainable params: 0
	_________________________________________________________________
	
当特征提取使用词袋序列，特征数取5000的前提下，结果如下，可以增加数据量可以提升性能。在数据量相同的情况下，比单层CNN效果略好。

<table>
    <tr>
        <td>数据量</td>
        <td>F1值</td>
    </tr>
    <tr>
        <td>1w</td>
        <td>0.87</td>
    </tr>
    <tr>
        <td>10w</td>
        <td>0.93</td>
    </tr>     
</table>

# TextCNN
TextCNN是利用卷积神经网络对文本进行分类的算法，由Yoon Kim中提出，本质上分别使用大小为3，4和5的一维卷积处理文本数据。这里的文本数据可以是定长的词袋序列模型，也可以使用词向量。

![使用CNN进行文档分类-图3](picture/使用CNN进行文档分类-图3.png)

TextCNN的一种实现方式，就是分别使用大小为3，4和5的一维卷积处理输入，然后使用MaxPooling1D进行池化处理，并将处理的结果使用Flatten层压平并展开。把三个卷积层的结果合并，作为下一个隐藏层的输入，为了防止过拟合，丢失50%的数据进行训练。


	 #CNN参数
    embedding_dims = 50
    filters = 100

    # Inputs
    input = Input(shape=[max_document_length])

    # Embeddings layers
    x = Embedding(max_features, embedding_dims)(input)

    # conv layers
    convs = []
    for filter_size in [3,4,5]:
        l_conv = Conv1D(filters=filters, kernel_size=filter_size, activation='relu')(x)
        l_pool = MaxPooling1D()(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)

    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)

    output = Dense(32, activation='relu')(out)

    output = Dense(units=2, activation='softmax')(output)

    #输出层
    model = Model([input], output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
可视化结果如下。

![使用CNN进行文档分类-图4](picture/使用CNN进行文档分类-图4.png)

打印CNN的结构。

	model.summary()

输出的结果如下所示，除了显示模型的结构，还可以显示需要训练的参数信息。

	__________________________________________________________________________________________________
	Layer (type)                    Output Shape         Param #     Connected to                     
	==================================================================================================
	input_1 (InputLayer)            (None, 200)          0                                            
	__________________________________________________________________________________________________
	embedding_1 (Embedding)         (None, 200, 50)      250000      input_1[0][0]                    
	__________________________________________________________________________________________________
	conv1d_1 (Conv1D)               (None, 198, 100)     15100       embedding_1[0][0]                
	__________________________________________________________________________________________________
	conv1d_2 (Conv1D)               (None, 197, 100)     20100       embedding_1[0][0]                
	__________________________________________________________________________________________________
	conv1d_3 (Conv1D)               (None, 196, 100)     25100       embedding_1[0][0]                
	__________________________________________________________________________________________________
	max_pooling1d_1 (MaxPooling1D)  (None, 99, 100)      0           conv1d_1[0][0]                   
	__________________________________________________________________________________________________
	max_pooling1d_2 (MaxPooling1D)  (None, 98, 100)      0           conv1d_2[0][0]                   
	__________________________________________________________________________________________________
	max_pooling1d_3 (MaxPooling1D)  (None, 98, 100)      0           conv1d_3[0][0]                   
	__________________________________________________________________________________________________
	flatten_1 (Flatten)             (None, 9900)         0           max_pooling1d_1[0][0]            
	__________________________________________________________________________________________________
	flatten_2 (Flatten)             (None, 9800)         0           max_pooling1d_2[0][0]            
	__________________________________________________________________________________________________
	flatten_3 (Flatten)             (None, 9800)         0           max_pooling1d_3[0][0]            
	__________________________________________________________________________________________________
	concatenate_1 (Concatenate)     (None, 29500)        0           flatten_1[0][0]                  
	                                                                 flatten_2[0][0]                  
	                                                                 flatten_3[0][0]                  
	__________________________________________________________________________________________________
	dropout_1 (Dropout)             (None, 29500)        0           concatenate_1[0][0]              
	__________________________________________________________________________________________________
	dense_1 (Dense)                 (None, 32)           944032      dropout_1[0][0]                  
	__________________________________________________________________________________________________
	dense_2 (Dense)                 (None, 2)            66          dense_1[0][0]                    
	==================================================================================================
	Total params: 1,254,398
	Trainable params: 1,254,398
	Non-trainable params: 0
	__________________________________________________________________________________________________

当特征提取使用词袋序列，特征数取5000的前提下，结果如下，可以看出增加数据量可以提升性能。在数据量相同的情况下，比单层CNN效果略好，与CNN+MLP效果相当。

<table>
    <tr>
        <td>数据量</td>
        <td>F1值</td>
    </tr>
    <tr>
        <td>1w</td>
        <td>0.88</td>
    </tr>
    <tr>
        <td>10w</td>
        <td>0.92</td>
    </tr>     
</table>

# TextCNN变种
- CNN-rand:设计好 embedding_size 这个超参数后,对不同单词的向量作随机初始化, 后续BP的时候作调整.
- static:拿 pre-trained vectors from word2vec,FastText or GloVe 直接用, 训练过程中不再调整词向量. 这也算是迁移学习的一种思想.
- non-static:pre-trained vectors + fine tuning,即拿word2vec训练好的词向量初始化, 训练过程中再对它们微调.
- multiple channel:类比于图像中的RGB通道, 这里也可以用static与non-static 搭两个通道来搞.

## static版本TextCNN
可以通过使用预先训练的词向量，训练过程中不再调整词向量。最简单的一种实现方式就是使用Word2Vec训练好的词向量。在gensim库中，在Google News dataset数据集训练出的词向量。

	model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
	print model['word'].shape

通过打印某个单词的词向量形状获取预先训练的词向量的维数，本例中为300。设置需要处理的单词的最大个数max_features，然后获取对应单词序列。

    #设置分词最大个数 即词袋的单词个数
    tokenizer = Tokenizer(num_words=max_features,lower=True)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    x=pad_sequences(sequences, maxlen=max_document_length)
    
通过tokenizer对象获取word到对应数字编号的映射关系表。 

    #获取word到对应数字编号的映射关系
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

枚举映射关系表，生成嵌入层的参数矩阵。虽然是在很大数据集上进行了训练，理论上还是存在单词无法查找到对应的词向量的可能，所以需要捕捉这种异常情况。

    #获取词向量的映射矩阵
    embedding_matrix = np.zeros((max_features + 1, embedding_dims))
    for word, i in word_index.items():
        #编号大于max_features的忽略 该字典是按照字典顺序 所以对应的id不一定是顺序的
        if i > max_features:
            continue
        try:
            embedding_matrix[i] = model[word].reshape(embedding_dims)
        except:
            print "%s not found!" % (word)

生成了嵌入层的参数矩阵矩阵后，可以使用该矩阵创建对应的嵌入层。在本例中，预先训练的词向量，训练过程中不再调整词向量，所以需要把trainable设置为False。

	# Inputs
	input = Input(shape=[max_document_length])
	
	# 词向量层，本文使用了预训练word2vec词向量，把trainable设为False
	x = Embedding(max_features + 1,
	                            embedding_dims,
	                            weights=[embedding_matrix],
	                            trainable=False)(input)

打印CNN的结构。

	model.summary()

输出的结果如下所示，除了显示模型的结构，还可以显示需要训练的参数信息。其中1,500,300个参数不可训练，这里指的就是固定的词向量。

	_____________________
	Layer (type)                    Output Shape         Param #     Connected to                     
	==================================================================================================
	input_5 (InputLayer)            (None, 200)          0                                            
	__________________________________________________________________________________________________
	embedding_5 (Embedding)         (None, 200, 300)     1500300     input_5[0][0]                    
	__________________________________________________________________________________________________
	conv1d_13 (Conv1D)              (None, 198, 200)     180200      embedding_5[0][0]                
	__________________________________________________________________________________________________
	conv1d_14 (Conv1D)              (None, 197, 200)     240200      embedding_5[0][0]                
	__________________________________________________________________________________________________
	conv1d_15 (Conv1D)              (None, 196, 200)     300200      embedding_5[0][0]                
	__________________________________________________________________________________________________
	max_pooling1d_13 (MaxPooling1D) (None, 99, 200)      0           conv1d_13[0][0]                  
	__________________________________________________________________________________________________
	max_pooling1d_14 (MaxPooling1D) (None, 98, 200)      0           conv1d_14[0][0]                  
	__________________________________________________________________________________________________
	max_pooling1d_15 (MaxPooling1D) (None, 98, 200)      0           conv1d_15[0][0]                  
	__________________________________________________________________________________________________
	flatten_13 (Flatten)            (None, 19800)        0           max_pooling1d_13[0][0]           
	__________________________________________________________________________________________________
	flatten_14 (Flatten)            (None, 19600)        0           max_pooling1d_14[0][0]           
	__________________________________________________________________________________________________
	flatten_15 (Flatten)            (None, 19600)        0           max_pooling1d_15[0][0]           
	__________________________________________________________________________________________________
	concatenate_5 (Concatenate)     (None, 59000)        0           flatten_13[0][0]                 
	                                                                 flatten_14[0][0]                 
	                                                                 flatten_15[0][0]                 
	__________________________________________________________________________________________________
	dropout_5 (Dropout)             (None, 59000)        0           concatenate_5[0][0]              
	__________________________________________________________________________________________________
	dense_9 (Dense)                 (None, 32)           1888032     dropout_5[0][0]                  
	__________________________________________________________________________________________________
	dense_10 (Dense)                (None, 2)            66          dense_9[0][0]                    
	==================================================================================================
	Total params: 4,108,998
	Trainable params: 2,608,698
	Non-trainable params: 1,500,300
	__________________________________________________________________________________________________
	
当特征提取使用词向量，且使用预训练好的词向量，特征数取5000的前提下，训练过程中词向量相关参数不可改变，结果如下，可以看出增加数据量可以提升性能。在数据量相同的情况下，比单层CNN效果略好，与CNN+MLP效果相当。

<table>
    <tr>
        <td>数据量</td>
        <td>F1值</td>
    </tr>
    <tr>
        <td>1w</td>
        <td>0.88</td>
    </tr>
    <tr>
        <td>10w</td>
        <td>0.91</td>
    </tr>     
</table>

## fine tuning版本的TextCNN
fine tuning版本的TextCNN的最大特点是使用预先训练的词向量，训练过程中词向量的参数参与整个反向传递过程，接受训练和调整。具体实现时设置trainable为True即可。

    # 词向量层，本文使用了预训练word2vec词向量并接受调整，把trainable设为True
    x = Embedding(max_features + 1,
                                embedding_dims,
                                weights=[embedding_matrix],
                                trainable=True)(input)


查看需要训练的参数信息，发现所有参数均可以参与训练过程。

	==================================================================================================
	Total params: 4,108,998
	Trainable params: 4,108,998
	Non-trainable params: 0
	__________________________________________________________________________________________________

当特征提取使用词向量，且使用预训练好的词向量，特征数取5000的前提下，训练过程中词向量相关参数可改变，结果如下，可以看出增加数据量可以提升性能。在数据量相同的情况下，比单层CNN效果略好，与CNN+MLP效果相当，比使用静态词向量的效果略好。

<table>
    <tr>
        <td>数据量</td>
        <td>F1值</td>
    </tr>
    <tr>
        <td>1w</td>
        <td>0.89</td>
    </tr>
    <tr>
        <td>10w</td>
        <td>0.92</td>
    </tr>     
</table>


# 参考文献

- Convolutional Neural Networks for Sentence Classification
- http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/