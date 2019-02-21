# Scikit-learn 
Scikit-learn是广受欢迎的入门级机器学习库，包含大量的机器学习算法和特征提取实现，使用非常简便。Scikit-learn实现的是浅层学习算法，神经网络仅实现了多层感知机。
Scikit-learn的安装方式如下：

	pip install scikit-learn
	
# TensorFlow
TensorFlow是谷歌基于DistBelief进行研发的第二代人工智能学习系统,可被用于语音识别或图像识别等多项机器学习和深度学习领域.它可在小到一部智能手机、大到数千台数据中心服务器的各种设备上运行。TensorFlow的安装方式如下：

	pip install tensorflow
	
# Keras
Keras是一个高级别的Python神经网络框架，能在TensorFlow或者 Theano 上运行。Keras的作者、谷歌AI研究员Francois Chollet宣布了一条激动人心的消息，Keras将会成为第一个被添加到TensorFlow核心中的高级别框架，这将会让Keras变成Tensorflow的默认API。
Keras的安装非常简便，使用pip工具即可。

	pip install keras

如果需要使用源码安装，可以直接从GitHub上下载对应源码。

>https://github.com/fchollet/keras


然后进入Keras目录安装即可。

	python setup.py install

# Anaconda
Anaconda是一个用于科学计算的Python开发平台，支持 Linux，Mac和Windows系统，提供了包管理与环境管理的功能，可以很方便地解决多版本Python并存、切换以及各种第三方包安装问题。Anaconda利用conda命令来进行package和environment的管理，并且已经包含了Python和相关的配套工具。Anaconda集成了大量的机器学习库以及数据处理必不可少的第三方库，比如NumPy，SciPy，Scikit-Learn以及TensorFlow等。
Anaconda的安装非常方便，从其官网的下载页面选择对应的安装包即可。
以我的Mac本为例，安装对应Anaconda安装包后，使用如下命令查看当前用户的profile文件的内容。

	cat ~/.bash_profile

我们可以发现在当前用户的profile文件的最后增加了如下内容，表明已经将Anaconda的bin目录下的命令添加到了PATH变量中，可以像使用系统命令一样直接使用Anaconda的命令行工具了。

	# added by Anaconda2 5.0.0 installer
	export PATH="/anaconda2/bin:$PATH"

Anaconda强大的包管理以及多种Python环境并存使用主要以来于conda命令，常用的conda命令列举如下。

	# 创建一个名为python27的环境，指定Python版本是2.7
	conda create --name python27 python=2.7
	# 查看当前环境下已安装的包
	conda list
	# 查看某个指定环境的已安装包
	conda list -n python27
	# 查找package信息
	conda search numpy
	# 安装package
	conda install -n python27 numpy
	# 更新package
	conda update -n python27 numpy
	# 删除package
	conda remove -n python27 numpy
	
假设我们已经创建一个名为python27的环境，指定Python版本是2.7，激活该环境的方法如下。

	source activate python27

如果要退出该环境，命令如下所示。
	
	source deactivate

在python27的环境下查看Python版本，果然是2.7版本。

	maidou:3book liu.yan$ source activate python27
	(python27) maidou:3book liu.yan$ 
	(python27) maidou:3book liu.yan$ python 
	Python 2.7.14 |Anaconda, Inc.| (default, Oct  5 2017, 02:28:52) 
	[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
	Type "help", "copyright", "credits" or "license" for more information.
	>>> 

查看python27环境下默认安装了哪些包，为了避免显示内容过多，过滤前6行查看。

	conda list | head -6
	# packages in environment at /anaconda2/envs/python27:
	#
	ca-certificates           2017.08.26           ha1e5d58_0  
	certifi                   2017.7.27.1      py27h482ffc0_0  
	libcxx                    4.0.1                h579ed51_0  
	libcxxabi                 4.0.1                hebd6815_0 

统计包的个数，除去2行的无关内容，当前环境下有16个包。

	conda list | wc -l
      18
      
查看目前一共具有几个环境，发现除了系统默认的root环境，又多出了我们创建的python27环境。

	conda info --envs
	# conda environments:
	#
	python27                 /anaconda2/envs/python27
	root                  *  /anaconda2

在python27环境下安装Anaconda默认的全部安装包，整个安装过程会比较漫长，速度取决于你的网速。

	conda install anaconda
	Fetching package metadata ...........
	Solving package specifications: .
	Package plan for installation in environment /anaconda2/envs/python27:

继续统计包的个数，除去2行的无关内容，当前环境下已经有238个包了。

	conda list | wc -l
      240

Anaconda默认安装的第三方包里没有包含TensorFlow和Keras，需要使用命令手工安装，以TensorFlow为例，可以使用conda命令直接安装。

	conda install tensorflow

同时也可以使用pip命令直接安装。
	
	pip install tensorflow

本书一共创建了两个环境，分别是python27和python36，顾名思义对应的Python版本分别为2.7和3.6，用于满足不同案例对python版本的不同要求。

# Gensim
Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。它支持包括TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法，支持流式训练，并提供了诸如相似度计算，信息检索等一些常用任务的API接口。
Gensim的安装方式如下：

	pip install gensim

# NTLK
NLTK由Steven Bird和Edward Loper在宾夕法尼亚大学计算机和信息科学系开发，在NLP领域中，最常使用的一个Python库。
NTLK的安装方式如下：

	pip install ntlk

NTLK分为模型和数据两部分，其中数据需要单独下载。

	>>>import nltk
	>>>nltk.download()

推荐选择all，设置好下载路径，然后点击Download，系统就开始下载。NLTK的数据包了，下载的时间比较漫长，大家要耐心等待。如果有个别数据包无法下载，可以切换到All Packages标签页，双击指定的包来进行下载。

# Jieba
Jieba，经常被人昵称为结巴，是最受欢迎的中文分词工具，安装方式如下：

	pip install jieba


# Jupyter notebook
Jupyter notebook中使用Anaconda中的环境需要单独配置，默认情况下使用的是系统默认的Python环境，以使用advbox环境为例。 首先在默认系统环境下执行以下命令，安装ipykernel。

	conda install ipykernel
	conda install -n advbox ipykernel

在advbox环境下激活，这样启动后就可以在界面上看到advbox了。

	python -m ipykernel install --user --name advbox --display-name advbox 

# GPU服务器
当数据量大或者计算量大时，GPU几乎成为必选，尤其是使用CNN和RNN时，几乎就是CPU杀手。目前主流的云上都提供了GPU服务器。以百度云为例，默认支持的tensorflow的GPU版本是1.4。
当你习惯使用python2.*时，推荐使用的组合为：
	
	- tensorflow-gpu==1.4
	- keras==2.1.5
	- python==2.7
	
当你习惯使用python5.*时，推荐使用的组合为：
	
	- tensorflow-gpu==1.4
	- keras==2.1.5
	- python==3.5
