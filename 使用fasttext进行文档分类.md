#fasttext原理
fasttext提供了一种有效且快速的方式生成词向量以及进行文档分类。
fasttext模型输入一个词的序列，输出这个词序列属于不同类别的概率。fasttext模型架构和Word2Vec中的CBOW模型很类似。不同之处在于，fasttext预测标签，而CBOW模型预测中间词。fasttext设计的初衷就是为了作为一个文档分类器，副产品是也生成了词向量。
![使用fasttext进行文档分类-图2](picture/使用fasttext进行文档分类-图2.png)

#fasttext特性
##n-gram
在词袋模型中，把单词当做独立的个体，没有考虑词前后的关系。比如"我打你"和“你打我“，使用词袋模型的话，这两句话是完全一样的。
词袋的特征为：

	["我",“打“,”你”]
	
"我打你"和“你打我“对应的特征向量均为：

	[1,1,1]

n-gram是对词袋模型的一种改善，它会关注一个单词的前后关系，比如n-gram中最常见的2-gram，就关注单词的前一个词，比如"我打你",就可以拆分为"我打"和"打你"。这两句话一起建模的话，2-gram对应的特征为：

	["我打"，"打你"，"你打"，"打我"]

"我打你"对应的特征向量为：

	[1,1,0,0]

"你打我"对应的特征向量为：

	[0,0,1,1]

与Word2Vec使用词袋模型不同，fasttext使用了n-gram模型，因此fasttext可以更有效的表达词前后的之间的关系。
##高效率
fasttext在使用标准多核CPU的情况下10分钟内处理超过10亿个词汇，特别是与深度模型对比，fastText能将训练时间由数天缩短到几秒钟。使用一个标准多核CPU，得到了在10分钟内训练完超过10亿词汇量模型的结果。

#安装fasttext
fasttext的安装非常简便，直接从github上同步最新的代码并进行安装即可。

	$ git clone https://github.com/facebookresearch/fastText.git
	$ cd fastText
	$ pip install .

#预训练模型
facebook已经基于其收集的海量语料，训练好了fasttext的词向量模型，目前已经支持了150多种语言。有需要的读者可以直接下载并使用，对应的链接为：

	https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
	
![使用fasttext进行文档分类](picture/使用fasttext进行文档分类-图1.png)

#数据集
数据集依然使用搜狗实验室提供的"搜狐新闻数据"，该数据来自搜狐新闻2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据，提供URL和正文信息。
对应的网址为：

	http://www.sogou.com/labs/resource/cs.php

数据文件的格式为：
	<doc>
	
	<url>页面URL</url>
	
	<docno>页面ID</docno>
	
	<contenttitle>页面标题</contenttitle>
	
	<content>页面内容</content>
	
	</doc>
	
我们可以看到数据文件中并没有标记页面内容属于哪个频道，如果需要做文档分类，搜狗提供了页面URL和频道之间的映射关系。

![使用fasttext进行文档分类-图3](picture/使用fasttext进行文档分类-图3.png)

下载SogouTCE文件，可以看到具体的映射关系举例如下：

	http://www.xinhuanet.com/auto/	汽车
	http://www.xinhuanet.com/fortune	财经
	http://www.xinhuanet.com/internet/	IT
	http://www.xinhuanet.com/health/	健康
	http://www.xinhuanet.com/sports	体育
	http://www.xinhuanet.com/travel	旅游
	http://www.xinhuanet.com/edu	教育
	http://www.xinhuanet.com/employment	招聘
	http://www.xinhuanet.com/life	文化
	http://www.xinhuanet.com/mil	军事
	http://www.xinhuanet.com/olympics/	奥运
	http://www.xinhuanet.com/society	社会

#数据清洗
搜狐新闻数据的文件默认编码格式为gb18030，因此解压缩后要线转换成utf-8格式。

	tar -zxvf news_sohusite_xml.full.tar.gz
	cat news_sohusite_xml.dat | iconv -f gb18030 -t utf-8 > news_sohusite_xml-utf8.txt
	
转换完格式后查看文件内容，文件以xml形式记录，举例如下：

	<doc>
	<url>http://gongyi.sohu.com/s2008/sourceoflife/</url>
	<docno>f2467af22cd2f0ea-34913306c0bb3300</docno>
	<contenttitle>中国西部是地球上主要干旱带之一，妇女是当地劳动力．．．</contenttitle>
	<content>同心县地处宁夏中部干旱带的核心区，　冬寒长，春暖迟，夏热短，秋凉早，干旱少雨，蒸发强烈，风大沙多。主要自然灾害有沙尘暴、干热风、霜冻、冰雹等，其中以干旱危害最为严重。由于生态环境的极度恶劣，导致农村经济发展缓慢，人民群众生产、生活水平低下，靠天吃饭的被动局
	面依然存在，同心，又是国家级老、少、边、穷县之一…［详细］</content>
	</doc>

但是数据文件并不是标准的xml格式，如下所示，该文件相对标准的xml格式缺少了根元素。

	<doc>
	    <url></url>
	    <docno></docno>
	    <contenttitle></contenttitle>
	    <content></content>
	</doc>
	<doc>
	    <url></url>
	    <docno></docno>
	    <contenttitle></contenttitle>
	    <content></content>
	</doc>

所有的doc节点都直接是最顶层，没有根节点。因此要添加根节点使该文本文件符合xml文件的规范，最简单的一种形式就是在文件的开始和结尾添加根元素标签。
	<?xml version="1.0" encoding="utf-8"?>
	<docs>
		<doc>
		    <url></url>
		    <docno></docno>
		    <contenttitle></contenttitle>
		    <content></content>
		</doc>
		<doc>
		    <url></url>
		    <docno></docno>
		    <contenttitle></contenttitle>
		    <content></content>
		</doc>
	</docs>

可以直接使用文本编辑工具在数据文件的开始和结尾进行修改，但是这有可能导致你的终端因为内存使用过大而崩溃。一种比较稳妥的做法是使用程序完成。

	def make_xml():
		print "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
	    print "<docs>"
	    with open("data/news_sohusite_xml-utf8.txt") as F:
	        for line in F:
	            print line
	        F.close()
	    print "</docs>"
	    
在终端执行该程序，并将标准输出的结果保存即可。
