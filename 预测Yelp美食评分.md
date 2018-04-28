# Yelp简介
Yelp是美国著名商户点评网站，创立于2004年，囊括各地餐馆、购物中心、酒店、旅游等领域的商户，用户可以在Yelp网站中给商户打分，提交评论，交流购物体验等。在Yelp中搜索一个餐厅或者旅馆，能看到它的简要介绍以及网友的点论，点评者还会给出多少星级的评价，通常点评者都是亲身体验过该商户服务的消费者，评论大多形象细致。
# Yelp Reviews
Yelp Reviews是Yelp为了学习目的而发布的一个开源数据集。它包含了由数百万用户评论，商业属性和来自多个大都市地区的超过20万张照片。这是一个常用的全球NLP挑战数据集，包含5,200,000条评论，174,000条商业属性。 数据集下载地址为：

> https://www.yelp.com/dataset/download

Yelp Reviews格式分为JSON和SQL两种，以JSON格式为例,其中最重要的review.json,包含评论数据，格式如下：

	{
    // string, 22 character unique review id
    "review_id": "zdSx_SD6obEhz9VrW9uAWA",

    // string, 22 character unique user id, maps to the user in user.json
    "user_id": "Ha3iJu77CxlrFm-vQRs_8g",

    // string, 22 character business id, maps to business in business.json
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // integer, star rating
    "stars": 4,

    // string, date formatted YYYY-MM-DD
    "date": "2016-03-09",

    // string, the review itself
    "text": "Great place to hang out after work: the prices are decent, and the ambience is fun. It's a bit loud, but very lively. The staff is friendly, and the food is good. They have a good selection of drinks.",

    // integer, number of useful votes received
    "useful": 0,

    // integer, number of funny votes received
    "funny": 0,

    // integer, number of cool votes received
    "cool": 0}

# 数据清洗
Yelp Reviews文件格式为JSON和SQL，使用起来并不是十分方便。专门有个开源项目用于解析该JSON文件：

> https://github.com/Yelp/dataset-examples

该项目可以将Yelp Reviews的Yelp Reviews转换成CSV格式，便于进一步处理，该项目的安装非常简便，同步完项目后直接安装即可。

	git clone https://github.com/Yelp/dataset-examples
	python setup.py install

假如需要把review.json转换成CSV格式，命令如下：

	python json_to_csv_converter.py /dataset/yelp/dataset/review.json

命令执行完以后，就会在review.json相同目录下生成对应的CSV文件review.csv。查看该CSV文件的表头，内容如下，其中最重要的两个字段就是text和stars，分别代表评语和打分。

	#CSV格式表头内容：
	#funny,user_id,review_id,text,business_id,stars,date,useful,cool
		
使用pandas读取该CSV文件，开发阶段可以指定仅读取前10000行。
	
	#开发阶段读取前10000行
	df = pd.read_csv(filename,sep=',',header=0,nrows=10000)
	
pandas的可以配置的参数非常多，其中比较重要的几个含义如下：

- sep : str, default ‘,’。指定分隔符。
- header: int or list of ints, default ‘infer’。指定行数用来作为列名，数据开始行数。如果文件中没有列名，设置为None。设置为0则认为第0行是列名
- nrows : int, default None 需要读取的行数（从文件头开始算起）
- skiprows : list-like or integer, default None。需要忽略的行数（从文件开始处算起），或需要跳过的行号列表（从0开始)。
- skip\_blank\_lines : boolean, default True。如果为True，则跳过空行；否则记为NaN

按照列名直接获取数据，读取评论内容和打分结果，使用list转换成list对象。
	
	text=list(df['text'])
	stars=list(df['stars'])

查看打分结果的分布。

    #显示各个评分的个数
    print df.describe()

分布结果如下，一共有10000个评分，最高分5分，最低1分，平均得分为3.74，50%的人打分不大于4分。


	              funny         stars        useful          cool
	count  10000.000000  10000.000000  10000.000000  10000.000000
	mean       0.649800      3.743800      1.669500      0.777800
	std        1.840679      1.266381      3.059511      1.958625
	min        0.000000      1.000000      0.000000      0.000000
	25%        0.000000      3.000000      0.000000      0.000000
	50%        0.000000      4.000000      1.000000      0.000000
	75%        1.000000      5.000000      2.000000      1.000000
	max       46.000000      5.000000     95.000000     43.000000

# 参考文献

- https://www.cnblogs.com/datablog/p/6127000.html