#coding:gbk

import requests
from lxml import html
from multiprocessing.dummy import Pool as ThreadPool
from pandas import Series
import pygal
from pygal.style import LightColorizedStyle as Lk,LightenStyle as Lp
from datetime import datetime

def header(referer):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/59.0.3071.115 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'authorization':'Bearer 2|1:0|10:1519705142|4:z_c0|92:Mi4xSFYxdUJnQUFBQUFBWUFBS0NxZlZDaVlBQUFCZ0FsVk5OaXFDV3dDblNVc3Q3MXFwQ2ltdzJNNGw0ejg0UmxYZlpR|18ca81465f4288860baa2e22d2bdfe0002e2b483742ddb7c8529745df9ec62b8',
        'Referer': '{}'.format(referer),
    }
    return headers
    
def get_html(url):
	urls=[]
	#其实这里用的2000不是很合理，更合理的应该是现货区关注者的数量然后除以20
	#这才是真正的数量,但是目前不需要特别强调
	for i in range (1):
		i=i*20
		end_url=url+str(i)
		urls.append(end_url)
	return urls

def get_id(url):
	r=requests.get(url,headers=header(url))
	m=r.json()
	_mess=m['data']
	_ids=[]
	#获取id即url_token
	for i in range(20):
		_Mess=_mess[i]
		_ids.append(_Mess['url_token'])
	return _ids

def get_peo_urls(ids):
	#由于知乎很不厚道的没有对关注者话题的关注量排名所以我取简只提取前三个
	urls2=[]
	for id in ids:
		for i in id:
			url="http://www.zhihu.com/api/v4/members/"+i+"/following-topic-contributions?include=data%5B%2A%5D.topic.introduction&limit=20&offset=0"
			urls2.append(url)
	return urls2
def get_topic(urls2):
	ip={'http':'122.230.156.248'}
	r=requests.get(urls2,headers=header(urls2),proxies=ip)
	print (r.status_code)
	m=r.json()
	_mess2=m['data']
	#这里存在某些用户关注的话题数目少于3，所以干脆直接pass
	#同时在经过测试时，发现这里的每一段数据都是字典
	#通过索引可以直接获取想要的数据
	try:
		for i in range(5):
			_Mess2=_mess2[i]
			return _Mess2['topic']['name']
	except:
		pass
def get_topic_nums(names,_Names):
	for i in names:
		for name in i:
			_Names.append(name)
	_N=Series(_Names)
	return _N.value_counts()
def show_topic_nums(Ser):
	#数据排序 并且分配
	Ser=Ser.sort_values(ascending=False)
	Names=Ser.keys()
	Nums=Ser.values
	
	my_config=pygal.Config()
	#x_label_rotation 用于让x标签斜向下，不重叠，这很重要
	my_config.x_label_rotation=45
	my_config.show_legend=False
	my_config.truncate_label=15
	my_config.show_y_guides=False
	my_config.width=1000

	the_style=Lp('#7FFFD4',base_style=Lk)
	chart=pygal.Bar(my_config,style=the_style,show_legend=False)
	chart.title="The java's Case"
	chart.x_labels=Names[:10]
	chart.add('',Nums[:10])

	

def main():
	begin=datetime.now()
	url="http://www.zhihu.com/api/v4/members/ao-di-66-50/followers?include=data%5B%2A%5D.answer_count%2Carticles_count%2Cgender%2Cfollower_count%2Cis_followed%2Cis_following%2Cbadge%5B%3F%28type%3Dbest_answerer%29%5D.topics&limit=20&offset="
	urls=get_html(url)
	ids=[]
	with ThreadPool(3) as pool:
		ids.extend(pool.map(get_id,urls))
	urls2=get_peo_urls(ids)
	Names=[]  #获取每个页面的话题名称
	_Names=[] #将各个话题名称汇总
	with ThreadPool(3) as pool2:
		Names.append(pool2.map(get_topic,urls2))
	Ser_names=get_topic_nums(Names,_Names)
	end=datetime.now()
	print ("可视化完成花费时间:",(end-begin))
	show_topic_nums(Ser_names)
	
main()
