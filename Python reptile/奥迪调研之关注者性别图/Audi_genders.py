#coding:gbk

import requests
import re
from multiprocessing.dummy import Pool as ThreadPool
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from pyecharts import Geo

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


def get_gender(urls):
	m=requests.get(urls,headers=header(urls))
	'''
	#内部是json不能用re进行查找
	pattern=r'\"gender\":\d'
	gender=re.findall(pattern,message)
	'''
	print (m)
	_mess=m.json()
	_getMess=_mess['data']
	gender=[]
	for i in range(20):
		_getMes=_getMess[i]
		print (_getMes)
		gender.append(_getMes['gender'])
	return gender


def get_arr_genders(genders):
	_freNum=[]
	x=[]
	for arr in genders:
		for i in arr:
			x.append(i)
	_freNum=Series(x)
	return _freNum.value_counts()

def get_pic_genders(_freNum):
	#目前获取了频数
	#-1 男性 1女性 0未添加性别
	#饼图
	pets='Man','Woman','No gender added'
	sizes=[]
	for i in _freNum:
		sizes.append(i)
	size=[sizes[0],sizes[1],sizes[2]]
	proportion=(0,0,0)
	plt.pie(size,explode=proportion,labels=pets,autopct='%0.1f%%',
	shadow=False,startangle=60) #初始角度 startangle
	plt.axis('equal') #圆形
	
	

	

def main():
	url="http://www.zhihu.com/api/v4/members/ao-di-66-50/followers?include=data%5B%2A%5D.answer_count%2Carticles_count%2Cgender%2Cfollower_count%2Cis_followed%2Cis_following%2Cbadge%5B%3F%28type%3Dbest_answerer%29%5D.topics&limit=20&offset="
	p=get_html(url)
	#多线程这里遇到一个问题 就是每个页面产生的数组很难合并到一个数组里 
	genders=[]
	with ThreadPool(100) as pool:
		genders.extend(pool.map(get_gender,p))
	_freNum=get_arr_genders(genders)
	
	get_pic_genders(_freNum)
	'''
	#不行，采用这种方法虽然可以直接将每个数组聚合到一个数组里，但是问题是太慢了
	genders=[]
	begin=datetime.datetime.now()
	for url in p:
		genders.extend(get_gender(url))
	end=datetime.datetime.now()
	print (genders)
	print (end-begin)
	'''
main()
	
	
	
	
	

	
	
	
	

