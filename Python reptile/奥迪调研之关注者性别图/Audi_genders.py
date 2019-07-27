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
	#��ʵ�����õ�2000���Ǻܺ����������Ӧ�����ֻ�����ע�ߵ�����Ȼ�����20
	#���������������,����Ŀǰ����Ҫ�ر�ǿ��
	for i in range (1):
		i=i*20
		end_url=url+str(i)
		urls.append(end_url)
	return urls


def get_gender(urls):
	m=requests.get(urls,headers=header(urls))
	'''
	#�ڲ���json������re���в���
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
	#Ŀǰ��ȡ��Ƶ��
	#-1 ���� 1Ů�� 0δ����Ա�
	#��ͼ
	pets='Man','Woman','No gender added'
	sizes=[]
	for i in _freNum:
		sizes.append(i)
	size=[sizes[0],sizes[1],sizes[2]]
	proportion=(0,0,0)
	plt.pie(size,explode=proportion,labels=pets,autopct='%0.1f%%',
	shadow=False,startangle=60) #��ʼ�Ƕ� startangle
	plt.axis('equal') #Բ��
	
	

	

def main():
	url="http://www.zhihu.com/api/v4/members/ao-di-66-50/followers?include=data%5B%2A%5D.answer_count%2Carticles_count%2Cgender%2Cfollower_count%2Cis_followed%2Cis_following%2Cbadge%5B%3F%28type%3Dbest_answerer%29%5D.topics&limit=20&offset="
	p=get_html(url)
	#���߳���������һ������ ����ÿ��ҳ�������������Ѻϲ���һ�������� 
	genders=[]
	with ThreadPool(100) as pool:
		genders.extend(pool.map(get_gender,p))
	_freNum=get_arr_genders(genders)
	
	get_pic_genders(_freNum)
	'''
	#���У��������ַ�����Ȼ����ֱ�ӽ�ÿ������ۺϵ�һ�����������������̫����
	genders=[]
	begin=datetime.datetime.now()
	for url in p:
		genders.extend(get_gender(url))
	end=datetime.datetime.now()
	print (genders)
	print (end-begin)
	'''
main()
	
	
	
	
	

	
	
	
	

