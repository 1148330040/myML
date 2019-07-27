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
	#��ʵ�����õ�2000���Ǻܺ����������Ӧ�����ֻ�����ע�ߵ�����Ȼ�����20
	#���������������,����Ŀǰ����Ҫ�ر�ǿ��
	for i in range (2000):
		i=i*20
		end_url=url+str(i)
		urls.append(end_url)
	return urls

def get_id(urls):
	r=requests.get(urls,headers=header(urls))
	m=r.json()
	#��ȡ��ע�ߵ���֤��Ϣ���ע���ǵ�����������������ɸѡ��ע��������50��
	_data=m['data']
	_person=[]
	#20��������Ѿ���ȷ�����Ҷ��ǣ�����ֱ��ʹ�þ�����
	for i in range(20):
		_datas=_data[i]
		_person.append(_datas['headline'])
	return _person

def get_work(_person):
	_endpers=[]
	for x in _person:
		for y in x:
			_endpers.append(y)
	_endPers=Series(_endpers)
	return _endPers.value_counts()
	
def show_pic(_endPers):
	_endPers=_endPers.sort_values(ascending=False)
	Nums=_endPers.values
	Names=_endPers.keys()
	
	#�Ա�������һЩ����
	my_config=pygal.Config()
	#x_label_rotation ������x��ǩб���£����ص��������Ҫ
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
	chart.render_to_file('Audi_works.svg')



def main():
	time1=datetime.now()
	url="http://www.zhihu.com/api/v4/members/ao-di-66-50/followers?include=data%5B%2A%5D.answer_count%2Carticles_count%2Cgender%2Cfollower_count%2Cis_followed%2Cis_following%2Cbadge%5B%3F%28type%3Dbest_answerer%29%5D.topics&limit=20&offset="
	urls=get_html(url)
	_person=[]
	with ThreadPool(2) as pool:
		_person.extend(pool.map(get_id,urls))
	series=get_work(_person)
	show_pic(series)
	time2=datetime.now()
	print (time2-time1)

	
main()
