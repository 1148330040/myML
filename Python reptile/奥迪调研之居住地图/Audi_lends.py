#coding:gbk
import requests
from lxml import html
from multiprocessing.dummy import Pool as ThreadPool
from pandas import Series
import pygal
from pygal.style import LightColorizedStyle as Lk,LightenStyle as Lp
from datetime import datetime
from bs4 import BeautifulSoup
from pyecharts import Map
import time
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
	#其实这里用的100不是很合理，更合理的应该是现货区关注者的数量然后除以20
	for i in range (10):
		i=i*20
		end_url=url+str(i)
		urls.append(end_url)
	return urls

def get_id(urls):
	r=requests.get(urls,headers=header(urls),timeout=5)
	m=r.json()
	ids=[]
	mess=m['data']
	for i in range(20):
		Mess=mess[i]
		ids.append(Mess['url_token'])
	return ids
def get_Ids(ids):
	Ids=[]
	for i in ids:
		for x in i:
			Ids.append(x)
	return Ids
def link_id_url(Ids):
	id_urls=[]
	for i in Ids:
		url="https://www.zhihu.com/api/v4/members/"+i+"?include=locations%2Cemployments%2Cgender%2Ceducations%2Cbusiness%2Cvoteup_count%2Cthanked_Count%2Cfollower_count%2Cfollowing_count%2Ccover_url%2Cfollowing_topic_count%2Cfollowing_question_count%2Cfollowing_favlists_count%2Cfollowing_columns_count%2Cavatar_hue%2Canswer_count%2Carticles_count%2Cpins_count%2Cquestion_count%2Ccolumns_count%2Ccommercial_question_count%2Cfavorite_count%2Cfavorited_count%2Clogs_count%2Cmarked_answers_count%2Cmarked_answers_text%2Cmessage_thread_token%2Caccount_status%2Cis_active%2Cis_bind_phone%2Cis_force_renamed%2Cis_bind_sina%2Cis_privacy_protected%2Csina_weibo_url%2Csina_weibo_name%2Cshow_sina_weibo%2Cis_blocking%2Cis_blocked%2Cis_following%2Cis_followed%2Cmutual_followees_count%2Cvote_to_count%2Cvote_from_count%2Cthank_to_count%2Cthank_from_count%2Cthanked_count%2Cdescription%2Chosted_live_count%2Cparticipated_live_count%2Callow_message%2Cindustry_category%2Corg_name%2Corg_homepage%2Cbadge%5B%3F(type%3Dbest_answerer)%5D.topics"
		id_urls.append(url)
	return id_urls


def get_lends(urls):
	r=requests.get(urls,headers=header(urls),timeout=5)
	m=r.json()
	lends=[] 
	#因为部分用户的居住地并没有填写所以直接略过
	me=m['locations']
	try:
		return me[0]['name']
	except:
		pass
def clean_data(lends):
	m=[]
	for a in lends:
		for x in a:
			if (x != None):
				m.append(x)
	r=Series(m).value_counts()

	return r.sort_values()
	
def show_lends(r):
	names=r.keys()
	nums=r.values
	map = Map("主要来源城市", width=1200, height=600,title_color="#fff", title_pos="center")
	map.add("", attr, value, maptype='广东', is_visualmap=True, visual_text_color='#000')
	map.show_config()
	map.render()
	
def main():
	time1=datetime.now()
	url="http://www.zhihu.com/api/v4/members/ao-di-66-50/followers?include=data%5B%2A%5D.answer_count%2Carticles_count%2Cgender%2Cfollower_count%2Cis_followed%2Cis_following%2Cbadge%5B%3F%28type%3Dbest_answerer%29%5D.topics&limit=20&offset="
	urls=get_html(url)
	ids=[] 					#单页面汇总
	Ids=[] 					#总汇总
	id_urls=[] 				#获取用户详细信息，包括隐藏信息
	lends=[]				#获取居住地所在
	with ThreadPool(3) as pool:
		ids.extend(pool.map(get_id,urls))
	Ids=get_Ids(ids) #id汇总
	id_urls=link_id_url(Ids)
	with ThreadPool(3) as pool:
		lends.append(pool.map(get_lends,id_urls))
	r=clean_data(lends)
	show_lends(r)
	time2=datetime.now()
	print (time2-time1)
	
main()
