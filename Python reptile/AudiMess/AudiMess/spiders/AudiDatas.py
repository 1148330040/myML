# -*- coding: UTF-8 -*-
import scrapy
from scrapy import Request,spider
from AudiMess.items import AudimessItem
import json
from datetime import datetime

class AudidatasSpider(scrapy.Spider):
    name = 'AudiDatas'
    start_urls = ['http://www.zhihu.com/']
    #虽然是直接把url给贴上去,没有拆分来的详细,但是胜在简单,便捷啊！
    test_url='https://www.zhihu.com/api/v4/members/ao-di-66-50/followers?include=data%5B*%5D.answer_count%2Carticles_count%2Cgender%2Cfollower_count%2Cis_followed%2Cis_following%2Cbadge%5B%3F(type%3Dbest_answerer)%5D.topics&offset={num}&limit=20'
    info_url='https://www.zhihu.com/api/v4/members/{name}?include=locations%2Cemployments%2Cgender%2Ceducations%2Cbusiness%2Cvoteup_count%2Cthanked_Count%2Cfollower_count%2Cfollowing_count%2Ccover_url%2Cfollowing_topic_count%2Cfollowing_question_count' \
             '%2Cfollowing_favlists_count%2Cfollowing_columns_count%2Cavatar_hue%2Canswer_count%2Carticles_count%2Cpins_count%2Cquestion_count%2Ccolumns_count%2Ccommercial_question_count%2Cfavorite_count%2Cfavorited_count%2Clogs_count%2Cmarked_answers_count' \
             '%2Cmarked_answers_text%2Cmessage_thread_token%2Caccount_status%2Cis_active%2Cis_bind_phone%2Cis_force_renamed%2Cis_bind_sina%2Cis_privacy_protected%2Csina_weibo_url%2Csina_weibo_name%2Cshow_sina_weibo%2Cis_blocking%2Cis_blocked%2Cis_following' \
             '%2Cis_followed%2Cmutual_followees_count%2Cvote_to_count%2Cvote_from_count%2Cthank_to_count%2Cthank_from_count%2Cthanked_count%2Cdescription%2Chosted_live_count%2Cparticipated_live_count%2Callow_message%2Cindustry_category%2Corg_name%2Corg_homepage' \
             '%2Cbadge%5B%3F(type%3Dbest_answerer)%5D.topics'
    def start_requests(self):
        start_time=datetime.now()
        for i in range(10):
            print (i,"zxczxczxc")
            print (self.test_url.format(num=int((i+1)*20)))
            yield Request(
                self.test_url.format(num=int((i+1)*20)),
                self.parse_follows
            )
        end_time=datetime.now()
        print ("项目花费时间为： ",(end_time-start_time))

    def parse_follows(self, response):
        print ("hello a a a a a ")
        result = json.loads(response.text)
        item = AudimessItem()
        mess=result['data']
        for i in range(20):
            Datas=mess[i]
            item['url_token']=Datas['url_token']
            yield Request(
                self.info_url.format(name=item['url_token']),
                self.get_mess
          )


    def get_mess(self,response):
        result=json.loads(response.text)
        item=AudimessItem()
        try:
            item['gender']=result['gender']
            item['business']=result['business']['name']

        except:
            pass
        try:
            lo = result['locations']
            item['locations'] = lo[0]['name']
        except:
            pass
        try:
            ed = result['educations']
            item['educations'] = ed[0]['name']
        except:
            pass
        yield item













