# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request

class spiderSpider(scrapy.Spider):
    name = 'spider'
    start_urls = ['http://www.baidu.com/']
    work_names_url = 'https://www.lagou.com/'  # 获取职业名称
    work_mess_url = 'https://www.lagou.com/zhaopin/Java/{nums1}/?filterOption={nums2}'
    work_url = 'https://www.lagou.com/jobs/positionAjax.json?needAddtionalResult=false'
    urls = []
    # 这里的url是获取行业分类名称下的具体工作的内容 这里是ajax要用requests.post(单方面的reqeuests库和scrapy没关系)
    # 使用post要和下面的post_param配合使用
    # 下面的header用于爬取具体的职位信息主要用来爬取工资数目
    cookies = {
        "JSESSIONID": "ABAAABAAAGGABCB090F51A04758BF627C5C4146A091E618",
        "_ga": "GA1.2.1916147411.1516780498",
        "_gid": "GA1.2.405028378.1516780498",
        "Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6": "1516780498",
        "user_trace_token": "20180510155458-df9f65bb-00db-11e8-88b4-525400f775ce",
        "LGUID": "20180510155458-df9f6ba5-00db-11e8-88b4-525400f775ce",
        "X_HTTP_TOKEN": "98a7e947b9cfd07b7373a2d849b3789c",
        "index_location_city": "%E5%85%A8%E5%9B%BD",
        "TG-TRACK-CODE": "index_navigation",
        "LGSID": "20180510175810-15b62bef-00ed-11e8-8e1a-525400f775ce",
        "PRE_UTM": "",
        "PRE_HOST": "",
        "PRE_SITE": "https%3A%2F%2Fwww.lagou.com%2F",
        "PRE_LAND": "https%3A%2F%2Fwww.lagou.com%2Fzhaopin%2FJava%2F%3FlabelWords%3Dlabel",
        "_gat": "1",
        "SEARCH_ID": "27bbda4b75b04ff6bbb01d84b48d76c8",
        "Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6": "1516788742",
        "LGRID": "20180510181222-1160a244-00ef-11e8-a947-5254005c3644"
    }
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "DNT": "1",
        "Host": "www.lagou.com",
        "Origin": "https://www.lagou.com",
        "Referer": "https://www.lagou.com/jobs/list_",
        "X-Anit-Forge-Code": "0",
        "X-Anit-Forge-Token": None,
        "X-Requested-With": "XMLHttpRequest"  # 请求方式XHR
    }
    def start_requests(self):
        post_param = {"first": "true", "pn": "1", "kd": "python"}
        yield Request(
            self.work_url,
            headers=self.header,
            method='post',          #默认是get
            meta=post_param,        #想要实现类似于requests.post的方法就要加这个meta
            cookies=self.cookies,
            callback=self.get_mess  #这个callback也不能省，如果是
        )
    def get_mess(self,response):
        mess=response.text
        print (mess)


