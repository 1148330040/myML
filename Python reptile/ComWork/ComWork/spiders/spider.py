# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request
from bs4 import BeautifulSoup
from ComWork.items import ComworkItem
from ComWork import settings
import requests
import re
import numpy as np
from datetime import datetime
import time         #用来进行限制爬取速度

class spiderSpider(scrapy.Spider):
    name = 'spider'
    start_urls = ['https://www.lagou.com/']
    work_mess_url='https://www.lagou.com/zhaopin/Java/{nums1}/?filterOption={nums2}'
    work_url = 'https://www.lagou.com/jobs/positionAjax.json?needAddtionalResult=false'
    namse=[]
    #这里的url是获取行业分类名称下的具体工作的内容 这里是ajax要用requests.post(单方面的reqeuests库和scrapy没关系)
    header={
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "DNT": "1",
    "Host": "www.lagou.com",
    "Origin": "https://www.lagou.com",
    "Referer": "https://www.lagou.com/jobs/list_",
    "X-Anit-Forge-Code": "0",
    "X-Anit-Forge-Token": None,
    "X-Requested-With": "XMLHttpRequest",  # 请求方式XHR
    }
    #下面的header用于爬取具体的职位信息主要用来爬取工资数目
    def parse(self, response):
        '''
        mess1 = BeautifulSoup(response.text, 'lxml')
        mess2 = mess1.find_all('dl')
        urls = []

        items = ComworkItem()
        for i in range(1, 2):
            block_data = mess2[i]  # soup 获取的数据块
            total_names = block_data.find_all('span')  # 工作分类的名称
            if (total_names[0].text != "产品经理"):
                names = block_data.find_all('a')  # 获取的a标签
                for i in names:
                    items['Names'] = i.get_text()
                    yield items
                    urls.append(i.get('href'))
            else:
                break  # 确保了数据范围是技术方面的工作信息
        print(urls)
        for i in range(3):
            n = 0
            salarys = []
            url = urls[i]
            while n < 1:
                time.sleep(0.5)
                n = n + 1
                mess = requests.get(urls[i] + str(n), cookies=settings.cookies, headers=self.header)
                soup = BeautifulSoup(mess.text, 'lxml')
                Moneys = soup.findAll(attrs={"class": "money"})
                for i in Moneys:
                    salarys.append(i.get_text())
            print(salarys)
            '''
        salarys=["zxczc"]
        yield Request(
            url='https://www.lagou.com/',          # 这里比较麻烦必须保证url每次都不同，不然会出现本身将其认定为同一个url从而只执行一次的问题
            meta={"Money": salarys},    #传递参数变量
            callback=self.get_Moneys
        )
        '''
        url='https://www.lagou.com/jobs/positionAjax.json'
        names=["java后端","java web","java分布式","python","php"]
        post_param = {"first": "false", "pn": "1", "kd": ""}
        for name in names:
            post_param['kd']=name
            yield scrapy.FormRequest( #想使用post类型就得用这个scrapy.FormRequest
                url=url,
                formdata=post_param,  #这里也很关键
                callback=self.get_Moneys,
                headers=self.header,
                cookies=settings.cookies,
                method='post',
                dont_filter=True       #这里屏蔽了scrapy自带的url检重
            )
        '''

    def get_Moneys(self, response):
        #正常情况下的传参如果出现items是不能传递成功的
        item=ComworkItem()
        print (response.meta['Money'])
        arr=[1,2,3,4]
        print (self.get_items(arr))
        '''
        items = ComworkItem()
        salaryz = []
        pattern = r'(\w*[0-9]+)'
        for i in response.meta['Money']:
            salaryz = salaryz + (re.findall(pattern, i))
        print(salaryz)
        salarys = np.array(salaryz).astype('int')
        print(salarys.mean())
        '''
    def get_items(self, response):
        print (response)
        return response[1]

    def get_work_mess(self, response):  # 此函数的作用是获取行业名称的具体分类名称，例如java包括java后端,java分布式等
        return 0

