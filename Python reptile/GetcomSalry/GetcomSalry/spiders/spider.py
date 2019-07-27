# -*- coding: utf-8 -*-
import scrapy
import requests
from scrapy import Request,FormRequest
from GetcomSalry import settings
import json
from pandas import DataFrame,Series
from bs4 import BeautifulSoup
import time
import numpy as np
import re
from GetcomSalry.items import GetcomsalryItem
class SpiderSpider(scrapy.Spider):

    name = 'spider'
    start_urls1 = 'https://www.lagou.com/'
    start_urls2 = 'https://www.liepin.com/it/'
    lie_url="https://www.liepin.com/zhaopin/?ckid=4e227a6024e4a712&fromSearchBtn=2&init=-1&dqs=150040&degradeFlag=1&key={names}&headckid=4e227a6024e4a712&d_pageSize=40&siTag=1SOHiA4eoigXpy03WSE4GQ~PFyeBDzhrrBRMg16Kch6rw&d_headId=0be816ae44dc576ccd9bf0ac4fb28d77&d_ckId=0be816ae44dc576ccd9bf0ac4fb28d77&d_sfrom=search_unknown&d_curPage={nums1}&curPage={nums2}"
    la_url = 'https://www.lagou.com/jobs/positionAjax.json?needAddtionalResult=false'
    post_param = {"first": "false", "pn": "1", "kd": ""}

    def start_requests(self):
        '''
        yield Request(
            url=self.start_urls1,
            callback=self.get_names_la,
            headers=settings.header
        )
        '''
        yield Request(
            url=self.start_urls2,
            callback=self.get_names_lie,
        )

    def get_names_la(self,response):
        names=[]
        urls = []
        mess1 = BeautifulSoup(response.text, 'lxml')
        '''
        mess2 = mess1.find_all('dl')
        for i in range(1, 2):
            block_data = mess2[i]  # soup 获取的数据块
            total_names = block_data.find_all('span')  # 工作分类的名称
            if (total_names[0].text != "产品经理"):
                total_mess = block_data.find_all('a')  # 获取的a标签
                for i in total_mess:
                    urls.append(i.get('href'))
        print (urls)
        for url in urls:
            print(url)
            time.sleep(0.5)
            mess=requests.get(url,headers=settings.header,cookies=settings.cookies2)
            soup = BeautifulSoup(mess.text, 'lxml')
            content = soup.findAll(attrs={"class": "r_search_a"})
            if len(content)==0:
                content= soup.findAll(attrs={"class": "r_search_con"})
            for i in content:
                names.append(i.get_text())
        # 这是经过上面代码爬取后获得的全部数据
        # 方便进行下面的爬取我先直接将其保存起来
        self.get_pass_la(response=names)
        '''
    # 将获取的行业名称通过post传递数据
    def get_pass_la(self,response):
        items=GetcomsalryItem()
        names=np.array(response)
        names=np.unique(names)
        for name in names:
            self.post_param['kd']=name
            items['Names1']=name
            yield items
            yield FormRequest(
                url=self.la_url,
                formdata=self.post_param,
                callback=self.get_salarys_la,
                cookies=settings.cookies2,
                headers=settings.header,
                method='post'
            )
            time.sleep(0.5)
    #获取网页中的salarys
    def get_salarys_la(self,response):
        items = GetcomsalryItem()
        pattern = r'"salary\"\:\"(\w+.?\w+)\"'
        salarys=re.findall(pattern,response.text)
        items['Salarys1']=self.solve_sal_la(salarys)
        yield items

    # 将数据(nK-nK)转换为数字
    def solve_sal_la(self,response):
        pattern = r'(\w+)k'
        salarys=[]
        for salary in response:
            salarys+=(re.findall(pattern,salary))
        salarys=np.array(salarys).astype('int').mean()
        return salarys

    #获取职位名称

    def get_names_lie(self,response):
        soup = BeautifulSoup(response.text, 'lxml')
        mess1 = soup.find_all('dd')
        names=[]
        for i in range(3):
            mess2 = mess1[i]
            mess3 = mess2.find_all('a')
            for i in mess3:
                names.append(i.get_text())
        self.get_salarys_lie(response=names)

    def get_salarys_lie(self,response):
        items=GetcomsalryItem()
        names=response
        n=0
        k=0
        for name in names:
            items['Names2']=name
            salarys = []        #获取每一中工作的所有工资数目
            # for k in range(3) :这是用作检验所以提取了一部分的值
            while k<n: #这是正常情况下的获取的所有页数的方法
                print ("第",(k+1),"页")
                if (k==0):
                    k2=0
                else:
                    k2=k+1
                urlx = self.lie_url.format(names=name, nums1=k, nums2=k2)
                mess = requests.get(urlx, cookies=settings.cookies1)
                time.sleep(0.3)
                soup = BeautifulSoup(mess.text, 'lxml')
                mess = soup.findAll(attrs={"class": "text-warning"})
                for i in mess:
                    salary = i.get_text()
                    if salary != "面议":
                        a = salary.split("-")
                        b = a[1].split("万")
                        salarys.append((int(a[0]) + int(b[0])) / 2)
                pages_mess = soup.findAll(attrs={"class": "pn"})
                if ((k + 1) < int(pages_mess[0].get('value'))):
                    n = int(pages_mess[0].get('value'))
                    k = k + 1
                else:
                    break
            #通过numpy 求出工资列表的平均数值
            salarys=int(np.array(salarys).mean()*10000)
            items['Salarys2']=salarys
            yield items





