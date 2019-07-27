# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request
from bs4 import BeautifulSoup
from LiePWork import settings
from LiePWork.items import LiepworkItem
import numpy as np
import requests
import time
class SpiderSpider(scrapy.Spider):
    name = 'spider'
    start_urls = ['https://www.liepin.com/it/']
    url="https://www.liepin.com/zhaopin/?ckid=4e227a6024e4a712&fromSearchBtn=2&init=-1&dqs=150040&degradeFlag=1&key={names}&headckid=4e227a6024e4a712&d_pageSize=40&siTag=1SOHiA4eoigXpy03WSE4GQ~PFyeBDzhrrBRMg16Kch6rw&d_headId=0be816ae44dc576ccd9bf0ac4fb28d77&d_ckId=0be816ae44dc576ccd9bf0ac4fb28d77&d_sfrom=search_unknown&d_curPage={nums1}&curPage={nums2}"
    salarys=[]
    def parse(self, response):
        n=1
        k = 0
        items=LiepworkItem()
        soup=BeautifulSoup(response.text,'lxml')
        mess1 = soup.find_all('dd')
        names=[]
        for i in range(3):
            mess2 = mess1[i]
            mess3 = mess2.find_all('a')
            for i in mess3:
                names.append(i.get_text())

        for j in range(3):
            salary1=[]
            items['Names']=names[j]
            while k<n :
                print ("第",(k+1),"页")
                urlx=self.url.format(names=names[j],nums1=k,nums2=k+1)
                mess=requests.get(urlx,cookies=settings.cookies)
                time.sleep(0.3)
                soup = BeautifulSoup(mess.text, 'lxml')
                mess = soup.findAll(attrs={"class": "text-warning"})
                for i in mess:
                    salary = i.get_text()
                    if salary != "面议":
                        a = salary.split("-")
                        b = a[1].split("万")
                        salary1.append((int(a[0]) + int(b[0])) / 2)
                print (salary1)
                pages_mess = soup.findAll(attrs={"class": "pn"})
                if ( (k+1) < int(pages_mess[0].get('value'))):
                    n=int(pages_mess[0].get('value'))
                    k=k+1
                else:
                    break

            s = np.array(salary1)
            items['Salarys']=np.mean(s)
            yield items
            self.get_salarys(np.mean(s))


#14.3620689655   19.5428571429   18.2178217822
    def get_salarys(self,response): #response可以是各种东西包括(爬取的url和其他变量)
        print (response)








