# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from pandas import DataFrame
import csv

class LiepworkPipeline(object):
    def open_spider(self):
        self.f=open('./data.csv','a')

    def process_item(self, item, spider):
        Salarys=item['Salarys']
        s={'Salarys':Salarys}

        data=DataFrame(s,index=item['Names'])

        mess = csv.writer(data)
        self.f.write(mess)
    def close_spider(self):
        self.f.close()
