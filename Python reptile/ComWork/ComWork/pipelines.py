# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pandas
from pandas import DataFrame
import csv
class ComworkPipeline(object):

    def open_spider(self, spider):
        self.f = open('./data.csv', 'a')

    def process_item(self, item, spider):
        Moneys = item['Moneys']

        data = {'Moneys': Moneys}

        datas = DataFrame(data, index=item['Names'])
        mess = csv.writer(datas)
        self.f.write(mess)

    def close_spider(self, spider):
        self.f.close()
