# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json


class AudimessPipeline(object):
    """生成json文件!"""

    def open_spider(self, spider):
        self.f = open('./data.json', 'a')

    def process_item(self, item, spider):
        content = json.dumps(dict(item), ensure_ascii=False) + '\n'
        #self.f.write(content.encode("utf-8"))  # python2
        self.f.write(content)  #python3

    def close_spider(self, spider):
        print("{}:爬虫数据处理完毕!".format(spider.name))
        self.f.close()
