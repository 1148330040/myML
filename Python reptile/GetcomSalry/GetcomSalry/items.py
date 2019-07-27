# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class GetcomsalryItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    Salarys1 = scrapy.Field() #拉钩
    Salarys2 = scrapy.Field() #猎聘
    Names1 = scrapy.Field()   #拉钩
    Names2 = scrapy.Field()   #猎聘
