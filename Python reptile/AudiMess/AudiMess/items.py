# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy

from scrapy import Item, Field

class AudimessItem(scrapy.Item):
    url_token = scrapy.Field()
    gender = scrapy.Field()
    educations = scrapy.Field()
    business = scrapy.Field()
    locations = scrapy.Field()

