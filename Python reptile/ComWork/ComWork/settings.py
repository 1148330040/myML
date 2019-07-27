# -*- coding: utf-8 -*-

# Scrapy settings for ComWork project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'ComWork'

SPIDER_MODULES = ['ComWork.spiders']
NEWSPIDER_MODULE = 'ComWork.spiders'


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'ComWork (+http://www.yourdomain.com)'


# Obey robots.txt rules
ROBOTSTXT_OBEY = True
cookies = {
    "JSESSIONID": "ABAAABAAAGGABCB090F51A04758BF627C5C4146A091E618",
    "_ga": "GA1.2.1916147411.1516780498",
    "_gid": "GA1.2.405028378.1516780498",
    "Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6": "1516780498",
    "user_trace_token": "20180516155458-df9f65bb-00db-11e8-88b4-525400f775ce",
    "LGUID": "20180516155458-df9f6ba5-00db-11e8-88b4-525400f775ce",
    "X_HTTP_TOKEN": "98a7e947b9cfd07b7373a2d849b3789c",
    "index_location_city": "%E5%85%A8%E5%9B%BD",
    "TG-TRACK-CODE": "index_navigation",
    "LGSID": "20180516175810-15b62bef-00ed-11e8-8e1a-525400f775ce",
    "PRE_UTM": "",
    "PRE_HOST": "",
    "PRE_SITE": "https%3A%2F%2Fwww.lagou.com%2F",
    "PRE_LAND": "https%3A%2F%2Fwww.lagou.com%2Fzhaopin%2FJava%2F%3FlabelWords%3Dlabel",
    "_gat": "1",
    "SEARCH_ID": "27bbda4b75b04ff6bbb01d84b48d76c8",
    "Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6": "1516788742",
    "LGRID": "20180516181222-1160a244-00ef-11e8-a947-5254005c3644"

}

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 90

# Configure a delay for requests for the same website (default: 0)
# See https://doc.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
#DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "DNT": "1",
    "Host": "www.lagou.com",
    "Origin": "https://www.lagou.com",
    "Referer": "https://www.lagou.com/jobs/list_",
    "X-Anit-Forge-Code": "0",
    "X-Anit-Forge-Token": None,
    "X-Requested-With": "XMLHttpRequest",  # 请求方式XHR
    'Accept-Language': 'en',
}

# Enable or disable spider middlewares
# See https://doc.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'ComWork.middlewares.ComworkSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    'ComWork.middlewares.ComworkDownloaderMiddleware': 543,
#}

# Enable or disable extensions
# See https://doc.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://doc.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    'ComWork.pipelines.ComworkPipeline': 300,
#}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://doc.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
