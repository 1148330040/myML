# -*- coding: utf-8 -*-

# Scrapy settings for LiePWork project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'LiePWork'

SPIDER_MODULES = ['LiePWork.spiders']
NEWSPIDER_MODULE = 'LiePWork.spiders'

cookies={
    'BAIDUID':'25C1636E10FD6E5F88C888371F20A1D8FG=1',
    'BDORZ':'B490B5EBF6F3CD402E515D22BCDA1598',
    'BDSFRCVID':'ADtsJeC6243ka3T70x35tW9ryKII0PnTH6aoPrDKYXUdBygarSBoEG0Pqx8g0KubKA',
    'BDUSS':'UUxTm1FczNLMnhhcHZONkpzMTRPVmtERXg4Y1J3TWZ1V2xxQ0V2d1dLd2ZKfnhaSVFBQUFBJCQAAAAAAAAAAAEAAACdg3C0eWNmenhjdmcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB-a1FkfmtRZW',
    'BIDUPSID':'C93D128CCC4A5811A839C6FD25865232',
    'H_BDCLCKID_SF':'tbPHoK-5JCvbfP0kh-rqK-FHqxby26nZL6neaJ5n0-nnhIT93j-bDbvQLJjbaCrf5T7e-Ro6bb-KS-tRy6CKj5O-jNAHqTnfb5kXWnTH-nT_KROvhjRNQb0yyxomtjDJtK3KoM-5tPP5Mtb5eJ5ByUFPDb3XLUkqKCOTQqbcyxnlehowbPQK0UDeQttjQnoOfIkja-5tWhDVsJ7TyURIbf47yMJHQTT2-DA_oKIbfC5P',
    'H_PS_PSSID':'1424_21109_26350',
    'PSINO':'2',
    'PSTM':'1512136573',
}
# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'LiePWork (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

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
   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
   'Accept-Language': 'en',
   'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
   'Referer':'https://www.liepin.com/it/'
}

# Enable or disable spider middlewares
# See https://doc.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'LiePWork.middlewares.LiepworkSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    'LiePWork.middlewares.LiepworkDownloaderMiddleware': 543,
#}

# Enable or disable extensions
# See https://doc.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://doc.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    'LiePWork.pipelines.LiepworkPipeline': 300,
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
