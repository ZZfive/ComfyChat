import base64
from googlesearch import search
import trafilatura
import datetime
import pprint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import feedparser
from urllib import parse
import re


class WebCrawler:
    def __init__(self):
        self.scraper = GoogleNewsScraper()
        self.url_scraper = URLScraper()
        
    def crawl_query(self, query, num_pages, start_n_days_ago, end_n_days_ago):
        url_list = self.scraper.scrape_query(query, num_pages, start_n_days_ago, end_n_days_ago)  # 先获取一组url
        print("URLS TO BE SCRAPED:")
        pprint.pprint(url_list)
        
        scraped_urls = self.url_scraper.scrape_url_list(url_list)  # 解析urls
        if len(scraped_urls) == 0:
            print("No URLs were scraped")
            return []
        
        print("SCRAPED URLS:")
        pprint.pprint(scraped_urls)


class GoogleScraper:
    
    FILTER_OUT_LIST_SITES = ["instagram.com/","youtube.com/", "spotify.com/", "music.apple.com/", "soundcloud.com/", ".gov/"]  # 需要过滤的网站
    
    def __init__(self):
        return
    
    def scrape_query(self, query, num_pages_to_scrape, from_n_days_ago=0, to_n_days_ago=0):
        if not 0 <= to_n_days_ago <= from_n_days_ago:
            raise ValueError(f"from_n_days_ago: {from_n_days_ago} must be >= to_n_days_ago: {to_n_days_ago} must be >= 0")
        if not from_n_days_ago == to_n_days_ago == 0:
            curr_time = datetime.datetime.now()
            start_date, end_date = curr_time - datetime.timedelta(days=from_n_days_ago), curr_time - datetime.timedelta(days=to_n_days_ago)
            start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

        print("SCRAPING FROM: " + start_date + " TO: " + end_date)
        full_query = query.strip(' ') + ' ' + ' '.join([f"-site:{site}" for site in self.FILTER_OUT_LIST_SITES]) + f" after:{start_date} before:{end_date}"
        url_list = search(full_query, num=10, stop=num_pages_to_scrape, pause=2)
        return list(url_list)


class GoogleNewsScraper:
    
    GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q="
    
    _ENCODED_URL_PREFIX = "https://news.google.com/rss/articles/"
    _ENCODED_URL_RE = re.compile(fr"^{re.escape(_ENCODED_URL_PREFIX)}(?P<encoded_url>[^?]+)")
    _DECODED_URL_RE = re.compile(rb'^\x08\x13".+?(?P<primary_url>http[^\xd2]+)\xd2\x01')

    def __init__(self):
        pass
    
    def scrape_query(self, query, num_pages_to_scrape, from_n_days_ago=0, to_n_days_ago=0):
        if not 0 <= to_n_days_ago <= from_n_days_ago:
            raise ValueError(f"from_n_days_ago: {from_n_days_ago} must be >= to_n_days_ago: {to_n_days_ago} must be >= 0")
        if not from_n_days_ago == to_n_days_ago == 0:
            curr_time = datetime.datetime.now()
            start_date, end_date = curr_time - datetime.timedelta(days=from_n_days_ago), curr_time - datetime.timedelta(days=to_n_days_ago)
            start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        
        print("SCRAPING FROM: " + start_date + " TO: " + end_date)
        full_query = query.strip(' ') + f" after:{start_date} before:{end_date}"
        url_query = parse.quote_plus(full_query)
                
        full_url = f'https://news.google.com/rss/search?q={url_query}&ceid=US:en&hl=en-US&gl=US'
        feed = feedparser.parse(full_url)

        entries = feed.entries[:num_pages_to_scrape]

        url_list = []
        for entry in entries:
            real_url = self.decode_google_news_url(entry.link)
            url_list.append(real_url)
            
        return url_list
    
    def decode_google_news_url(self, url: str) -> str:
        match = self._ENCODED_URL_RE.match(url)
        encoded_text = match.groupdict()["encoded_url"] 
        encoded_text += "===" 
        decoded_text = base64.urlsafe_b64decode(encoded_text)
        match = self._DECODED_URL_RE.match(decoded_text)
        primary_url = match.groupdict()["primary_url"] 
        primary_url = primary_url.decode()
        return primary_url


class URLScraper:
    
    LINE_REMOVAL_WORD_COUNT_THRESHOLD = 10
    
    def __init__(self):
        return
    
    def scrape_url_list(self, url_list):
        (url_contents) = []
        for url in url_list:
            print(f"Scraping {url}:\n")
            downloaded = trafilatura.fetch_url(url)  # 使用 Trafilatura 库下载和提取每个URL的内容
            result = trafilatura.extract(downloaded, target_language="EN", include_tables=False, include_comments=False, favor_precision=True, deduplicate=True, only_with_metadata=True)
            if result is None:
                continue
            line_list = result.split("\n")
            removed_short_lines_list = [line for line in line_list if line.count(' ') + 1 > self.LINE_REMOVAL_WORD_COUNT_THRESHOLD]
            (url_contents).append((url, ('\n'.join(removed_short_lines_list))))
        return(url_contents)


if __name__ == '__main__':
    spider = WebCrawler()
    
    query = 'ComfyUI介绍'
    num_pages = 5
    start_n_days_ago = 4
    end_n_days_ago = 1
    spider.crawl_query(query, num_pages, start_n_days_ago, end_n_days_ago)