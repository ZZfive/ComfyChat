import os
import datetime
import pprint
from typing import List
from urllib.parse import quote, unquote

from googlesearch import search
import trafilatura

save_dir = r'D:\git_github\ComfyChat\data\web_crawl_txt'


class WebCrawler:
    def __init__(self):
        self.scraper = GoogleScraper()
        self.url_scraper = URLScraper()
        
    def crawl_topic(self, topic: str, num_pages: int, start_n_days_ago: int, end_n_days_ago: int) -> None:
        url_list = self.scraper.scrape_topic(topic, num_pages, start_n_days_ago, end_n_days_ago)  # 先获取一组url
        print("URLS TO BE SCRAPED:")
        pprint.pprint(url_list)
        
        scraped_urls = self.url_scraper.scrape_url_list(url_list)  # 解析urls
        if len(scraped_urls) == 0:
            print("No URLs were scraped")
            return []
        print(len(scraped_urls))
        assert len(url_list) == len(scraped_urls)
        for i in range(len(url_list)):
                if scraped_urls[i]:
                    save_path = os.path.join(save_dir, f"{quote(url_list[i], safe='')}.txt")  # 将url编码使其可以保存，可以使用unquote(encoded_url)转为原url
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(scraped_urls[i])


class GoogleScraper:
    
    FILTER_OUT_LIST_SITES = ["instagram.com/","youtube.com/", "spotify.com/", "music.apple.com/", "soundcloud.com/", ".gov/"]  # 需要过滤的网站
    
    def __init__(self):
        return
    
    def scrape_topic(self, topic: str, num_pages_to_scrape: int, from_n_days_ago: int=0, to_n_days_ago: int=0) -> List[str]:
        if not 0 <= to_n_days_ago <= from_n_days_ago:
            raise ValueError(f"from_n_days_ago: {from_n_days_ago} must be >= to_n_days_ago: {to_n_days_ago} must be >= 0")
        if not from_n_days_ago == to_n_days_ago == 0:
            curr_time = datetime.datetime.now()
            start_date, end_date = curr_time - datetime.timedelta(days=from_n_days_ago), curr_time - datetime.timedelta(days=to_n_days_ago)
            start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

        print("SCRAPING FROM: " + start_date + " TO: " + end_date)
        full_query = topic.strip(' ') + ' ' + ' '.join([f"-site:{site}" for site in self.FILTER_OUT_LIST_SITES]) + f" after:{start_date} before:{end_date}"
        url_list = search(full_query, num=10, stop=num_pages_to_scrape, pause=2)
        return list(url_list)


class URLScraper:
    
    LINE_REMOVAL_WORD_COUNT_THRESHOLD = 10
    
    def __init__(self):
        return
    
    def scrape_url_list(self, url_list: List[str]) -> List[str]:
        url_contents = []
        for url in url_list:
            print(f"Scraping {url}:\n")
            downloaded = trafilatura.fetch_url(url)  # 使用 Trafilatura 库下载和提取每个URL的内容
            result = trafilatura.extract(downloaded)
            if result is None:
                url_contents.append('')
            else:        
                url_contents.append(result)
        return url_contents
    

def web_crawl(topic_urls: str | List[str], num_pages: int = 20, start_n_days_ago: int = 10, end_n_days_ago: int = 0) -> None:
    if type(topic_urls) == str:
        print(f'传入的{topic_urls}待搜索主题，通过google搜索urls收集数据')
        spider = WebCrawler()
        spider.crawl_topic(topic_urls, num_pages, start_n_days_ago, end_n_days_ago)
    else:
        print(f'传入的{topic_urls}具体urls，直接通过trafilatura收集数据')
        for url in topic_urls:
            downloaded = trafilatura.fetch_url(url)
            result = trafilatura.extract(downloaded)
            if result:
                save_path = os.path.join(save_dir, f"{quote(url, safe='')}.txt")  # 将url编码使其可以保存，可以使用unquote(encoded_url)转为原url
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(result)


if __name__ == '__main__':
    # spider = WebCrawler()
    
    # topic = 'ComfyUI介绍'
    # num_pages = 5
    # start_n_days_ago = 4
    # end_n_days_ago = 1
    # spider.crawl_topic(topic, num_pages, start_n_days_ago, end_n_days_ago)

    # a = 'https%3A%2F%2Fwww.songshuhezi.com%2Fcomfy_ui%2Findex.html.txt'
    # a = a[:-4]
    # print(unquote(a))

    urls = ['https://zhuanlan.zhihu.com/p/662041596']
    web_crawl(urls)