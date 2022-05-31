import os
import queue
import shutil
import sys
import time
import requests
import logging
from bs4 import BeautifulSoup


class MainPages:
    def __init__(self):
        self.detailpages = queue.Queue()
        self.main_url = "https://www.blitzer.de/bilder/seite/"
        self.detail_url_prefix = "https://www.blitzer.de"

    def collect_detailpages(self, page_nr):
        self.scrape_page(page_nr)

    def scrape_page(self, page_nr):
        r = requests.get(self.main_url + str(page_nr))
        try:
            if r.status_code != 200:
                raise ValueError("Status Code != 200")
        except ValueError as e:
            print(e)
            print("status code: " + str(r.status_code) + "on page " + page_nr)
            logging.error("status code: " + str(r.status_code) + "on page " + page_nr)
            sys.exit()
        self.parse_html(r.text)

    def parse_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        entries = soup.select(".card-body")
        for entry in entries:
            for link in entry.find_all('a'):
                link = link.get('href')
                if link.startswith("/bilder/"):
                    self.detailpages.put(self.detail_url_prefix + link)

    def all_detail_pages_visited(self):
        return self.detailpages.empty()

    def get_detailpage(self):
        return self.detailpages.get(block=False)

    def get_amount_detailpages(self):
        return self.detailpages.qsize()


class DetailPage:
    def __init__(self):
        self.image_links = queue.Queue()
        # self.download_folder = os.getcwd() + "\\images\\"
        self.download_folder = "F:\\images\\"

    def download_images(self, detailpage):
        print("detailpage: " + detailpage)
        logging.info("detailpage: " + detailpage)
        r = requests.get(detailpage)
        try:
            if r.status_code != 200:
                raise ValueError("Status Code != 200")
        except ValueError as e:
            print(e)
            print("status code: " + str(r.status_code) + "on detailpage " + detailpage)
            logging.error("status code: " + str(r.status_code) + "on detailpage " + detailpage)
            sys.exit()
        self.parse_html(r.text)

    def parse_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        atags = [entry for entry in soup.find_all('a') if entry.get('href').startswith('#')]
        self.extract_image_links(atags)

    def extract_image_links(self, atags):
        for tag in atags:
            link = tag.get('data-file')
            link = link.replace("{size}", "original")
            self.image_links.put(link)
        self.download()

    def download(self):
        print("amount of pictures on this page: " + str(self.image_links.qsize()))
        logging.info("amount of pictures on this page: " + str(self.image_links.qsize()))
        while not self.image_links.empty():
            time.sleep(0.1)  # do not overuse server

            link = self.image_links.get(block=False)
            filename = "_".join(link.split("/")[-4:])
            r = requests.get(link, stream=True)
            if r.status_code == 200:
                r.raw.decode_content = True

                with open(self.download_folder + filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                print('Image sucessfully Downloaded: ', self.download_folder + filename)
                logging.info(f'Image sucessfully Downloaded: {self.download_folder + filename}')
            else:
                print('Image Couldn\'t be retreived')
                logging.warning("Image could not be retrieved")


if __name__ == "__main__":
    logging.basicConfig(filename='scraper.log', encoding='utf-8', level=logging.DEBUG)
    current_mainpage = MainPages()
    detailpage = DetailPage()
    for page_nr in range(143, 563):
        current_mainpage.collect_detailpages(page_nr)
        print(f"abgespeicherte Detailseiten auf Seite {page_nr}: {current_mainpage.get_amount_detailpages()}")
        logging.info(
            f"abgespeicherte Detailseiten auf Seite {page_nr}: {current_mainpage.get_amount_detailpages()}")
        while not current_mainpage.all_detail_pages_visited():
            time.sleep(0.1)
            print(f"currently on page {page_nr}")
            detailpage.download_images(current_mainpage.get_detailpage())
