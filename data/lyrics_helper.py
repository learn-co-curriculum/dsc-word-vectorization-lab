import requests as r
from bs4 import BeautifulSoup

def get_page_content(url):
    result = r.get(url)
    content = result.content
    if result.status_code == 200:
        print("Page retrieved successfully!")
        return content
    else:
        print("Error: Status Code {}".format(result.status_code))








if __name__ == '__main__':
    sample_url = 'https://www.azlyrics.com/lyrics/weeknd/prayforme.html'
    get_page_content(sample_url)
