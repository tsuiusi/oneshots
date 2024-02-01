from bs4 import BeautifulSoup
import urllib3
import requests
from test import download_file

url = "https://business.columbia.edu/heilbrunn/resources/graham-and-doddsville-newsletter"

http = urllib3.PoolManager()

response = http.request("GET", url)

soup = BeautifulSoup(response.data, "html.parser")

links = []

for link in soup.find_all('p'):
    try:
        links.append(link.a.get('href'))
    except:
        pass

year = 2023
sc = -1

for i in links:
    i = "https://business.columbia.edu" + i
    if sc == -1:
        season = 'fall'
    else:
        season = 'spring'

    filename = season + '_' + str(year)
    download_file(i, filename + '.pdf')

    sc *= -1
    if sc == -1:
        year -= 1


