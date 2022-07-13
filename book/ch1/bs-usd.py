from bs4 import BeautifulSoup
import urllib.request as req

url = 'https://www.cnbc.com/quotes/KRW='
res = req.urlopen(url)

soup = BeautifulSoup(res, 'html.parser')

price = soup.select_one('div.QuoteStrip-lastTimeAndPriceContainer > div.QuoteStrip-lastPriceStripContainer > span.QuoteStrip-lastPrice').string

print('usd/krw = ', price)
