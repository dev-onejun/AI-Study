# This code doesn't work.
import os
from dotenv import load_dotenv
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By

load_dotenv()

USER = os.getenv('KONKUK_USER')
PASS = os.getenv('KONKUK_PASS')

url_login = 'https://kuis.konkuk.ac.kr/index.do'
url_target = 'https://kuis.konkuk.ac.kr/index.do'

options = ChromeOptions()
options.add_argument('-headless')
browser = Chrome(options=options)

browser.get(url_login)
print('로그인 페이지에 접근합니다')

e = browser.find_element(by=By.CSS_SELECTOR, value='#uuid-061281c1-d2ef-62e6-6715-17b7e38efd4e > input')
e.clear()
e.send_keys(USER)
browser.save_screenshot('id.png')
e = browser.find_element(by=By.CSS_SELECTOR, value='#uuid-fc372183-57d5-4b8c-d70d-af8aea0fb0ad > input')
e.clear()
e.send_keys(PASS)
browser.save_screenshot('pw.png')

form = browser.find_element(by=By.CSS_SELECTOR, value='#uuid-03f0f8a2-6aa6-a063-7f6b-afff8310e8be > a')
form.submit()
print('로그인 버튼을 클릭합니다')

#browser.get(url_target)
#print('학사정보시스템으로 이동합니다')

browser.save_screenshot('after_login.png')

