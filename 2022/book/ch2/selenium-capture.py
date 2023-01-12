from selenium import webdriver

url = "http://www.naver.com/"

""" it doesn't work in firefox. i might seem that the problem is the version of it.
options = webdriver.FirefoxOptions()
options.add_argument('-headless')

browser = webdriver.Firefox(options=options)
"""

options = webdriver.ChromeOptions()
options.add_argument('-headless')

browser = webdriver.Chrome(options=options)


browser.get(url)
browser.save_screenshot("Website.png")
browser.quit()

