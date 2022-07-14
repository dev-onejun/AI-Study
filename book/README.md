# 파이썬을 이용한 머신러닝, 딥러닝 실전 개발 입문(개정판)
* 쿠지라 히코우즈쿠에 지음, 윤인성 옮김, 위키북스 출판
* [Github Link](https://github.com/wikibook/pyml-rev)
* 다음의 컴퓨터 환경에서 소스코드를 실행했습니다.
	- CPU		: Intel Pentium Gold G6405
	- OS		: Ubuntu Server 22.04 LTS
	- Python	: Version 3.8, venv

## 1장, 크롤링과 스크레이핑
### urllib.request
* urllib.request.urlretrieve()
* urllib.request.urlopen()
* QueryString을 이용해 데이터 보내기

### BeautifulSoup
``` python
soup = BeautifulSoup(html, 'html.parser')

title = soup.find(id='title')
links = soup.find_all('a')
for a in links:
    href = a.attrs['href']
    text = a.string
```

* CSS Selector
``` python
soup.select('#ve-list < li.black')
```

* 정규조합식

* 페이지에 링크된 자료까지 모두 받기(재귀 이용)
    - [source code](./ch1/cr-getall.py)

## 2장, 고급 스크레이핑
### requests
* requests.session()을 이용한 사이트 로그인
* requests.get()
* requests.post()
* requests.put()
* requests.delete()

### Selenium
* [Selenium with Python](http://selenium-python.readthedocs.io/index.html)
* [SeleniumHQ Documentation](http://docs.seleniumhq.org/docs)
	- [Selenium Github](https://github.com/SeleniumHQ/seleniumhq.github.io)

### 정기적인 크롤링
* cron (macOS and Linux)
    - `crontab -e`
    - It can alert using email
* Task Scheduler (Windows)

