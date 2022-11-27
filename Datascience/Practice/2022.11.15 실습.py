from selenium import webdriver
from selenium.webdriver.common.by import By

from db_conn import *

def crawl_naver_top_ranked_movie_list_one_page(conn, cur, driver, page):
    url = "https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=pnt" + "&page=" + str(page)

    driver.get(url)

    movie_list_body_xpath = '/html/body/div/div[4]/div/div/div/div/div[1]/table/tbody'
    movie_list_element = driver.find_element(By.XPATH, movie_list_body_xpath)
    movies = movie_list_element.find_elements(By.TAG_NAME, 'tr')

    insert_sql = """
        insert into naver_top_ranked_movie_list(m_id, m_rank, url, title, point)
            values(%s, %s, %s, %s, %s);
    """

    rows = []
    for movie in movies:
        rank_element = movie.find_element(By.CLASS_NAME, 'ac')
        m_rank = int(rank_element.find_element(By.XPATH, 'img').get_attribute('alt'))
        print('m_rank=', m_rank)

        title_element = movie.find_element(By.CLASS_NAME, 'title')
        title = title_element.find_element(By.XPATH, 'div/a').text
        url = title_element.find_element(By.XPATH, 'div/a').get_attribute('href')
        m_id = int(url.split('=')[-1])
        print('title=', title)
        print('url=', url)
        print('m_id=', m_id)

        point = float(movie.find_element(By.CLASS_NAME, 'point').text)
        print('point=', point)

        row = (m_id, m_rank, url, title, point)
        rows.append(row)

        print('\n')

    cur.executemany(insert_sql, rows)
    conn.commit()

def crawl_naver_top_ranked_movie_list():
    conn, cur = open_db()

    chromedriver_file = './chromedriver'
    #driver = webdriver.Chrome(chromedriver_file)
    driver = webdriver.Chrome()

    # delete table
    truncate_sql = '''truncate table naver_top_ranked_movie_list;'''
    cur.execute(truncate_sql)
    conn.commit()

    import time
    for page in range(1, 41):
        print('page number=', page)
        crawl_naver_top_ranked_movie_list_one_page(conn, cur, driver, page)
        # sleep 2 seconds for each page
        time.sleep(2)

    close_db(conn, cur)
    driver.quit()

if __name__=='__main__':
    crawl_naver_top_ranked_movie_list()
