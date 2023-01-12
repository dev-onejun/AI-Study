import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()

apikey = os.getenv('OPENWEATHER_APIKEY')
cities = ['Seoul,KR', 'Tokyo,JP', 'New York,US']

api = 'http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}'

k2c = lambda k: k - 273.15

for name in cities:
    url = api.format(city=name, key=apikey)

    r = requests.get(url)

    data = json.loads(r.text)

    print('+ 도시 =', data['name'])
    print('| 날씨 =', data['weather'][0]['description'])
    print('| 최고 기온 =', k2c(data['main']['temp_max']))
    print('')
