import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt

class Car:
    brand = ''
    model = ''
    generations = []
    
    def __init__(self, *args):
        self.brand, self.model, self.generations = args

    def __str__(self):
        return f'{self.brand} {self.model}'.upper()

class OlxFilter:
    base = 'https://www.olx.pt/carros-motos-e-barcos/carros/'
    brand = 'renault'
    search_params = {
        'page':'1', 
        'search%5Bfilter_enum_modelo%5D%5B0%5D': '',
        'search%5Bfilter_float_price%3Afrom%5D':'1000', # exclude outliers
        'search%5Bfilter_float_quilometros%3Ato%5D':'250000', # exclude outliers
        'search%5Bfilter_float_price%3Ato%5D': '',
        'search%5Bfilter_float_year%3Afrom%5D': '',
        'search%5Bfilter_float_year%3Ato%5D': '',
        'search%5Bfilter_enum_combustivel%5D%5B0%5D': '',
    }

    def set_search_param(self, key, value):
        for k in self.search_params:
            if key in k:
                self.search_params[k] = str(value)
    
    def set_year_range(self, start, end):
        self.set_search_param('year%3Afrom', start)
        self.set_search_param('year%3Ato', end)

    def __init__(self, car):
        self.car = car
        self.set_search_param('model', car.model)


    def get_url(self):
        query = [f'{key}={value}' for key, value in self.search_params.items() if value]
        return f'{self.base}/{self.car.brand}/?{"&".join(query)}'


def collect_prices_from_page(filter):
    page = requests.get(filter.get_url())

    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find("table", id="offers_table")
    prices = []
    if results:
        for p in results.find_all('p', class_='price'):
            price = float(p.find('strong').text.replace('.','').replace(',','.')[:-2])
            prices.append(price/1000)
    return np.array(prices)

def collect_prices(filter):
    prices = []
    for i in range(1,10):
        filter.page = str(i)
        prices += collect_prices_from_page(filter)
    return prices

def price_by_year(filter, years):
    prices = []
    for i in years:
        filter.set_year_range(i, i+1)
        p = collect_prices_from_page(filter)
        prices.append([np.mean(p), np.std(p), len(p)] )
    return prices 

#collect_prices()

def plot(x,y, title=''):
    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Year")
    plt.ylabel("Price in Thousand Euros")
    
    plt.legend(['mean','std', 'num of samples'])
    plt.title(title)

cars = [
    #Car('peugeot', '308', [(2007,2013), (2013, 2021)]),
    Car('peugeot', '3008', [(2008,2016), (2016,2021)]),
    Car('peugeot', '2008', [(2008,2016), (2016,2021)]),
    #Car('opel', 'astra', [(2013,2021)]),
    #Car('peugeot', '308', [(2013, 2021)]),
    #Car('renault', 'megane-coupe', [(2013, 2021)]),

]
for car in cars:
    for generation in car.generations:
        olx_filter = OlxFilter(car)
        x = range(*generation)
        y = price_by_year(olx_filter, x)
        plot(x, y, f'{car} {generation}')
plt.show()

