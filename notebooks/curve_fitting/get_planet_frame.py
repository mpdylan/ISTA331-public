from bs4 import BeautifulSoup
import pandas as pd

''' 
This is to be cut-and-paste to start in-class e.g.
'''

def get_planet_frame():
    ''' list comprehensions, enumerate '''
    ''' transposes as it creates the lol '''
    soup = BeautifulSoup(open('planets.html'))
    rows = soup.table.find_all('tr')
    # rows.pop(0) is the header with the planet names but first td is blank
    index = [td.a.get_text().capitalize() for td in rows.pop(0).find_all('td')[1:]]
    rows.pop() # the last row is also a header - get rid of it
    columns = [tr.td.a.get_text() for tr in rows]
    data = [[] for i in range(len(index))]
    for row in rows:
        for i, td in enumerate(row.find_all('td')[1:]):
            data[i].append(td.get_text())
    return pd.DataFrame(data, index, columns)


    
    