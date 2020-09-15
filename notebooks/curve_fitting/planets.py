from bs4 import BeautifulSoup
import pandas as pd

''' 
Start an ipython sell with ipython --matplotlib
We do HTML parsing in 350.  Posting this code, but not teaching it.
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
    
pf = get_planet_frame()
print(pf)

import matplotlib.pyplot as plt

plt.plot(pf['Mass'], pf['Gravity'], linestyle='', marker='o')
# wtf??!!!  That's not right.  Why are the x values out of order?
# They're strings, for one thing.

mgs = pd.Series(pf['Gravity'].values, pf['Mass'].values).sort_index()
# huh.  That doesn't look sorted.  Take a closer look at the index:
mgs.index # Ok, let's start over

import numpy as np

index = pf['Mass'].values.astype(np.float64)
data = pf['Gravity'].values.astype(np.float64)
mgs = pd.Series(data, index).sort_index()
mgs # ok, better

ax = mgs.plot(linestyle='', marker='o')
# Whoa, that doesn't look like a straight line.  Gravity is supposed to 
# be proportional to m: g = Gm/r^2.  What's going on?
# Must be gravity on the surface so radius is messing with our result.  
# Let's find out.
ax # you can see why I called it ax

radii = pf['Diameter'].astype(np.float64) # oops, commas
radii = pd.Series(pf['Diameter'].values, index).sort_index()
radii = radii.replace({',': ''}, regex=True).astype(np.float64) / 2
# got this from Stack Overflow, wouldn't have figured it from the docs
# the regex is saying search for this within the values and replace
# We'll do regex's in the Spring in 350

mgs = mgs * radii**2 
ax = mgs.plot(linestyle='', marker='o') 
# ok, that looks like a line, let's fit one to it

import statsmodels.api as sm

m_array = sm.add_constant(mgs.index) # necessary to get the intercept
model = sm.OLS(mgs, m_array)
results = model.fit()
for attribute in dir(results):
    print(attribute)
results.summary()
results.mse_resid # this is what we want to use to get RMSE,
# it has the correct ddof (which should be the number of params we are fitting)
results.params
type(results.params) # would be an array if y was a list or an array 
# instead of a Series.  Give a pandas, get a pandas.  Order would be reversed, too.

ax.plot(mgs.index, results.params['x1'] * mgs.index + results.params['const'])

# let's look at at different relationship in this data:
# mass vs. distance from the Sun

index = pf['Distance from Sun']
index # that Moon will mess us up, let's stick to stuff orbiting the Sun
index = index.drop('Moon').values.astype(np.float64)
data = pf['Mass'].drop('Moon').values.astype(np.float64)
# we had masses in our index before, but the Moon was in there
# easier to just start over
dms = pd.Series(data, index) # already in sorted order, don't need sort_index
dms
ax = dms.plot(linestyle='', marker='o')

d_array = sm.add_constant(dms.index) 
model = sm.OLS(dms, d_array)
results = model.fit()
results.summary()
results.mse_resid # this is what we want to use to get RMSE,
# it has the correct ddof (which should be the number of params we are fitting)
results.params

ax.plot(dms.index, results.params['x1'] * dms.index + results.params['const'])
# it doesn't look like a parabola would fit this very well either, 's
# let try a cubic.  We are going to use a trick.  OLS will fit as many
# coefficients to linear variables as we want, so we are going to turn the 
# higher order terms into linear variables by doing some of the math beforehand:
X = np.column_stack((dms.index, dms.index**2, dms.index**3))
X = sm.add_constant(X)
model = sm.OLS(dms, X)
results = model.fit()
results.summary() # coeffs come back in the order we gave the vals,
# so x1 is still the linear coeff, x2 is the quadratic, etc.
ax.plot(dms.index, results.params['x3'] * dms.index**3 + results.params['x2'] * dms.index**2 + results.params['x1'] * dms.index + results.params['const'])

# not so great, look at this:
# https://en.wikipedia.org/wiki/Rayleigh_distribution
# we are going to optimize (is not a variable, e = 2.781828...
# f(x) = a * (bx + c) / alpha^2 * e^(-(bx + c)^2 / (2 * alpha^2)) + d
# describe what optimizing is

from scipy.optimize import curve_fit

def func(x, alpha, a, b, c, d):
    return a * (b * x + c) / alpha**2 * np.e**(-(b * x + c)**2 / (2 * alpha**2)) + d

popt, pcov = curve_fit(func, dms.index, dms)
'''
# I like this but a less weird way follows:
f = lambda x: func(x, *popt) # a lambda is a one-line anonymous function 
# Well, until I assigned it to a variable, i.e. named it.
rmse = (sum((f(x) - dms[x])**2 for x in dms.index) / 4)**0.5
'''
rmse = (sum((func(x, *popt) - dms[x])**2 for x in dms.index) / 4)**0.5
# 4 = num_data_pts - num_params_optimized = N - ddof = 9 - 5
rmse # worse than cubic!
popt # ughh.  We didn't give starting vals for our params,
# so by default, curve_fit sets them to 1.  Most didn't change at all.

y = [f(x) for x in dms.index]
ax.plot(dms.index, y)

# put the matplotlib fig next to the wikipedia fig and estimate params
params_0 = [0.6, 1000, 0.001, 0, 0]
popt, pcov = curve_fit(func, dms.index, dms, params_0)
f = lambda x: func(x, *popt)
rmse = (sum((f(x) - dms[x])**2 for x in dms.index) / 4)**0.5
rmse # really bad, but much better.  Next year, maybe try gamma distribution

y = [f(x) for x in dms.index]
ax.plot(dms.index, y)
 