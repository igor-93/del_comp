import csv
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from scipy import stats
from scipy.optimize import linprog
from scipy.linalg import lstsq

from collections import defaultdict

years = np.arange(2001,2017)   # for data

class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self



def convert_to_float(data):
    data = np.array(data)
    idx = np.where(data == '')
    data[idx] = 0
    idx = np.where(data == '..')
    data[idx] = 0
    data = data.astype(float)
    row_means = np.true_divide(data.sum(1),(data!=0).sum(1))
    for i in range(data.shape[0]):
        idx = np.where(data[i,:] == 0)
        data[i,idx] = row_means[i]
        j0 = data.shape[1]-2
        if j0 == 0:
            data[i,j0] = 2*data[i,j0-1] - data[i,j0-2]
        j = data.shape[1]-1
        #if data[i,j] == 0:
        data[i,j] = 2*data[i,j-1] - data[i,j-2]
            
    return data


def convert_to_float_ex(data):
    data = np.array(data)
    data = data.astype(float)
    #row_means = np.true_divide(data.sum(1),(data!=0).sum(1))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] != 0:
                continue
            if j == 0 or (j!=0 and data[i,j-1]==0):
                data[i,j] = 2*data[i,j+1] - data[i,j+2]
            if j == data.shape[1]-1 or (j!=data.shape[1]-1 and data[i,j+1]==0):
                data[i,j] = 2*data[i,j-1] - data[i,j-2]    
                
    data[data <0] = 0.0
    return data

    

def regress(year_series, values):
    year_series = year_series.reshape((len(year_series),1))
    ridge = Ridge()
    ridge.fit(year_series, values)
    new_val = ridge.predict(year_series[-1]+1)
    return new_val, ridge.coef_

    
########################################## LOAD THE DATA #########################################
file_path_ngo_budget = 'Data/NGO_DataDisbursement.csv'
ngo_file = open(file_path_ngo_budget, 'r')
reader = csv.DictReader(ngo_file)
reader = sorted(reader, key=lambda d: d['ISO_code'])
countries = []
curr_country = ''
dict_country_data = defaultdict(list)
for row in reader:
    country = row['ISO_code']
    countries.append(country)
    if curr_country != country:
        curr_country = country
    rows = [float(row[str(y)]) if row[str(y)] != '' else 0.0 for y in years ]    
    dict_country_data[curr_country].append(rows)    
    
countries = list(set(countries))
countries = sorted(countries)
country_data = []
for c in countries:
    country_data.append(np.sum(dict_country_data[c], axis=0))

country_data = np.array(country_data)
country_data = country_data.astype(float)
country_data = convert_to_float_ex(country_data)
country_data = country_data.T
    
    
f = 'Data/child_mort.csv'
ff = open(f, 'r')
reader = csv.DictReader(ff)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_ch_m = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    rows = [row[str(y)+' [YR'+str(y)+']'] for y in years ]
    data_ch_m.append(rows)
    
    
file_population = 'Data/population.csv'   
pop_file = open(file_population, 'r')
reader = csv.DictReader(pop_file)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_pop = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    rows = [row[str(y)+' [YR'+str(y)+']'] for y in years ]
    
    data_pop.append(rows)
    

f = 'Data/water.csv'
ff = open(f, 'r')
reader = csv.DictReader(ff)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_water = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    rows = [row[str(y)+' [YR'+str(y)+']'] for y in years ]
    data_water.append(rows) 

    
data_pop = convert_to_float(data_pop)
data_ch_m = convert_to_float(data_ch_m)
data_water = convert_to_float(data_water)



data_ch_m = np.multiply(data_ch_m/10000.0,data_pop) #Assume that 10 % of the Population are Children
country_data = ((country_data.T / np.sum(country_data,1))*1e9).T



# ITERATE FOR NEXT 5 YEARS
for new_year in range(2017, 2021):

    ################### REGRESSION OF FUTURE WATER; SANITARY and POPULATION ####################################
    #reg_san, coeff_san = regress(years, data_san.T)

    reg_water, coeff_water = regress(years, data_water.T)



    reg_pop, coeff_pop = regress(years, data_pop.T)


    #################### LEAST SQUARE to find MORTALITY COEFFS ####################################

    data_ch_m_curr = data_ch_m
    data_water_curr = np.multiply(data_water/100.0,data_pop) 

    coeff = []
    #err = []
    for idx,c in enumerate(country_data.T):
        l = len(c)
        A = np.ones(l) #Constant
        A = np.vstack([A,c]) #Money Poured into the Country
        A = np.vstack([A,data_water_curr[idx,:]]) #Number of People which have access to clean water
        A = A.T
        reg = LinearRegression()
        reg.fit(A,data_ch_m_curr[idx,:].reshape(-1,1)) # Estimate the Coefficients for Child Mortality = alpha + beta * Funds + ceta * Acces Water weighed by Population
        C = reg.coef_[0]
        coeff.append(C)


    coeff = np.array(coeff)




    #################################### LINEAR PROGRAM ######################################################

    beta = coeff[:,1] #Sum ovuer the Money weighted by our coefficients
    reg_water_curr = np.multiply(reg_water/100.0,reg_pop) #Number of People which have access to clean water which was predicted
    reg_water_curr = np.multiply(reg_water_curr,coeff[:,2]) #Multiply by coefficient 

    c = coeff[:,0] + reg_water_curr
    c = c.reshape(-1)


    A = np.ones(len(beta))
    eye_m = -np.diag(beta)
    A = np.vstack([A,eye_m])
    A = np.vstack([A,np.eye(len(beta))])

    b = np.zeros(len(beta) +1)
    b[0] = 1e9 # It has to sum up to 1 Billion 
    b[1:] += c
    b = np.hstack([b,country_data[-1,:] * 1.3]) # We can't spend more than 1.3 the former value


    res = linprog(beta, A_ub=A, b_ub=b) #Linear Program A*x <= b, Maximize for the money poured into the countries
    X = res['x']

    new_mort_rate = np.multiply(X,beta) + c 


    #################################### UPDATE THE VALUES ####################################



    data_water = np.hstack((data_water,reg_water.T))

    data_pop = np.hstack((data_pop,reg_pop.T))
    country_data = np.vstack((country_data, X))
    new_mort_rate = new_mort_rate.reshape((len(new_mort_rate),1))
    data_ch_m = np.hstack((data_ch_m, new_mort_rate))
    years = np.hstack((years, new_year))

print('done')

#################################### PLOTTING ####################################

v = np.argsort(country_data[-1])[::-1][:5]
x_range = np.arange(2015,2020)
for idx in v:
    plt.plot(np.arange(2010,2020),country_data[:-10,idx], label = countries[idx])
    plt.legend()

plt.xlabel("Years")
plt.ylabel("USD")
plt.title("Funds Allocation change for selected Years")
plt.show()

import sklearn

reg = sklearn.linear_model.LinearRegression()

new_mort_rate = np.sum(data_ch_m,0)

x = np.arange(2001,2015).reshape(-1,1)
y = new_mort_rate[:14]
reg.fit(x,y)
y = reg.predict(x_range.reshape(-1,1))

plt.plot(np.arange(2010,2020),new_mort_rate[-10:], label = "Our Model")
plt.plot(x_range,y+192000, label = "Linear Prediction")
plt.legend()
plt.title("Child Mortatility")
plt.show()

plt.plot(np.arange(2011,2021),np.sum(country_data != 0,1)[-10:])

plt.xlabel("Years")
plt.ylabel("Number of Countries")
plt.title("Number of Countries with Funds allocated")
plt.show()

plot_val = np.median(country_data[-5:],0)
x_ticks = range(len(plot_val))
plt.bar(x_ticks,plot_val,color='g')
plt.xticks(x_ticks, countries, rotation='vertical')
plt.xlabel("Countries")
plt.ylabel("Median Money Invested")
plt.title("Median Money Invested By Country (Predicted Data)")
plt.show()
