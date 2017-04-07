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

    
# LOAD DATA   
file_path_ngo_budget = 'Data/NGO_DataDisbursement.csv'
ngo_file = open(file_path_ngo_budget, 'r')
reader = csv.DictReader(ngo_file)
reader = sorted(reader, key=lambda d: d['ISO_code'])
countries = []
curr_country = ''
dict_country_data = defaultdict(list)
for row in reader:
    #print(row)
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
    #dict_country_data[c] = np.sum(dict_country_data[c], axis=0)
    country_data.append(np.sum(dict_country_data[c], axis=0))

country_data = np.array(country_data)
country_data = country_data.astype(float)
print(country_data)
#plt.matshow(country_data.T)
#plt.colorbar()
#plt.show()    
country_data = convert_to_float_ex(country_data)
print(country_data)
country_data = country_data.T
plt.matshow(country_data)
plt.colorbar()
plt.show()
    
file_path_l_e = 'Data/life_exp.csv'
le_file = open(file_path_l_e, 'r')
reader = csv.DictReader(le_file)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_l_e = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    #print(row['Country Code'])
    rows = [row[str(y)+' [YR'+str(y)+']'] for y in years ]
    data_l_e.append(rows)
    
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
#2000 [YR2000]
for row in reader:
    if row['Country Code'] not in countries:
        continue
    rows = [row[str(y)+' [YR'+str(y)+']'] for y in years ]
    
    data_pop.append(rows)
    
    
found_countries = []
f = 'Data/sanitation2.csv'
ff = open(f, 'r')
reader = csv.DictReader(ff)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_san = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    rows = [row[str(y)+' [YR'+str(y)+']'] for y in years ]
    found_countries.append(row['Country Code'])
    data_san.append(rows)   
found_countries = set(found_countries) 
fail = list(set(countries) - found_countries)
for f in fail:
    print(f)

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
    
    
f = 'Data/poverty.csv'
ff = open(f, 'r')
reader = csv.DictReader(ff)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_poverty = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    rows = [row[str(y)+' [YR'+str(y)+']'] for y in years ]
    data_poverty.append(rows)    
    
    
f = 'Data/prevalence.csv'
ff = open(f, 'r')
reader = csv.DictReader(ff)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_prel = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    rows = [row[str(y)+' [YR'+str(y)+']'] for y in years ]
    data_prel.append(rows)   
    
data_l_e = convert_to_float(data_l_e)
data_l_e = data_l_e.T
plt.matshow(data_l_e)
plt.colorbar()
plt.show()
#print(data)

print(countries)

data_pop = convert_to_float(data_pop)
data_ch_m_ORIG = convert_to_float(data_ch_m)
data_san_ORIG = convert_to_float(data_san)
data_water_ORIG = convert_to_float(data_water)
data_poverty_ORIG = convert_to_float(data_poverty)
data_prel_ORIG = convert_to_float(data_prel)
#data_ch_m = np.multiply(data_ch_m, data_pop) # abs valus
#data_san = np.multiply(data_san, data_pop) # abs valus
#data_ch_m = data_ch_m * 0.001
#data_san = data_san * 0.01



# plt.matshow(data_ch_m.T)
# plt.colorbar()
# plt.show()

# plt.matshow(data_san.T)
# plt.colorbar()
# plt.show()

# plt.matshow(data_water.T)
# plt.colorbar()
# plt.show()

# plt.matshow(data_poverty.T)
# plt.colorbar()
# plt.show()

# plt.matshow(data_prel.T)
# plt.colorbar()
# plt.show()


data_san = data_san_ORIG
data_water = data_water_ORIG
data_ch_m =data_ch_m_ORIG


# ITERATE FOR NEXT 5 YEARS
for new_year in range(2017, 2022):

    ################### REGRESSION OF FUTURE WATER; SANITARY and POPULATION ####################################
    reg_san, coeff_san = regress(years, data_san.T)

    reg_water, coeff_water = regress(years, data_water.T)

    reg_pop, coeff_pop = regress(years, data_pop.T)


    #################### LEAST SQUARE to find MORTALITY COEFFS ####################################

    data_ch_m = np.multiply(data_ch_m/10000.0,data_pop)
    #data_san= np.multiply(data_san/100.0,data_pop)
    #data_water = np.multiply(data_water/100.0,data_pop)

    print(country_data.shape,data_ch_m.shape,data_san.shape)
    coeff = []
    err = []
    for idx,c in enumerate(country_data.T):
        l = len(c)
        A = np.ones(l)
        A = np.vstack([A,c])
        #A = np.vstack([A,data_san[idx,:]])
        #A = np.vstack([A,data_water[idx,:]])
        #plt.plot(range(l),data_pop[idx,:])
        #plt.show()
        A = A.T
        #C,R = lstsq(A,data_ch_m[idx,:])[:2]
        reg = LinearRegression()
        reg.fit(A,data_ch_m[idx,:].reshape(-1,1))
        C = reg.coef_[0]
        p = reg.p[0]
        e = reg.score(A,data_ch_m[idx,:])
        err.append(e.mean())
        #C = list(zip(C,p))
        coeff.append(C)
    #print(err)
    plt.show()
    coeff = np.array(coeff)
    #print(coeff.shape)
    #x_range = range(len(coeff))
    #plt.plot(x_range,coeff[:,1])
    #plt.show()
    #print(coeff)




    #################################### LINEAR PROGRAM ######################################################

    beta = coeff[:,1]
    #reg_san = np.multiply(reg_san/100.0,reg_pop)
    #reg_water = np.multiply(reg_water/100.0,reg_pop)

    #reg_san = np.multiply(reg_san,coeff[:,2])
    #reg_water = np.multiply(reg_water,coeff[:,3])

    c = coeff[:,0] #+ reg_san + reg_water
    c = c.reshape(-1)
    #print(c.shape)

    A = np.ones(len(beta))
    eye_m = -np.diag(beta)
    A = np.vstack([A,eye_m])
    #print(A)

    b = np.zeros(len(beta) +1)
    b[0] = 1e9
    b[1:] += c

    #idx = np.where(np.logical_or(b[1:] >= 0,-1*beta >= 0))


    #beta = beta[idx]
    #b = np.zeros(len(beta) +1)
    #c = c[idx]
    #b[0] = 1e9
    #b[1:] += c
    #A = np.ones(len(beta))
    #eye_m = -np.diag(beta)
    #A = np.vstack([A,eye_m])



    res = linprog(beta, A_ub=A, b_ub=b)
    X = res['x']
    #print(res['x'],np.sum(res['x']))
    new_mort_rate = np.multiply(X,beta) + c
    print(np.sum(new_mort_rate))


    #################################### UPDATE THE VALUES ####################################


    # check if they have to be rescaled

    print('data_san: ', data_san.shape)
    print('reg_san: ', reg_san.shape)
    data_san = np.hstack([data_san,reg_san.T])

    reg_water = np.hstack((data_water,reg_water.T))
    country_data = np.vstack((country_data, X))
    
    new_mort_rate = new_mort_rate.reshape((len(new_mort_rate),1))
    print('data_ch_m: ', data_ch_m.shape)
    print('new_mort_rate: ', new_mort_rate.shape)
    data_ch_m = np.hstack((data_ch_m, new_mort_rate))
    
    years.append(new_year)
    print('iter for ', new_year)

print('done')
