import csv
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

years = range(2000,2016)   # for data


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
    return data

    
    
file_path_ngo_budget = 'Data/NGO_DataDisbursement.csv'
ngo_file = open(file_path_ngo_budget, 'r')
reader = csv.DictReader(ngo_file)
countries = []
for row in reader:
    countries.append(row['ISO_code'])
    
countries = list(set(countries))
countries = sorted(countries)
print(len(countries))


file_path_l_e = 'life_exp.csv'
le_file = open(file_path_l_e, 'r')
reader = csv.DictReader(le_file)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_l_e = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    #print(row['Country Code'])
    rows = [row[str(y)] for y in years ]
    data_l_e.append(rows)
print(len(data_l_e))    
    
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
f = 'Data/sanitation.csv'
ff = open(f, 'r')
reader = csv.DictReader(ff)
reader = sorted(reader, key=lambda d: d['Country Code'])
data_san = []
for row in reader:
    if row['Country Code'] not in countries:
        continue
    rows = [row[str(y)] for y in years ]
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
    rows = [row[str(y)] for y in years ]
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
    
#print(data)
data_l_e = np.array(data_l_e)
idx = np.where(data_l_e == '')
data_l_e[idx] = 0
data_l_e = data_l_e.astype(float)
data_l_e = data_l_e.T
plt.matshow(data_l_e)
plt.colorbar()
plt.show()
#print(data)

print(countries)

data_pop = convert_to_float(data_pop)
data_ch_m = convert_to_float(data_ch_m)
data_san = convert_to_float(data_san)
data_water = convert_to_float(data_water)
data_poverty = convert_to_float(data_poverty)
data_prel = convert_to_float(data_prel)
#data_ch_m = np.multiply(data_ch_m, data_pop) # abs valus
#data_san = np.multiply(data_san, data_pop) # abs valus
#data_ch_m = data_ch_m * 0.001
#data_san = data_san * 0.01


data_ch_m = data_ch_m.T
plt.matshow(data_ch_m)
plt.colorbar()
plt.show()

data_san = data_san.T
plt.matshow(data_san)
plt.colorbar()
plt.show()
data_water = data_water.T
plt.matshow(data_water)
plt.colorbar()
plt.show()
data_poverty = data_poverty.T
plt.matshow(data_poverty)
plt.colorbar()
plt.show()
data_prel = data_prel.T
plt.matshow(data_prel)
plt.colorbar()
plt.show()