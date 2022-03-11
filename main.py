import streamlit as st
import pandas as pd
import numpy as np
from futilities.distances import cal_dist
from futilities.distances import cal_depot
from futilities.transfloat import commatoperiod
from futilities.transfloat import strtofloat
from futilities.GA import GA
import matplotlib.pyplot as plt
from datetime import datetime
import math


st.markdown("<h1 style='text-align: center; color: red;'>Barra de progreso</h1>", unsafe_allow_html=True)

my_bar = st.progress(0)

df = pd.read_csv('Lista_Datos_completos_data_09_02_Original.csv', sep=";")

muni = 'MUN_DES'

st_options = np.sort(df[muni].drop_duplicates().values)
muni_s = st.sidebar.multiselect('Ciudades',st_options,help='Seleccione la ciudad a planear')
current = [st.sidebar.date_input("Fecha de planeación", datetime.now(),help='Seleccione la fecha')]
cars = st.sidebar.number_input('Número de camiones',min_value=1,step=1)

op_limit = 20
vol_limit = [14.0,20.0]
st_CENDIS_LOG = st.sidebar.number_input("Ingrese longitud de origen",value=-74.8518516,step=1e-8,format="%.7f")
st_CENDIS_LAT = st.sidebar.number_input("Ingrese latitud de origen",value=10.9358485,step=1e-8,format="%.7f") 
st_done = st.sidebar.button("DONE")

lat = 'LATITUD'
lon = 'LONGITUD'
date = 'FEC_DES'
volumen = 'M3'
neighborhood = 'NOMBRE_EQUIVALENTE'
identifier = "ID"
accvolumen = 'vol_cumulative'
accop = 'op_cumulative'
bucket_by_vol = 'vol_bucket'
bucket_by_op = 'op_bucket'
df[date] = pd.to_datetime(df[date],format='%d/%m/%Y')

st.dataframe(df)

CENDIS = [st_CENDIS_LOG,st_CENDIS_LAT]
rng = np.random.default_rng(2022)

is_muni = df.loc[:,muni].isin(muni_s)
print(is_muni)
is_date = df.loc[:,date].isin(current)
df_muni = df[(is_muni) & (is_date)].copy()
df_muni[volumen] = df_muni[volumen]/1000
df_muni = df_muni.sort_values(by=[neighborhood,volumen])
df_muni[accvolumen] = df_muni.groupby([neighborhood])[volumen].cumsum()
n_op = df_muni.shape[0]
M3_total = df_muni[volumen].sum()
min_car = math.ceil(max([n_op/vol_limit[0],M3_total/op_limit]))


# In[5]:


volbucket_size = vol_limit[1]
df_muni[bucket_by_vol] = (df_muni[accvolumen]/volbucket_size).apply(math.ceil)
df_muni[accop] = df_muni.groupby([neighborhood,bucket_by_vol])[volumen].cumcount() + 1
opbucket_size = op_limit
df_muni[bucket_by_op] = (df_muni[accop]/opbucket_size).apply(math.ceil)
df_muni[identifier] = df_muni[neighborhood] + "_" + df_muni[bucket_by_vol].astype(str) + "_" + df_muni[bucket_by_op].astype(str)


# In[6]:


try:
  df_muni.loc[:,(lat)] = strtofloat(commatoperiod(df_muni.loc[:,(lat)]))
  df_muni.loc[:,(lon)] = strtofloat(commatoperiod(df_muni.loc[:,(lon)]))
  dist_matrix = cal_dist(df_muni,lat,lon,identifier)
  dist_CENDIS = cal_depot(df_muni, lat, lon, identifier, CENDIS)
except:
  print("Revise las latitudes y longitudes en su archivo, pueden haber algunas erróneas")

if st_done:
# In[7]:


    info_s = 'sum'
    info_c = 'count'
    info_table = df_muni.groupby(identifier)[volumen].agg([info_s,info_c])


# In[8]:

    epocas = 200
    barProgress = 100/epocas
    Model = GA(df_muni,dist_matrix,dist_CENDIS,info_table,cars,vol_limit,op_limit,rng,my_bar)
    Model.evolution(epocas)
    plt.plot(Model.means/Model.means.max())


# In[9]:


    carOrg = np.empty([Model.noptions],dtype=int)

    RouteModel = Model.orderingTraceSample(Model.bestOfBest["Cars"],Model.bestOfBest["Tour"])
    CarsModel = Model.bestOfBest["Cars"]

    for car in range(len(CarsModel)):
      carOrg[CarsModel[car]['inicio']:CarsModel[car]['fin']+1] = car + 1

    df_muni["CAMION"] = df_muni['ID'].apply(lambda x: carOrg[np.where(RouteModel == x)][0])
    df_muni["ORDEN"] = df_muni['ID'].apply(lambda x: np.where(RouteModel == x)[0][0])
    df_muni = df_muni.sort_values(by=['CAMION',"ORDEN"])


# In[10]:


    df_final_muni = df_muni.drop(['vol_cumulative','vol_bucket',"op_cumulative","op_bucket","ID","ORDEN"], axis = 1) 
    df_final_muni.to_excel("output.xlsx")


    # import geopy

    # from geopy.geocoders import Nominatim
    # from geopy.extra.rate_limiter import RateLimiter

    # street = st.sidebar.text_input("Street", "75 Bay Street")
    # city = st.sidebar.text_input("City", "Toronto")
    # province = st.sidebar.text_input("Province", "Ontario")
    # country = st.sidebar.text_input("Country", "Canada")

    # geolocator = Nominatim(user_agent="Ruta J")
    # geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    # location = geolocator.geocode(street+", "+city+", "+province+", "+country)

    # lat = location.latitude
    # lon = location.longitude

    # map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})

    # st.map(map_data) 