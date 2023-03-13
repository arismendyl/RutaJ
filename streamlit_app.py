import streamlit as st
import pandas as pd
import numpy as np
from futilities.distances import cal_dist
from futilities.distances import cal_depot
from futilities.transfloat import commatoperiod
from futilities.transfloat import strtofloat
from futilities.GA import GA
import matplotlib.pyplot as plt
import plotly.express as px
import time
import math
import base64
import io
from PIL import Image
import plotly.graph_objects as go
import requests
from secret import secretUrl
import json

@st.experimental_memo(ttl=1800.0)
def cargandoDatos(secretUrl):
  response = requests.get(url = secretUrl)
  data = response.json()["op"]
  jsonResponse = json.dumps(data)
  df = pd.read_json(jsonResponse)
  df.drop_duplicates(keep='first',inplace=True)
  return df

st.set_page_config(
    page_title="RUTA J",
    page_icon=":truck:")

if __name__ == "__main__":

  timestr = time.strftime("%Y%m%d")

  bTrue = st.sidebar.button("Actualizar OP",help="Presione cada vez que quiera actualizar las OP")

  if bTrue:
    cargandoDatos.clear()

  try:
    df = cargandoDatos(secretUrl)
    dataUploaded = True
  except:
    st.warning("No se encuentran los datos de las Ops disponibles. Vuelva a cargar más tarde")
    dataUploaded = False

  if dataUploaded:
    
    st_titleOfPage = st.markdown("<h1 style='text-align: center;'>RUTA J</h1>", unsafe_allow_html=True)

    locationFilter = st.sidebar.radio("Filtrar las OP por ciudad o departamento:",('Ciudad', 'Departamento'), index=0)

    if locationFilter == 'Ciudad':
      muni = 'MUN_DES'
    else:
      muni = 'DEP_DES'

    pedidoSAP = "NUM_PEDIDO_SAP"
    celular = "CELULAR"

    lat = 'LATITUD'
    lon = 'LONGITUD'
    date = 'FEC_DES'
    zona = 'ZONA_LOG'
    entrega = 'NUM_ENTREGA_SAP'
    volumen = 'M3'
    neighborhood = 'NOMBRE_EQUIVALENTE'
    dangerousness = "ZONA_LOG"
    identifier = "ID"
    accvolumen = 'vol_cumulative'
    accop = 'op_cumulative'
    bucket_by_vol = 'vol_bucket'
    bucket_by_op = 'op_bucket'
    df[date] = pd.to_datetime(df[date]).dt.date
    df[entrega] = df[entrega].fillna(0).astype('int64')
    df[celular] = df[celular].fillna(0).astype('str')
    df[pedidoSAP] = df[pedidoSAP].fillna(0).astype('int64')
    st_mapcontainer = st.container()
    st_barcontainer = st.container()

    st_options = np.sort(df[muni].drop_duplicates().values)
    if locationFilter == 'Ciudad':
      muni_s = st.sidebar.multiselect('Ciudades',st_options,help='Seleccione la ciudad a planear')
    else:
      muni_s = st.sidebar.multiselect('Departamentos',st_options,help='Seleccione el departamento a planear')

    current = st.sidebar.multiselect("Fecha de planeación (AA-MM-DD)", np.sort(df['FEC_DES'].drop_duplicates().values),help='Seleccione la fecha')

    op_limit = 20
    vol_limit = [14.0,20.0]
    #st_CENDIS_LOG = st.sidebar.number_input("Ingrese longitud de origen",value=-74.8518516,step=1e-8,format="%.7f")
    #st_CENDIS_LAT = st.sidebar.number_input("Ingrese latitud de origen",value=10.9358485,step=1e-8,format="%.7f") 

    if "load_state" not in st.session_state:
      st.session_state.load_state = False

    #CENDIS = [st_CENDIS_LOG,st_CENDIS_LAT]
    CENDIS = [-74.8518516,10.9358485]
    rng = np.random.default_rng(2022)

    is_muni = df.loc[:,muni].isin(muni_s)
    is_date = df.loc[:,date].isin(current)
    df_muni = df[(is_muni) & (is_date)].copy()
    is_not_dangerous = ~df_muni.loc[:,zona].isin(['C0'])
    df_muni_C0 = df_muni[(~is_not_dangerous)].copy()
    df_muni = df_muni[(is_not_dangerous)].copy()

    isNotNull = ~df.loc[:,entrega].isin([0])
    entregasAlready = df_muni.loc[:,entrega].values
    isNotAlready = ~df.loc[:,entrega].isin(entregasAlready) 
    entrega_options = np.sort(df[(isNotNull)&(isNotAlready)].loc[:,entrega].drop_duplicates().values)
    entregasAdicionales = st.sidebar.multiselect('Entregas adicionales',entrega_options,help='Digite el número de entrega de los adelantos')
    isAnAdicional = df.loc[:,entrega].isin(entregasAdicionales)
    opAdicionales = df[(isAnAdicional)].copy()

    if len(entregasAdicionales)>0:
      frames = [opAdicionales.copy(),df_muni.copy()]
      df_muni = pd.concat(frames)

    df_muni[volumen] = df_muni[volumen]/1000
    M3_total = df_muni[volumen].sum()
    n_op = df_muni.shape[0]
    cubicajeTitle = st_mapcontainer.markdown("<h4 style='text-align: center;font-family:poppins;color:red;'>Cubicaje Total: "+str(round(M3_total,2)).replace(".", ",")+" m3</h4>", unsafe_allow_html=True)
    entregaTitle = st_mapcontainer.markdown("<h4 style='text-align: center;font-family:poppins;color:red;'>Entregas Totales: "+str(n_op).replace(".", ",")+"</h4>", unsafe_allow_html=True)
    st.dataframe(df_muni.reset_index(drop=True))
    df_muni = df_muni.sort_values(by=[neighborhood,volumen])
    df_muni[accvolumen] = df_muni.groupby([neighborhood])[volumen].cumsum()

    min_car = math.ceil(max([n_op/vol_limit[0],M3_total/op_limit]))

    try:
      ncars = st.sidebar.number_input('Número de camiones',min_value=1,value=min_car,step=1,help="Digite el número de camiones disponibles")
      ncars = int(ncars)

      if ncars < min_car:
        st.sidebar.warning("Podrías necesitar mínimo {} carros".format(min_car))
    except:
      pass


    st_loadData = st.sidebar.button("Ejecutar Ruta J",help='Oprima para realizar la planeación')


    volbucket_size = vol_limit[1]
    df_muni[bucket_by_vol] = (df_muni[accvolumen]/volbucket_size).apply(math.ceil)
    df_muni[accop] = df_muni.groupby([neighborhood,bucket_by_vol])[volumen].cumcount() + 1
    opbucket_size = op_limit
    df_muni[bucket_by_op] = (df_muni[accop]/opbucket_size).apply(math.ceil)
    df_muni[identifier] = df_muni[neighborhood] + "_" + df_muni[bucket_by_vol].astype(str) + "_" + df_muni[bucket_by_op].astype(str)

    #st.image(deliveryImage)

    # In[6]:

    if st_loadData: #or st.session_state.load_state:

      my_bar = st.empty()
      my_bar.progress(0)


      #st.session_state.load_state = True

      try:
        df_muni.loc[:,(lat)] = strtofloat(commatoperiod(df_muni.loc[:,(lat)]))
        df_muni.loc[:,(lon)] = strtofloat(commatoperiod(df_muni.loc[:,(lon)]))
        dist_matrix = cal_dist(df_muni,lat,lon,identifier)
        dist_CENDIS = cal_depot(df_muni, lat, lon, identifier, CENDIS)
        st_titleOfOPMap = st_mapcontainer.markdown("<h3 style='text-align: center;font-family:poppins;'>Mapa de órdenes de pedido</h3>", unsafe_allow_html=True)
        fig = px.scatter_mapbox(df_muni, lat=lat, lon=lon, color_discrete_sequence=["fuchsia"], hover_name=neighborhood, zoom=11, title="Mapa de Órdenes de Pedido")
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_geos(fitbounds="locations")  
        fig.update_layout(height=400,margin={"r":0,"t":0,"l":0,"b":0})
        st_mapcontainer.plotly_chart(fig)
      except:
        st.warning("Revise las latitudes y longitudes en su archivo, pueden haber algunas erróneas")

    # In[7]:

      info_s = 'sum'
      info_c = 'count'
      info_table = df_muni.groupby(identifier)[volumen].agg([info_s,info_c])

    # In[8]:

      epocas = 280
      barProgress = 100/epocas
      Model = GA(df=df_muni,dist_table=dist_matrix,cendis_table=dist_CENDIS,info_table=info_table,cars=ncars,weightlimit=vol_limit,oplimit=op_limit,rng=rng,bar=my_bar)
      Model.evolution(epocas)
      plt.plot(Model.means/Model.means.max())


    # In[9]:


      carOrg = np.empty([Model.noptions],dtype=object)
      carOrg[:] = np.nan 

      RouteModel = Model.orderingTraceSample(Model.bestOfBest["Cars"],Model.bestOfBest["Tour"])
      CarsModel = Model.bestOfBest["Cars"]

      for car in range(len(CarsModel)):
        if CarsModel[car] is not None:
          carOrg[CarsModel[car]['inicio']:CarsModel[car]['fin']+1] = car + 1

      df_muni["CAMION"] = df_muni['ID'].apply(lambda x: carOrg[np.where(RouteModel == x)][0])
      df_muni["ORDEN"] = df_muni['ID'].apply(lambda x: np.where(RouteModel == x)[0][0])
      df_muni = df_muni.sort_values(by=['CAMION',"ORDEN"])
      df_final_muni = df_muni.drop(['vol_bucket',"op_cumulative","op_bucket","ID"], axis = 1) 

      st_titleOfRouteMap = st_mapcontainer.markdown("<h3 style='text-align: center;font-family:poppins;'>Mapa de ruta</h3>", unsafe_allow_html=True)
      fig2 = go.Figure()
      grouped = df_final_muni.groupby("CAMION")
      for name, group in grouped:
        is_camion = df_final_muni.loc[:,("CAMION")].isin([name])
        group = df_final_muni[(is_camion)].copy()
        fig2.add_trace(
          go.Scattermapbox(
          mode = "markers+lines",
          lat = group[lat].tolist(),
          lon = group[lon].tolist(),
          hovertext = group[neighborhood].tolist(),
          marker = {"size": 10},
          name = "CAMIÓN {}".format(int(name)))
        )

      fig2.update_layout(height=400,margin={"r":0,"t":0,"l":0,"b":0}, mapbox = {
            'style': "open-street-map",
            'zoom': 11,
            'center': {'lon': df_final_muni[lon].mean(), 'lat': df_final_muni[lat].mean() }})
      fig2.update_geos(fitbounds="geojson")  
      st_mapcontainer.plotly_chart(fig2)

    # In[10]:

      def csv_downloader(data):
        towrite = io.BytesIO()
        downloaded_file = data.to_excel(towrite, encoding='utf-8', index=False, header=True)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        new_filename = "Ruta_J_{}.xlsx".format(timestr)
        linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{new_filename}"><h3 style="text-align: center; text-decoration: underline; border-style: dashed">Descargar Archivo 📥</h3></a>'
        st.markdown(linko, unsafe_allow_html=True)

      my_bar.empty()

      df_muni_C0['CAMION'] = 'ZONA DIFICIL ACCESO'

      df_final_muni['ACUMULADO'] = df_muni.groupby(['CAMION'])[volumen].cumsum()
      df_final_muni['ORDEN'] = df_muni.groupby(['CAMION'])[volumen].cumcount()+1
      df_final_muni = pd.concat([df_final_muni, df_muni_C0])

      csv_downloader(df_final_muni.loc[:,('REM','C_AGR','DEP_DES','MUN_DES','NOMBRE_EQUIVALENTE','DIR_DES','CAMION','ACUMULADO','ORDEN','NUM_ENTREGA_SAP','M3','TELEFONOS','CELULAR','ZONA','NUM_PEDIDO_SAP')])
