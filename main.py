# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:40:15 2023

@author: Daniel Pessoa
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import numpy as np
from radar import create_radar

header = st.container()
dataset = st.container()
features = st.container()
modeltraining = st.container()
scaler = StandardScaler()



@st.cache_data #Aplicando cashing
def get_data(filename):
 #   "Dataframe de origem"
    df_all_stats = pd.read_csv(filename).drop(columns = ['Unnamed: 0'])
    df_all_stats = df_all_stats[(df_all_stats['90s'] > 0)].reset_index().drop(columns = 'index')
    df_all_stats['nome_amigavel'] = [f"{nome} - {clube}" for nome, clube in zip(df_all_stats['Player'], df_all_stats['Club'])]
    
   # "Separando as posições"
    
    df_all_stats['Position'] = df_all_stats['Pos'].str[:2]
    df_all_stats['Position_2'] = df_all_stats['Pos'].str[3:]
    df_all_stats['Position'] = df_all_stats['Position'].replace({'MF': 'Midfielder', 'DF': 'Defender', 'FW': 'Forward', 'GK': 'Goalkeeper'})
    df_all_stats['Position_2'] = df_all_stats['Position_2'].replace({'MF': 'Midfielder', 'DF': 'Defender',
                                                 'FW': 'Forward', 'GK': 'Goalkeeper'})
    
    return df_all_stats

@st.cache_data
def convert_df(df):
    
    try:
        return df.to_csv().encode('latin')
    except:
        return df.to_csv().encode('utf-8')
    
@st.cache_data #Aplicando cashing
def get_sig_variable(dataframe):   
    
    plyr_info = dataframe[['Player','Nation','Competition','Confederation','Country','Club', 'Position']]
    
    df_int = dataframe.select_dtypes(include=['int', 'float64'])
    percent_threshold = 0.7  
    zero_percentage = df_int.apply(lambda col: (col == 0).mean())
    #variance_selector = VarianceThreshold(threshold=0.7)
   # variance_selector.fit_transform(df_int)
    #selected_features = np.array(np.where(variance_selector.get_support()))[0]
    selected_features = zero_percentage[zero_percentage >= percent_threshold].index
    
    #df_int = df_int.iloc[:, selected_features]
    df_int = df_int.drop(columns=selected_features)
    
    dataframe = pd.merge(plyr_info, df_int, left_index=True, right_index=True, how='left')   
    
    dataframe = dataframe[dataframe['Competition'] != 'Coupe de la Ligue'].reset_index().drop(columns = 'index')
    
    return dataframe


@st.cache_data #Aplicando cashing
def select_country(fa):

    if fa != 'All':
        list_country = sorted(list(df_all_stats[df_all_stats['Confederation']== fa]['Country'].unique()))
        list_country.insert(0, 'All')

    else:
        list_country = sorted(list(df_all_stats['Country'].unique()))
        list_country.insert(0, 'All')
       
    return list_country


@st.cache_data #Aplicando cashing
def select_comp(country_comp,fa):

       
    if country_comp == 'All' and fa == 'All':
        
        list_comp = sorted(list(df_all_stats['Competition'].unique()))
        list_comp.insert(0, 'All')
        
        list_club = sorted(list(df_all_stats['Club'].unique()))
        list_club.insert(0, 'All')
        
    elif country_comp == 'All' and fa != 'All':
        list_comp = sorted(list(df_all_stats[df_all_stats['Confederation']== fa]['Competition'].unique()))
        list_comp.insert(0, 'All')
        list_club = sorted(list(df_all_stats[df_all_stats['Confederation']== fa]['Club'].unique()))

    else:                     
        
        list_comp = sorted(list(df_all_stats[df_all_stats['Country']== country_comp]['Competition'].unique()))        
        list_club = sorted(list(df_all_stats[(df_all_stats['Country']== country_comp)]['Club'].unique()))
        list_club.insert(0, 'All')

        
    return list_comp


@st.cache_data #Aplicando cashing
def select_club(comp,fa):

      
    if comp == 'All' and fa == 'All':
        
        list_comp = sorted(list(df_all_stats['Competition'].unique()))
        list_comp.insert(0, 'All')
        
        list_club = sorted(list(df_all_stats['Club'].unique()))
        list_club.insert(0, 'All')
        
    elif comp == 'All' and fa != 'All':
        list_comp = sorted(list(df_all_stats[df_all_stats['Confederation']== fa]['Competition'].unique()))
        list_comp.insert(0, 'All')
        list_club = sorted(list(df_all_stats[df_all_stats['Confederation']== fa]['Club'].unique()))

    else:                     
        
        list_comp = sorted(list(df_all_stats[df_all_stats['Country']== country_comp]['Competition'].unique()))        
        list_club = sorted(list(df_all_stats[(df_all_stats['Country']== country_comp)  & (df_all_stats['Competition']== comp) ]['Club'].unique()))
        list_club.insert(0, 'All')

        
    return  list_club

@st.cache_data #Aplicando cashing
def select_player(position,dataframe):
        
    if position == 'All' :   
        list_players = list(dataframe['nome_amigavel'].unique())
    else : 
        dataframe = dataframe[(dataframe['Position'] == position)].reset_index().drop(columns = 'index')
        list_players = list(dataframe['nome_amigavel'].unique())
        
    return list_players, dataframe


    

with header:
    imagem_path = 'https://i.pinimg.com/736x/5a/79/02/5a7902ddfb16049b621f1a5c341b1320.jpg'
    
    st.markdown(
        f"<div style='display: flex; align-items: center; justify-content: center;'>"
        f"<img src='{imagem_path}' alt='Imagem' style='width: 100px; height: 100px; margin-right: 10px;'>"
        "<div>"
        f"<h1 style='text-align: left; font-family: Arial, sans-serif;'>Sistema de recomendação de jogadores ⚽📈</h1>"
        "<p style='font-size: 19px; font-weight: bold; color: black;'>"
        "🔴 Escolha um jogador à esquerda para obter os atletas com os perfis mais semelhantes."
        "</p>"
        "</div>"
        "</div>",
        unsafe_allow_html=True
    )
 
     
    
    
with dataset:
   # st.header('FBREF Dataset')
   # st.text('Dataset Disponivel no Site: fbref.com')
    
    #Dataframe de origem
    df_all_stats = get_data('merged_df_2.csv')
    df_copy = df_all_stats.copy()
    
    
    #Convertido em estatisticas por 90 min
  #  df_all_stats = convert_into_90(df_all_stats)

 #   st.write(df_all_stats.head(20))
    
    
    
    
with features:
#    st.header('Variáveis por Posição')

    col1, col2 = st.columns(2)
    col_side1,col_side2 =  st.sidebar.columns(2)
    
    #Padrnização do dataframe
    range_players = range(5,65,5)
    num_lista = st.sidebar.selectbox('Quantos jogadores deseja ver?',options = range_players ,index = 0)    
    
    
    
    #Selecionando idades máxima e mínima
    ages = range(15,45,1)
    idade_minima = col_side1.selectbox('Idade Mínima',options = ages,index = 0)
    ages_maior = [age for age in ages if age > idade_minima]
    idade_maxima = col_side2.selectbox('Idade Máxima',options = ages_maior,index = 0)
    
    #Selecionando a nacionalidade dos atletas que quero buscar
    list_nation = sorted(list(df_all_stats[df_all_stats['Nation'] != '0']['Nation'].unique()))
    list_nation.insert(0,'All')
    nation = st.sidebar.selectbox('Quer buscar atletas de qual nacionalidade? :',options = list_nation,index = 0) 
    
    
    #Selecionando o país a ser observado
    list_fa = sorted(list(df_all_stats['Confederation'].unique()))
    list_fa.insert(0, 'All')
    fa = st.sidebar.selectbox('Em qual confederação deseja observar atletas?',options = list_fa,index = 0)
    
    list_country = select_country(fa)     
    country_comp = st.sidebar.selectbox('Em qual país deseja observar atletas?',options = list_country,index = 0)
    
    #Selecionando Competição
    list_comp = select_comp(country_comp,fa)
    comp = st.sidebar.selectbox('Selecione a Competição',options = list_comp,index = 0)
    list_club = select_club(comp,fa)
    list_club.insert(0, 'All')
    club = st.sidebar.selectbox('Deseja Observar Algum Clube Específico?:',options = list_club,index = 0)  
    
    #Padrnização do dataframe
    list_positions = ['All','Midfielder', 'Defender', 'Forward', 'Goalkeeper']
    position = st.sidebar.selectbox('Selecione uma posição que deseja observar:',options = list_positions ,index = 0)
    
    
    #Selecionando Jogador a Ser Comparado
    list_players,dataframe = select_player(position,df_all_stats)
    player_name = st.sidebar.selectbox('Selecione o jogador',options = sorted(list_players),index = 0)  
    
    index_player = {name: index for index, name in enumerate(df_all_stats['nome_amigavel'])}  
    index = index_player[player_name]
    plry_position = df_all_stats['Position'].iloc[index]
    plyr_real_name = df_all_stats['Player'].iloc[index]

    

    
with modeltraining:
  #  st.header('Treino do Modelo')
 #   st.text('Teste')
    

    
    

    df_all_stats = get_sig_variable(df_all_stats)
    df_new_ind = df_all_stats.select_dtypes(include=['int', 'float64'])
    data_new_normalized = scaler.fit_transform(df_new_ind)
    
    # "Pegando os dados do jogador a ser comparado"
    player_data = df_new_ind.iloc[index].values.reshape(1, -1)
    player_data_normalized = scaler.transform(player_data)        


    
    similarities = cosine_similarity(player_data_normalized, data_new_normalized)
    similarities_converted = ((similarities + 1) / 2 * 100).round(1)
    
    
  
    
    result_df = pd.DataFrame({'Player': df_all_stats['Player'],'Nation':df_all_stats['Nation'],'Age':df_all_stats['Age'],'Position':df_all_stats['Position'],'Competition':df_all_stats['Competition'],'Club': df_all_stats['Club'],'% similarity': similarities_converted[0]}).drop_duplicates()
    
    
    result_df = pd.merge(result_df[['Player','Nation', 'Competition','Club', 'Position', '% similarity']], df_all_stats, how='left', on=['Player','Nation', 'Club', 'Position','Competition'])
    
    if position == 'All':
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Age'] >= idade_minima) & (result_df['Age'] <= idade_maxima) & (result_df['Position']== plry_position) ].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()
    else:
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Age'] >= idade_minima) & (result_df['Age'] <= idade_maxima) & (result_df['Position']== position)].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()

    if fa == 'All':
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Position']== plry_position)].drop_duplicates().reset_index().drop(columns = 'index')
    else:
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Confederation'] == fa) & (result_df['Position']== plry_position)].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()
    
    if  comp == 'All':
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Position']== plry_position)].drop_duplicates().reset_index().drop(columns = 'index')
    else:
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Competition'] == comp) & (result_df['Position']== plry_position)].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()

    if  club == 'All':
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Position']== plry_position)].drop_duplicates().reset_index().drop(columns = 'index')
    else:
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Club'] == club) & (result_df['Position']== plry_position)].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()


    if nation == 'All':
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Position']== plry_position)].drop_duplicates().reset_index().drop(columns = 'index')
    else:
        result_df = result_df[(result_df['Player'] !=plyr_real_name) & (result_df['Nation'] == nation) & (result_df['Position']== plry_position)].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()

        



    st.markdown("""<p style='font-size: 24px; font-weight: bold; color: black;'> Os {} jogadores mais semelhantes à {} entre {} e {} anos.</p>""".format(num_lista,player_name,idade_minima,idade_maxima), unsafe_allow_html=True)
        
   # col2.subheader('Jogadores mais semelhantes à {}'.format(player_name))    
    st.write(result_df.head(num_lista), width=100)


    col_buttom1,col_buttom2,col_buttom3 = st.columns(3)


    csv_file = convert_df(result_df.head(num_lista))
    player_name_file = player_name.lower().replace(" ", "_")
    filename = 'analise_similaridade_{}_top_{}_{}_a_{}_anos.csv'.format(player_name_file,num_lista,idade_minima,idade_maxima)
    col_buttom1.download_button( label="Download data as CSV",data=csv_file,file_name=filename,mime='text/csv')
    
    # Crie o select box
    options = list(result_df['Player'].unique())[:num_lista]
    plyr2 = col_buttom3.selectbox("Com qual jogador gostaria de comparar?", options)

    # Imprima o valor selecionado
  #  col_buttom2.write("Data to download:", select_box)

       
    fbref_link = "https://fbref.com/en/"
    st.sidebar.caption("Fonte de Dados:")
    st.sidebar.markdown(f'<a href="{fbref_link}" target="_blank"><img src="https://cdn-images.threadless.com/threadless-media/artist_shops/shops/sportsreference/products/2222678/original-1629135299-2b19dc9a2fbc3ddebb137b63bb692e76.png?v=3&d=eyJvcHMiOiBbWyJ0cmltIiwgW2ZhbHNlLCBmYWxzZV0sIHt9XSwgWyJyZXNpemUiLCBbXSwgeyJ3aWR0aCI6IDk5Ni4wLCAiYWxsb3dfdXAiOiBmYWxzZSwgImhlaWdodCI6IDk5Ni4wfV0sIFsiY2FudmFzX2NlbnRlcmVkIiwgWzEyMDAsIDEyMDBdLCB7ImJhY2tncm91bmQiOiAiZmZmZmZmIn1dLCBbInJlc2l6ZSIsIFs4MDBdLCB7fV0sIFsiY2FudmFzX2NlbnRlcmVkIiwgWzgwMCwgODAwLCAiI2ZmZmZmZiJdLCB7fV0sIFsiZW5jb2RlIiwgWyJqcGciLCA4NV0sIHt9XV0sICJmb3JjZSI6IGZhbHNlLCAib25seV9tZXRhIjogZmFsc2V9" alt="Fonte de Dados:" style="height: 60px;"></a>', unsafe_allow_html=True)
 
 
 
    col_radar1, col_radar2 = st.columns(2)
 
    radar1 = create_radar(df_copy,player_name,tipo_nome = 'amigavel')
    radar2 = create_radar(df_copy,plyr2,tipo_nome = 'normal')
  
    col_radar1.pyplot(radar1)
    col_radar2.pyplot(radar2)
   