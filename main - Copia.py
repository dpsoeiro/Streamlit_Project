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
import numpy as np

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
    
    
  #  "Jogando tudo para  90min"
    
  #  df_info = df_all_stats[['Player','Club','Age','Pos']]
  #  df_ind =  df_all_stats.select_dtypes(include=['int', 'float64'])
  #  df_ind =  df_ind.div(df_ind['90s'], axis=0)
    
  #  df_all_stats = pd.merge(df_info, df_ind, left_index=True, right_index=True, how='left')
    
  #  "Tratando campo de idade"
    df_all_stats['Age'] = [age.split('-')[0] if '-' in age else age for age in df_all_stats['Age']]
    df_all_stats['Age'] = [age.split('.')[0] if '.' in age else age for age in df_all_stats['Age']]
    df_all_stats['Age'] = df_all_stats['Age'].astype('int')
    
   # "Separando as posições"
    
    df_all_stats['Position'] = df_all_stats['Pos'].str[:2]
    df_all_stats['Position_2'] = df_all_stats['Pos'].str[3:]
    df_all_stats['Position'] = df_all_stats['Position'].replace({'MF': 'Midfielder', 'DF': 'Defender', 'FW': 'Forward', 'GK': 'Goalkeeper'})
    df_all_stats['Position_2'] = df_all_stats['Position_2'].replace({'MF': 'Midfielder', 'DF': 'Defender',
                                                 'FW': 'Forward', 'GK': 'Goalkeeper'})
    
    
  #  df_all_stats_div = df_all_stats.div(df_all_stats['90s'], axis=0)

    return df_all_stats

@st.cache_data
def convert_df(df):
    
    try:
        return df.to_csv().encode('latin')
    except:
        return df.to_csv().encode('utf-8')
    
@st.cache_data #Aplicando cashing
def get_sig_variable(dataframe):   
    df_int = dataframe.select_dtypes(include=['int', 'float64'])
    variance_selector = VarianceThreshold(threshold=0.5)
    variance_selector.fit_transform(df_int)
    selected_features = np.array(np.where(variance_selector.get_support()))[0]
    
    df_int = df_int.iloc[:, selected_features]
    
    dataframe = pd.merge(dataframe[['Player','Nation','Competition','Confederation','Country','Club', 'Position']], df_int, left_index=True, right_index=True, how='left')
    
    
    dataframe = dataframe[dataframe['Competition'] != 'Coupe de la Ligue'].reset_index().drop(columns = 'index')
    
    return dataframe
    

with header:
 #   st.title('Sistema de recomendação de jogadores ⚽📈')
   #  st.title("Meu Título Centralizado")
     st.markdown("<h1 style='text-align: center,font-family: Arial, sans-serif;'>Sistema de recomendação de jogadores ⚽📈</h1>", unsafe_allow_html=True)
     st.markdown("""
         <p style='font-size: 19px; font-weight: bold; color: black;'> 🔴 Escolha um jogador à esquerda para obter os atletas com os perfis mais semelhantes.</p>
     """, unsafe_allow_html=True)
 #    st.text('Olhando para os jogadores, para avaliar perfis')
 
 
     
    
    
with dataset:
   # st.header('FBREF Dataset')
   # st.text('Dataset Disponivel no Site: fbref.com')
    
    #Dataframe de origem
    df_all_stats = get_data('df_all_stats_2023_2024_3.csv')
    
    
    #Convertido em estatisticas por 90 min
  #  df_all_stats = convert_into_90(df_all_stats)

 #   st.write(df_all_stats.head(20))
    
    
    
    
#with features:
#    st.header('Variáveis por Posição')
    
    
    #st.markdown('* **teste teste teste')
  #  st.markdown('* **teste2 teste2 teste2')
    
    
with modeltraining:
  #  st.header('Treino do Modelo')
 #   st.text('Teste')
    
    
    col1, col2 = st.columns(2)
    col_side1,col_side2 =  st.sidebar.columns(2)
        
      
    #Selecionando idades máxima e mínima
    ages = range(15,45,1)
    idade_minima = col_side1.selectbox('Idade Mínima',options = ages,index = 0)
    ages_maior = [age for age in ages if age > idade_minima]
    idade_maxima = col_side2.selectbox('Idade Máxima',options = ages_maior,index = 0)
    
    #Selecionando o país a ser observado
    list_fa = sorted(list(df_all_stats['Confederation'].unique()))
    list_fa.insert(0, 'All')
    fa = st.sidebar.selectbox('Selecione a Confederação da Competição',options = list_fa,index = 0)
    
    if fa != 'All':
        list_country = sorted(list(df_all_stats[df_all_stats['Confederation']== fa]['Country'].unique()))
        list_country.insert(0, 'All')

    else:
        list_country = sorted(list(df_all_stats['Country'].unique()))
        list_country.insert(0, 'All')
       
    country_comp = st.sidebar.selectbox('Selecione o País da Competição',options = list_country,index = 0)
    
  #  if fa != 'All' and country_comp != 'All':
      
    if country_comp == 'All' and fa == 'All':
        list_comp = sorted(list(df_all_stats['Competition'].unique()))
        list_comp.insert(0, 'All')
        comp = st.sidebar.selectbox('Selecione a Competição',options = list_comp,index = 0) 
        
        list_club = sorted(list(df_all_stats['Club'].unique()))
        list_club.insert(0, 'All')
        club = st.sidebar.selectbox('Selecione o Clube do Jogador Alvo:',options = list_club,index = 0)         
    elif country_comp == 'All' and fa != 'All':
        list_comp = sorted(list(df_all_stats[df_all_stats['Confederation']== fa]['Competition'].unique()))
        list_comp.insert(0, 'All')
        comp = st.sidebar.selectbox('Selecione a Competição',options = list_comp,index = 0)
        
        list_club = sorted(list(df_all_stats[df_all_stats['Confederation']== fa]['Club'].unique()))
        list_club.insert(0, 'All')
        club = st.sidebar.selectbox('Selecione o Clube do Jogador Alvo:',options = list_club,index = 0)        
    else:                     
        
        list_comp = sorted(list(df_all_stats[df_all_stats['Country']== country_comp]['Competition'].unique()))
        comp = st.sidebar.selectbox('Selecione a Competição',options = list_comp,index = 0)
        
        list_club = sorted(list(df_all_stats[(df_all_stats['Country']== country_comp) & (df_all_stats['Competition']== comp)]['Club'].unique()))
        list_club.insert(0, 'All')
        club = st.sidebar.selectbox('Selecione o Clube do Jogador Alvo:',options = list_club,index = 0) 

    
    
    #Padrnização do dataframe
    list_positions = ['All','Midfielder', 'Defender', 'Forward', 'Goalkeeper']
    position = st.sidebar.selectbox('Selecione uma posição',options = list_positions ,index = 0)
    
    #Padrnização do dataframe
    range_players = range(5,65,5)
    num_lista = st.sidebar.selectbox('Quantos jogadores deseja ver?',options = range_players ,index = 0)    
    

    
    if position == 'All' and club == 'All':
    
        #Selecionando o jogador a ser comparado"
    
    
        list_players = list(df_all_stats['Player'].unique())   

        player_name = st.sidebar.selectbox('Selecione o jogador',options = sorted(list_players),index = 0)    
        index_player = {name: index for index, name in enumerate(df_all_stats['Player'])}  
        index = index_player[player_name]
    
        
    elif position == 'All' and club != 'All':        
     #   df_all_stats = df_all_stats[df_all_stats['Club'] == club].reset_index().drop(columns = 'index')
        list_players = list(df_all_stats[df_all_stats['Club'] == club]['Player'].unique())   

        player_name = st.sidebar.selectbox('Selecione o jogador',options = sorted(list_players),index = 0)    
        index_player = {name: index for index, name in enumerate(df_all_stats['Player'])}  
        index = index_player[player_name]
        
    elif position != 'All' and club != 'All': 
        df_all_stats = df_all_stats[(df_all_stats['Position'] == position)].reset_index().drop(columns = 'index')
        list_players = list(df_all_stats[(df_all_stats['Club'] == club) & (df_all_stats['Position'] == position)]['Player'].unique())   


        player_name = st.sidebar.selectbox('Selecione o jogador',options = sorted(list_players),index = 0)    
        index_player = {name: index for index, name in enumerate(df_all_stats['Player'])}  
        index = index_player[player_name]
        
    elif position != 'All' and club == 'All':   
        df_all_stats = df_all_stats[df_all_stats['Position'] == position].reset_index().drop(columns = 'index')
        list_players = list(df_all_stats[df_all_stats['Position'] == position]['Player'].unique())   

        player_name = st.sidebar.selectbox('Selecione o jogador',options = sorted(list_players),index = 0)    
        index_player = {name: index for index, name in enumerate(df_all_stats['Player'])}  
        index = index_player[player_name]
    
    
    list_nation = sorted(list(df_all_stats[df_all_stats['Nation'] != '0']['Nation'].unique()))
    list_nation.insert(0,'All')
    nation = st.sidebar.selectbox('Nacionalidade dos jogadores :',options = sorted(list_nation),index = 0) 
    

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
        result_df = result_df[(result_df['Player'] !=player_name) & (result_df['Age'] >= idade_minima) & (result_df['Age'] <= idade_maxima)].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()
    else:
        result_df = result_df[(result_df['Player'] !=player_name) & (result_df['Age'] >= idade_minima) & (result_df['Age'] <= idade_maxima) & (result_df['Position']== position)].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()

    if fa == 'All':
        result_df = result_df .drop_duplicates()
    else:
        result_df = result_df[(result_df['Confederation'] == fa) ].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()
    
    if  comp == 'All':
        result_df = result_df.drop_duplicates()
    else:
        result_df = result_df[(result_df['Competition'] == comp) ].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()


    if nation == 'All':
        result_df = result_df.drop_duplicates()
    else:
        result_df = result_df[(result_df['Nation'] == nation)].sort_values(by ='% similarity' ,ascending = False).reset_index().drop(columns = 'index').drop_duplicates()

        



    st.markdown("""<p style='font-size: 24px; font-weight: bold; color: black;'> Os {} jogadores mais semelhantes à {} entre {} e {} anos.</p>""".format(num_lista,player_name,idade_minima,idade_maxima), unsafe_allow_html=True)
        
   # col2.subheader('Jogadores mais semelhantes à {}'.format(player_name))    
    st.write(result_df.head(num_lista))
    csv_file = convert_df(result_df.head(num_lista))
    player_name_file = player_name.lower().replace(" ", "_")
    filename = 'analise_similaridade_{}_top_{}_{}_a_{}_anos.csv'.format(player_name_file,num_lista,idade_minima,idade_maxima)
    st.download_button( label="Download data as CSV",data=csv_file,file_name=filename,mime='text/csv')

       
    fbref_link = "https://fbref.com/en/"
    st.sidebar.caption("Fonte de Dados:")
    st.sidebar.markdown(f'<a href="{fbref_link}" target="_blank"><img src="https://cdn-images.threadless.com/threadless-media/artist_shops/shops/sportsreference/products/2222678/original-1629135299-2b19dc9a2fbc3ddebb137b63bb692e76.png?v=3&d=eyJvcHMiOiBbWyJ0cmltIiwgW2ZhbHNlLCBmYWxzZV0sIHt9XSwgWyJyZXNpemUiLCBbXSwgeyJ3aWR0aCI6IDk5Ni4wLCAiYWxsb3dfdXAiOiBmYWxzZSwgImhlaWdodCI6IDk5Ni4wfV0sIFsiY2FudmFzX2NlbnRlcmVkIiwgWzEyMDAsIDEyMDBdLCB7ImJhY2tncm91bmQiOiAiZmZmZmZmIn1dLCBbInJlc2l6ZSIsIFs4MDBdLCB7fV0sIFsiY2FudmFzX2NlbnRlcmVkIiwgWzgwMCwgODAwLCAiI2ZmZmZmZiJdLCB7fV0sIFsiZW5jb2RlIiwgWyJqcGciLCA4NV0sIHt9XV0sICJmb3JjZSI6IGZhbHNlLCAib25seV9tZXRhIjogZmFsc2V9" alt="Fonte de Dados:" style="height: 60px;"></a>', unsafe_allow_html=True)
 
 
   