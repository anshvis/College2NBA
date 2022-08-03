from soupsieve import select
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nbaengine as nb
import urllib

st.title('College2NBA')

st.markdown("""
This app predicts a numeric probability of a certain college player making the NBA in a given year.
""")


with st.form(key = 'form1'):
        player_name = st.text_input("Player Name") #if someone doesnt enter a player name, it sends an error message
        selected_year = st.slider("Year",2008,2021,2021)
        button = st.form_submit_button(label = 'Calculate')

decision = bool

if button:
    if len(player_name) == 0:
        st.error("Please enter a valid input")
    else:

        player = 'college_player-' + str(selected_year)  + '.csv'
        draft = 'draft-' + str(selected_year) + '.csv'
        player_data = nb.clean_data(player, draft)
        prob = nb.run_engine(player, draft)
        print(prob)
        player_data['Draft Probability'] = prob

        player_data = player_data.drop('Drafted', axis=1)
        player_data = player_data[player_data.index.str.startswith(player_name)]
        player_prob = player_data[-1]
        player_data = player_data.drop('Draft Probability', axis=1)

        

        st.success("We predict {} {} get drafted in NBA {} Draft".format(player_name, decision, selected_year))
        st.caption("Our model works at an accuracy rate of {}%".format(round(player_prob)*100, 2))
    
        st.header("{}'s Player Stats for the {}-{} Season".format(player_name, selected_year-1, selected_year))

        

        st.dataframe(player_data)




# # Download NBA player stats data
# # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
#     return href

# st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)