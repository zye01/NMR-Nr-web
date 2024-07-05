import pandas as pd
import numpy as np
import streamlit as st
from streamlit_funcs import st_nh3, common_params
from datetime import date,datetime
import plotly.express as px
import plotly.graph_objects as go


source_id = common_params.source_id

def run():
    st.set_page_config(layout="wide")
    st.markdown("# NH\u2083 Measurements")
    sidebar_configuration()
    

    # Initiate the state
    state = st.session_state
    initiate_state(state)

    # Select start_date, end_date, source
    data_selections(state)
    cm1,cm2 = st.columns(2)
    show_statistics(state)

    plot_stations(state)
    
    
    # st.map(state.sel_sdf)

def plot_station_map(state,cm):
    px.set_mapbox_access_token(open(".mapbox_token").read())
    fig = px.scatter_mapbox(state.sel_sdf,
                        lat=state.sel_sdf.lat,
                        lon=state.sel_sdf.lon,
                        hover_name="sid",
                        zoom=1)
    fig.update_layout(
            autosize=True,
            hovermode='closest',
            showlegend=False,
            # width=250,
            height=350,
            mapbox=dict(
                bearing=0,
                center=dict(
                    lat=56,
                    lon=13
                ),
                pitch=0,
                zoom=2,
                style='light'
            ),
            margin=dict(l=2, r=2, t=2, b=2)

        )
    cm.plotly_chart(fig, use_container_width=True)


def sidebar_configuration():
    st.sidebar.markdown("# NH\u2083-N Database:")
    st.sidebar.markdown(
        "NH\u2083-N measurements collected from:<ol><li>**EBAS** database</li> <li>**Denmark** monitoring"\
        +" network</li> <li>**France** monitoring network</li> <li>**Holland** monitoring network</li></ol>", unsafe_allow_html=True)
    st.sidebar.markdown('More to add...')
    st.sidebar.markdown('$Unit: \u03BCg-N/m^3$')

def plot_stations(state):
    select_1site(state)
    cm1,cm2 = st.columns([1,3])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=state.s1_df['ed'].values,
            y=state.s1_df['val'].values,
            name='Gaps',
        ))
    # fig = px.timeline(state.s1_df, x_start="st", x_end="ed", y="val") 
    fig.update_layout(
                   xaxis_title='Date',
                   yaxis_title='NH\u2083 (\u03BCg-N/m^3)',
                   margin=dict(l=2, r=2, t=2, b=2),
                #    width=1000,
                   height=380,
                   )
    
    cm2.plotly_chart(fig, use_container_width=True)
    plot_station_map(state,cm1)

def select_1site(state):
    dmc1,dmc2 = st.columns(2)  
    st_source = dmc1.radio('Station from:',source_id.keys(),index=0)
    if st_source=='All':
        sitename = dmc2.selectbox('Select station:',state.sel_stlist,)
    else:
        state.st_source = source_id[st_source]
        state.funcs.stations_from_source(state)
        sitename = dmc2.selectbox('Select station:',state.sel_source_stlist)
    state.selected_site = sitename.split(": ")[0]
    state.funcs.get_1site_data(state)



def data_selections(state):
    # Select start_date, end_date, source
    with st.expander('Select time period and sources:',expanded=True):
        state.c01, state.c02= st.columns([1,3])
        date_method = state.c01.radio('Choose date based on:',['Year','Date'],index=0)
        if date_method=='Year':
            selected_years = st.slider('Year range:',state.first_date.year,\
                state.last_date.year,(2017,2019))
            state.start_date,state.end_date = date(selected_years[0],1,1), date(selected_years[1],12,31)
        else:
            def_st,def_ed = date(2016,1,1), date(2019,12,31)
            state.c11, state.c12= st.columns(2)
            state.start_date = state.c11.date_input('Start date:', value=def_st,\
                min_value=state.first_date,max_value=state.last_date)
            state.end_date = state.c12.date_input('End date:',value=def_ed,\
                min_value=state.first_date,max_value=state.last_date)

        source = state.c02.multiselect('Select the data source:',source_id.keys(),\
            default=source_id.keys())
        state.source = [source_id[i] for i in source]

    if 'start_date0' not in state:
        load_data(state)
    else:
        state.logic1 = (state.start_date0 != state.start_date) or (state.end_date0!=state.end_date) or \
             (state.source0!=state.source)
        if state.logic1:
            load_data(state)

def show_statistics(state):
    state.funcs.calculate_statistics(state)
    dm1,dm2,dm3 = st.columns([4,1,1])
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    dm1.markdown('<p class="big-font">EBAS-'+str(state.tot_sts[0])+\
        ' stations, '+str(state.tot_perc[0])+'%; Denmark-'+str(state.tot_sts[1])\
        +' stations, '+str(state.tot_perc[1])+'%; France-'+str(state.tot_sts[2])+\
            ' stations, '+str(state.tot_perc[2])+'%; Holland-'+str(state.tot_sts[3])+\
            ' stations, '+str(state.tot_perc[3])+'%</p>',unsafe_allow_html=True)
    
    # cm.markdown('EBAS-**'+str(state.tot_sts[0])+'**, Denmark-**'+str(state.tot_sts[1])\
    #     +'**, France-**'+str(state.tot_sts[2])+'**',\
    #     unsafe_allow_html=True)
    

    dm2.download_button(
        label='Download data',
        data = state.sel_df.to_csv(index=False),
        file_name='selected_nh3_data.csv')
    dm3.download_button(
        label='Site meta',
        data = state.sel_sdf.to_csv(index=False),
        file_name='selected_nh3_site_meta.csv')  
    

def load_data(state):
    state.funcs.read_nh3_data(state)
    state.funcs.get_nh3_stations(state)
    
    state.start_date0,state.end_date0 = state.start_date,state.end_date
    state.source0 = state.source


def initiate_state(state):
    if 'funcs' not in state:
        state.funcs = st_nh3()
        state.nh3_db = state.funcs.db
        state.nh3_df = state.nh3_db.read_data()
        state.first_date = state.nh3_df['st'].min()
        state.last_date = state.nh3_df['ed'].max()
    state.autoload = True

if __name__ == '__main__':
    run()
