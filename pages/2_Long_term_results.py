import os
import pandas as pd
import numpy as np
import streamlit as st
# from datetime import date,datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_funcs import common_funcs, common_params
# import plotly.figure_factory as ff

# from streamlit_funcs.commons import common_params, common_funcs


st.set_page_config(layout="wide")
models = ['DEHM', 'EMEP', 'MATCH']

basepath = '.'
datapath = os.path.join(basepath, 'post_process', 'data')
all_simcases = [f'{model}_{run}' for model in models for run in ['BD', 'noBD']]
all_cases = all_simcases + ['Obs']
line_props = {
    'Obs': {'color': 'black', 'width': 3},
    'DEHM': {'color': 'royalblue', 'width': 2},
    'DEHM_BD': {'color': 'royalblue', 'width': 2},
    'DEHM_noBD': {'color': 'royalblue', 'width': 2, 'dash': 'dash'},
    'EMEP': {'color': 'orangered', 'width': 2},
    'EMEP_BD': {'color': 'orangered', 'width': 2},
    'EMEP_noBD': {'color': 'orangered', 'width': 2, 'dash': 'dash'},
    'MATCH': {'color': 'green', 'width': 2},
    'MATCH_BD': {'color': 'green', 'width': 2},
    'MATCH_noBD': {'color': 'green', 'width': 2, 'dash': 'dash'},
}
seasons = common_params.seasons
par_dict = common_params.par_dict

def run():

    st.title('Long term model runs 2010-2020')

    # Initiate the state
    state = st.session_state
    initiate_state(state)

    select_data(state,st.sidebar)

    # plot_landuse(state,st.sidebar)
    
    # Plot the station map
    plot_station_map(state, st.sidebar)

    tab1, tab2, tab3 = st.tabs(['Time series','Diurnal cycle', 'Metrics'])

    load_data(state)

    with tab1:
        get_data_agg(state)
        # Time series
        if state.no_obs:
            st.markdown(f'##### No observations available at {state.sname} for {state.par}.')

        plot_time_series(state, st)
        plot_ts_bias(state,st)
        plot_ts_diff(state,st)
        # plot_metrics(state,st)

    with tab2:
        get_diurnal_data(state)
        plot_diurnal_cycle(state)

    with tab3:
        plot_metrics(state,st)
        plot_heatmaps(state,st)

        if state.sid != 'All stations':
            plot_landuse(state,st)

def print_available_stations(state, cm):
    if state.sid == 'All stations':
        stnames = ', '.join(state.available_stations)
        totst = len(state.available_stations)
        cm.markdown(f'Observations are available for {totst} stations:')
        cm.markdown(f'{stnames}')
        
def plot_heatmaps(state,cm):
    if state.df_agg_merge is None:
        cm.markdown('### Correlations between models')
    else:
        cm.markdown('### Correlations between models and observations')
    cm.markdown('Note: diff = BD - noBD')


    corr = get_heatmap_correlations(state)
    corr = corr.round(2)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    # remove null in text labels
    labels = corr.mask(~mask).values
    labels = np.where(~mask, '', labels)

    fig = go.Figure(data=go.Heatmap(
            z=corr.mask(~mask),
            x=corr.columns,
            y=corr.columns,
            colorscale='Tropic',
            text=labels,
            texttemplate="%{text}",
            textfont=dict(size=15),
            zmin=-1,
            zmax=1,
            xgap=1,
            ygap=1,
            showscale=False,
        ),
        )
    
    fig.update_layout(
        margin=dict(l=2, r=2, t=2, b=2),
        width=700,
        height=500,
        font=dict(size=18),
        xaxis=dict(showline=False, title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=False, title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        # yaxis_autorange='reversed',
        )
    
    # move x-axis to the top
    # fig.update_xaxes(side='top')
    cm.plotly_chart(fig)    

def get_heatmap_correlations(state):
    model_cases = [f'{model}_{run}' for model in state.sel_models for run in ['BD','noBD','diff']]
    if state.df_agg_merge is not None:
        for model in state.sel_models:
            state.df_agg_merge[f'{model}_diff'] = state.df_agg_merge[f'{model}_BD'] - state.df_agg_merge[f'{model}_noBD']
        df = state.df_agg_merge[model_cases+['Obs']]
    else:
        df = state.df_agg_sim[model_cases]

    corr = df.corr()

    return corr



def plot_metrics(state,cm):

    if state.no_obs:
        return
    
    print_available_stations(state, cm)
    tot_obs = len(state.df_agg_merge)
    cm.markdown(f'Total number of observations: {tot_obs}')

    # Calculate metrics
    get_metrics(state)

    c1,c2,c3 = st.columns(3)
    c1.markdown('### MB')
    c2.markdown('### RMSE')
    c3.markdown('### Pearsonr')


    metrics_figure(state,'MB',c1)
    metrics_figure(state,'RMSE',c2)
    metrics_figure(state,'Pearsonr',c3)

def metrics_figure(state,mtr,cm):
    df = state.df_metrics[['Case',mtr]]
    # separate column Case into model and run
    df[['Model','Run']] = df['Case'].str.split('_',expand=True)
    df = df.drop(columns=['Case'])
    df = df.pivot(index='Model', columns='Run', values=mtr).reset_index()

    # Plot the barchart
    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )
    for run in ['BD','noBD']:
        fig.add_trace(go.Bar(
            x=df['Model'],
            y=df[run],
            name=run,
            text=df[run].round(2),
            marker_color='#ff5c5c' if run=='noBD' else '#40daff'
        ))

    fig.update_layout(
        margin=dict(l=2, r=2, t=2, b=2),
        # width=1200,
        height=300,
        font=dict(size=18),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        )
    
    cm.plotly_chart(fig, use_container_width=True)
    


def get_metrics(state):
    # Calculate MB, RMSE, and Correlation for each simulation case
    dict_metrics = {'Case':[],'MB':[],'RMSE':[],'Pearsonr':[]}
    for icase in state.aval_simcases:
        df = state.df_agg_merge[[icase,'Obs']]
        df = df.dropna()
        metrics = common_funcs.calc_metrics(sim=df[icase], obs=df['Obs'])
        dict_metrics['Case'].append(icase)
        dict_metrics['MB'].append(metrics['MB'])
        dict_metrics['RMSE'].append(metrics['RMSE'])
        dict_metrics['Pearsonr'].append(metrics['Pearsonr'])
    
    state.df_metrics = pd.DataFrame.from_dict(dict_metrics)



def plot_landuse(state,cm):
    cm.markdown(f'### Landuse types for DEHM and MATCH at {state.sname}')
    c1, c2 = cm.columns(2)

    # Plot for DEHM
    # c1.markdown('### DEHM')
    f_dehm = os.path.join(datapath, 'landuse_DEHM.csv')
    plot_landuse_pie(state,f_dehm,c1,'DEHM')

    # Plot for MATCH
    # c2.markdown('### MATCH')
    f_match = os.path.join(datapath, 'landuse_MATCH.csv')
    plot_landuse_pie(state,f_match,c2,'MATCH')

    

def plot_landuse_pie(state,infile,cm,label):
    df = pd.read_csv(infile, header=0)
    lu_cols = df.columns[1:]
    df = df[df.site==state.sid][lu_cols]
    df = pd.melt(df, var_name='Landuse', value_name='Percentage')

    # set the color
    colors = px.colors.qualitative.Set3
    color_map = dict(zip(lu_cols, colors[:len(lu_cols)]))

    fig = px.pie(df, values='Percentage', names='Landuse',title=label, color='Landuse', color_discrete_map=color_map)
    # hide the legend
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    fig.update_layout(showlegend=True)
    # title in the middle
    fig.update_layout(title_x=0.5,title_xanchor='center')
    # fig.update_layout(height=250)

    cm.plotly_chart(fig, use_container_width=True)



def plot_diurnal_cycle(state):
    c1, c2 = st.columns([1, 1])
    c1.markdown("<h2 style='text-align: center; color: black;'>Diurnal </h2>", unsafe_allow_html=True)
    c2.markdown("<h2 style='text-align: center; color: black;'>BD minus noBD </h2>", unsafe_allow_html=True)

    # Get the range of y-axis
    c1_range = get_range(state.df_dc_sn[state.dc_cases].values.flatten())
    c2_range = get_range(state.df_dc_sn_diff[state.aval_models].values.flatten(),zero=True)


    # plot for the whole period
    diurnal_fig(state, c1, state.df_dc, state.dc_cases,y_range=c1_range)
    diurnal_fig(state, c2, state.df_dc_diff, state.aval_models,diff=True,y_range=c2_range)
    
    # plot for each season
    for iseason, sseason in seasons.items():
        c1.markdown(f"<h4 style='text-align: center; color: black;'>{sseason}</h4>", unsafe_allow_html=True)
        c2.markdown(f"<h4 style='text-align: center; color: black;'>{f'{sseason} diff'}</h4>", unsafe_allow_html=True)
        df = state.df_dc_sn[state.df_dc_sn['Season']==iseason]
        diurnal_fig(state, c1, df, state.dc_cases, y_range=c1_range)
        df = state.df_dc_sn_diff[state.df_dc_sn_diff['Season']==iseason]
        diurnal_fig(state, c2, df, state.aval_models,diff=True, y_range=c2_range)

def get_range(arr,zero=False):
    min_val, max_val = arr.min(), arr.max()
    if zero:
        # include 0 in the range
        min_val = min(min_val,0)
        max_val = max(max_val,0)

    padding = 0.05 * (max_val - min_val)
    min_val -= padding
    max_val += padding

    return [min_val,max_val]


def diurnal_fig(state, cm, df, cases, title=None,diff=False,y_range=None):
    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )
    for icase in cases:
        fig.add_trace(go.Scatter
            (   x=df['Hour'],
                y=df[icase],
                name=icase,
                line=line_props[icase],
            )
        )
    
    if diff:
        # Plot a horizontal line at 0
        fig.add_shape(
            type="line",
            x0=df['Hour'].min(),
            y0=0,
            x1=df['Hour'].max(),
            y1=0,
            line=dict(
                color="black",
                width=1,
            ),
        )

    fig.update_layout(
        xaxis_title='Hour',
        yaxis_title=f'{state.par} ({par_dict[state.par]["units"]})',
        margin=dict(l=2, r=2, t=2, b=2),
        # width=1200,
        height=400,
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.4,
        xanchor="left",
        x=0.01,
        font=dict(size=13)
        ),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        )
    
    if title is not None:
        fig.update_layout(title=title,title_x=0.15, title_xanchor='left', title_y=0.95)

    if y_range is not None:
        fig.update_yaxes(range=y_range)
    
    cm.plotly_chart(fig, use_container_width=True)

    

def get_diurnal_data(state):
    # Calculate diurnal cycle for the simulation data, state.df_sim
    state.df_sim['Hour'] = state.df_sim['Time'].dt.hour
    # get the season
    state.df_sim['Season'] = state.df_sim['Time'].dt.month.apply(lambda x: 'DJF' if x in [12,1,2] else 'MAM' if x in [3,4,5] else 'JJA' if x in [6,7,8] else 'SON')
    # print(state.df_sim)
    df = state.df_sim[['Hour']+state.aval_simcases]
    state.df_dc_sim = df.groupby(['Hour']).mean().reset_index()
    df = state.df_sim[['Season','Hour']+state.aval_simcases]
    state.df_dc_sn_sim = df.groupby(['Season','Hour']).mean().reset_index()
    


    if state.no_obs or state.dths.min() > 1:
        st.markdown(f'##### No observations for diurnal cycle available at {state.sname} for {state.par}.')
        
        state.df_dc = state.df_dc_sim
        state.df_dc_sn = state.df_dc_sn_sim
        state.dc_cases = state.aval_simcases

    else:
        # Calculate diurnal cycle for the merged data, state.df_merge
        state.df_merge['Hour'] = state.df_merge['st'].dt.hour
        state.df_merge['Season'] = state.df_merge['st'].dt.month.apply(lambda x: 'DJF' if x in [12,1,2] else 'MAM' if x in [3,4,5] else 'JJA' if x in [6,7,8] else 'SON')
        df = state.df_merge[['Hour']+state.aval_cases]
        state.df_dc = df.groupby(['Hour']).mean().reset_index()
        df = state.df_merge[['Season','Hour']+state.aval_cases]
        state.df_dc_sn = df.groupby(['Season','Hour']).mean().reset_index()
        state.dc_cases = state.aval_cases
    
    get_diurnal_diff(state)

def get_diurnal_diff(state):
    # Calculate the difference between BD and noBD for each model
    state.df_dc_diff = state.df_dc.copy()
    for model in state.aval_models:
        state.df_dc_diff[model] = state.df_dc[f'{model}_BD'] - state.df_dc[f'{model}_noBD']
    state.df_dc_diff = state.df_dc_diff[['Hour']+state.aval_models]

    state.df_dc_sn_diff = state.df_dc_sn.copy()
    for model in state.aval_models:
        state.df_dc_sn_diff[model] = state.df_dc_sn[f'{model}_BD'] - state.df_dc_sn[f'{model}_noBD']
    state.df_dc_sn_diff = state.df_dc_sn_diff[['Season','Hour']+state.aval_models]

def plot_ts_diff(state, cm):
    # The difference between BD and noBD for each model
    cm.markdown('### BD minus noBD for each model')
    if state.sid == 'All stations':
        cm.markdown('Note: mean difference for all 11 stations')

    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )

    for model in state.aval_models:
        icase = f'{model}_BD'
        if icase not in state.df_agg_sim.columns:
            continue

        # print(state.df_agg_sim)
        fig.add_trace(go.Scatter
            (   x=state.df_agg_sim['Time'],
                y=state.df_agg_sim[f'{model}_diff'],
                name=model,
                line=line_props[icase],
            )
        )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=f'{state.par} difference ({par_dict[state.par]["units"]})',
        margin=dict(l=2, r=2, t=2, b=2),
        # width=1200,
        height=400,
        legend=dict(
        orientation="h",
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=0.01,
        font=dict(size=15)
        ),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        )
    
    cm.plotly_chart(fig, use_container_width=True)


def plot_ts_bias(state, cm):
    # The difference between BD and noBD for each model
    if state.df_agg_merge is None:
        return

    cm.markdown('### Bias (simulation minus observations)')

    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )

    # aval_cases = [icase for icase in all_cases if icase in state.df_agg_merge.columns]
    # aval_cases.remove('Obs')
    for icase in state.aval_simcases:
    
        diff = state.df_agg_merge[icase] - state.df_agg_merge['Obs']
        fig.add_trace(go.Scatter
            (   x=state.df_agg_merge['ed'],
                y=diff,
                name=icase,
                line=line_props[icase],
            )
        )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=f'{state.par} bias ({par_dict[state.par]["units"]})',
        margin=dict(l=2, r=2, t=2, b=2),
        # width=1200,
        height=400,
        legend=dict(
        orientation="h",
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=0.01,
        font=dict(size=15)
        ),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        )
    
    cm.plotly_chart(fig, use_container_width=True)



def plot_time_series(state, cm):
    cm.markdown('### Time series of all runs')
    print_available_stations(state, cm)

    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )
    
    # aval_cases = [icase for icase in all_cases if icase in state.df_agg_sim.columns]
    for icase in state.aval_cases:
        ldict = line_props[icase]
        # Set the line style
        if icase == 'Obs' and state.df_agg_merge is not None:
            fig.add_trace(go.Scatter(
                        x=state.df_agg_merge['ed'],
                        y=state.df_agg_merge[icase],
                        name=icase,
                        line=ldict,
                    ))
        elif state.sid == 'All stations':
            fig.add_trace(go.Scatter(
                        x=state.df_agg_merge['ed'],
                        y=state.df_agg_merge[icase],
                        name=icase,
                        line=ldict,
                    ))
        elif icase != 'Obs':
            fig.add_trace(go.Scatter(
                        x=state.df_agg_sim['Time'],
                        y=state.df_agg_sim[icase],
                        name=icase,
                        line=ldict,
                    ))
            
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=f'{state.par} ({par_dict[state.par]["units"]})',
        margin=dict(l=2, r=2, t=2, b=2),
        # width=1200,
        height=400,
        legend=dict(
        orientation="h",
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=0.01,
        font=dict(size=15)
        ),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        )

    cm.plotly_chart(fig, use_container_width=True)  
        


def plot_station_map(state,cm):
    px.set_mapbox_access_token(open(".mapbox_token").read())
    fig = px.scatter_mapbox(state.sdf,
                        lat=state.sdf.lat,
                        lon=state.sdf.lon,
                        hover_name="sname",
                        size='size',
                        size_max=8,
                        zoom=0)
    fig.update_layout(
            autosize=True,
            hovermode='x',
            height=250,
            mapbox=dict(
                bearing=0,
                center=dict(
                    lat=56,
                    lon=11
                ),
                pitch=0,
                zoom=3,
                style='light'
            ),
            margin=dict(l=2, r=2, t=2, b=2),
        )
    cm.plotly_chart(fig, use_container_width=True)

def select_data(state,cm):
    state.sel_models = cm.multiselect('Select models', models, ['DEHM','EMEP','MATCH'])
    state.time_agg = cm.selectbox('Time interval', ['Hourly','Daily','Monthly'],1)
    state.par = cm.selectbox('Select component', list(par_dict.keys()))
    state.par_sunit = par_dict[state.par]['sunit']

    # Select station
    state.st_options = ['All stations'] + state.stnames
    state.station = cm.selectbox('Select station', state.st_options)
    state.yrs = cm.slider('Select year', 2010, 2020, (2018, 2019))

    if state.station == 'All stations':
        state.sname = 'All stations'
        state.sid = 'All stations'
        state.soid = 'All stations'

    else:  
        state.sname, state.sid = state.station.split('(')
        state.sname = state.sname.strip()
        state.sid = state.sid[:-1].strip()
        state.soid = state.soid_dict[state.sid]


def get_data_agg(state):
    
    if state.time_agg == 'Hourly':
        state.tagg = 'H'
    elif state.time_agg == 'Daily':
        state.tagg = 'D'
    else:
        state.tagg = 'MS'
    # state.tagg = state.time_agg[0]

    # Load data aggregation
    if state.tagg == 'H':
        state.df_agg_sim = state.df_sim
        if state.df_merge is not None and state.dths.min() == 1:
            state.df_agg_merge = state.df_merge
        else:
            state.df_agg_merge = None
    else:
        state.df_agg_sim = state.df_sim.resample(state.tagg, on='Time').mean().reset_index()
        if state.df_merge is not None:
            state.df_agg_merge = state.df_merge.resample(state.tagg, on='ed').mean().reset_index()
        else:
            state.df_agg_merge = None

    # calculate BD and noBD difference for each model
    for imodel in state.sel_models:
        state.df_agg_sim[f'{imodel}_diff'] = state.df_agg_sim[f'{imodel}_BD'] - state.df_agg_sim[f'{imodel}_noBD']


def load_stations(state):
    infile = os.path.join(datapath, 'sites.csv')
    state.sdf = pd.read_csv(infile, header=0)
    state.sids = state.sdf['sid'].values
    # build a library based on 'sid' and 'obs_id'
    state.soid_dict = dict(zip(state.sdf['sid'],state.sdf['obs_id']))
    state.sname_dict = dict(zip(state.sdf['obs_id'],state.sdf['sname']))
    state.soids = state.sdf['obs_id'].values
    state.snames = state.sdf['sname'].values
    state.stnames = [f'{sname} ({sid})' for sid,sname in zip(state.sids,state.snames)]
    state.sdf['size'] = 5
    # print(state.stnames)

def very_mean(array_like):
    if any(pd.isnull(array_like)):
        return np.nan
    else:
        return array_like.mean()

def load_simulations(state):
    # Load simulation data
    dfs = []
    for yr in range(state.yrs[0],state.yrs[1]+1):
        for imodel in state.sel_models:
            simfile = os.path.join(datapath,'models', f'{imodel}_{yr}_{state.par}.csv')
            if not os.path.exists(simfile):
                continue
            df = pd.read_csv(simfile, header=0)
            # Calculate mean of all station columns to get the 'All stations' column
            df['All stations'] = df[state.sids].apply(very_mean, axis=1)
            df = df[['Time','Case',state.sid]]
            dfs.append(df)
    state.df_sim = pd.concat(dfs)
    # print(state.df_sim)
    state.df_sim['Time'] = np.array(pd.to_datetime(state.df_sim['Time']))
    state.aval_simcases = list(state.df_sim['Case'].unique())
    state.df_sim = state.df_sim.pivot(index='Time', columns='Case', values=state.sid).reset_index()
    state.aval_models = [model for model in state.sel_models if f'{model}_BD' in state.aval_simcases]


def load_data(state):
    load_simulations(state)
    load_merged_data(state)


def load_merged_data(state):
    # Load merged data
    dfs = []
    for yr in range(state.yrs[0],state.yrs[1]+1):
        mergefile = os.path.join(datapath,'merged', f'All_merged_{yr}_{state.par}_{state.par_sunit}.csv')
        if not os.path.exists(mergefile):
            print(f'{mergefile} does not exist')
            continue
        df = pd.read_csv(mergefile, header=0)
        df['ed'] = np.array(pd.to_datetime(df['ed']))
        dfs.append(df)
    # print(dfs)
    df_merge = pd.concat(dfs)
    if df_merge.empty or ((state.soid not in df_merge.site.values) & (state.soid != 'All stations')):
        state.df_merge = None
        state.df_agg_merge = None
        state.no_obs = True
        return

    state.no_obs = False

    # df_merge['st'] = np.array(pd.to_datetime(df_merge['st'],format='ISO8601'))
    df_merge['st'] = np.array(pd.to_datetime(df_merge['st'],infer_datetime_format=True))
    # df_merge['ed'] = np.array(pd.to_datetime(df_merge['ed'],format='ISO8601'))
    df_merge['ed'] = np.array(pd.to_datetime(df_merge['ed'],infer_datetime_format=True))
    
    if state.soid == 'All stations':
        sel_df = df_merge.copy()
        state.osites = sel_df['site'].unique()
        state.dths = sel_df['dtH'].unique()
        # Calculate mean by 'st' and 'ed'
        sel_df = sel_df[['st','ed','Obs']+state.aval_simcases]
        state.df_merge = sel_df.groupby(['st','ed']).mean().reset_index()
    else:
        sel_df = df_merge[df_merge.site==state.soid]
        state.osites = sel_df['site'].unique()
        state.dths = sel_df['dtH'].unique()
        state.df_merge = sel_df.drop(columns=['site','dtH'], axis=1)
    # print(state.df_merge)
    state.aval_cases = ['Obs'] + state.aval_simcases

    # get the station names, convert state.osites to station names
    state.available_stations = [state.sname_dict[site] for site in state.osites]



def initiate_state(state):
    if 'runs' not in state:
        state.runs = ['BD','noBD']
        
    if 'models' not in state:
        state.models = models
    
    # state.autoload = True
    load_stations(state)
    # load_metrics(state)


if __name__ == "__main__":
    run()