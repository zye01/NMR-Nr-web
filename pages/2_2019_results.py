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
    'DEHM_noBD': {'color': 'royalblue', 'width': 1, 'dash': 'dash'},
    'EMEP': {'color': 'orangered', 'width': 2},
    'EMEP_BD': {'color': 'orangered', 'width': 2},
    'EMEP_noBD': {'color': 'orangered', 'width': 1, 'dash': 'dash'},
    'MATCH': {'color': 'green', 'width': 2},
    'MATCH_BD': {'color': 'green', 'width': 2},
    'MATCH_noBD': {'color': 'green', 'width': 1, 'dash': 'dash'},
}
seasons = common_params.seasons
par_dict = common_params.par_dict

def run():

    st.title('Three models simulations in 2019')

    # Initiate the state
    state = st.session_state
    initiate_state(state)

    select_data(state,st.sidebar)

    plot_landuse(state,st.sidebar)
    # Plot the station map
    plot_station_map(state, st.sidebar)

    tab1, tab2 = st.tabs(['Time series','Diurnal cycle'])

    with tab1:
        c1, c2 = st.columns([1, 3])
        get_data_agg(state)
        # Time series
        if state.no_obs:
            st.markdown(f'##### No observations available at {state.sname} for {state.par} in 2019.')

        plot_time_series(state, st)
        plot_ts_diff(state,st)
        plot_ts_bias(state,st)
        plot_metrics(state,st)

    with tab2:
        get_diurnal_data(state)
        plot_diurnal_cycle(state)


def plot_metrics(state,cm):

    if state.no_obs:
        return

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
        height=400,
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
    for icase in all_simcases:
        df = state.df_agg_merge[[icase,'Obs']]
        df = df.dropna()
        metrics = common_funcs.calc_metrics(sim=df[icase], obs=df['Obs'])
        dict_metrics['Case'].append(icase)
        dict_metrics['MB'].append(metrics['MB'])
        dict_metrics['RMSE'].append(metrics['RMSE'])
        dict_metrics['Pearsonr'].append(metrics['Pearsonr'])
    
    state.df_metrics = pd.DataFrame.from_dict(dict_metrics)



def plot_landuse(state,cm):

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
    fig.update_layout(showlegend=False)
    # title in the middle
    fig.update_layout(title_x=0.5,title_xanchor='center',)
    # fig.update_layout(height=250)

    cm.plotly_chart(fig, use_container_width=True)



def plot_diurnal_cycle(state):
    c1, c2 = st.columns([1, 1])
    c1.markdown("<h2 style='text-align: center; color: black;'>Diurnal </h2>", unsafe_allow_html=True)
    c2.markdown("<h2 style='text-align: center; color: black;'>BD minus noBD </h2>", unsafe_allow_html=True)

    # Get the range of y-axis
    c1_range = get_range(state.df_dc_sn[state.dc_cases].values.flatten())
    c2_range = get_range(state.df_dc_sn_diff[models].values.flatten(),zero=True)


    # plot for the whole period
    diurnal_fig(state, c1, state.df_dc, state.dc_cases,y_range=c1_range)
    diurnal_fig(state, c2, state.df_dc_diff, models,diff=True,y_range=c2_range)
    
    # plot for each season
    for iseason, sseason in seasons.items():
        c1.markdown(f"<h4 style='text-align: center; color: black;'>{sseason}</h4>", unsafe_allow_html=True)
        c2.markdown(f"<h4 style='text-align: center; color: black;'>{f'{sseason} diff'}</h4>", unsafe_allow_html=True)
        df = state.df_dc_sn[state.df_dc_sn['Season']==iseason]
        diurnal_fig(state, c1, df, state.dc_cases, y_range=c1_range)
        df = state.df_dc_sn_diff[state.df_dc_sn_diff['Season']==iseason]
        diurnal_fig(state, c2, df, models,diff=True, y_range=c2_range)

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
    df = state.df_sim[['Hour']+all_simcases]
    state.df_dc_sim = df.groupby(['Hour']).mean().reset_index()
    df = state.df_sim[['Season','Hour']+all_simcases]
    state.df_dc_sn_sim = df.groupby(['Season','Hour']).mean().reset_index()
    


    if state.no_obs or state.dths.min() > 1:
        st.markdown(f'##### No observations for diurnal cycle available at {state.sname} for {state.par} in 2019.')
        
        state.df_dc = state.df_dc_sim
        state.df_dc_sn = state.df_dc_sn_sim
        state.dc_cases = all_simcases
    
    else:
        # Calculate diurnal cycle for the merged data, state.df_merge
        state.df_merge['Hour'] = state.df_merge['st'].dt.hour
        state.df_merge['Season'] = state.df_merge['st'].dt.month.apply(lambda x: 'DJF' if x in [12,1,2] else 'MAM' if x in [3,4,5] else 'JJA' if x in [6,7,8] else 'SON')
        df = state.df_merge[['Hour']+all_cases]
        state.df_dc = df.groupby(['Hour']).mean().reset_index()
        df = state.df_merge[['Season','Hour']+all_cases]
        state.df_dc_sn = df.groupby(['Season','Hour']).mean().reset_index()
        state.dc_cases = all_cases
    
    get_diurnal_diff(state)

def get_diurnal_diff(state):
    # Calculate the difference between BD and noBD for each model
    state.df_dc_diff = state.df_dc.copy()
    for model in models:
        state.df_dc_diff[model] = state.df_dc[f'{model}_BD'] - state.df_dc[f'{model}_noBD']
    state.df_dc_diff = state.df_dc_diff[['Hour']+models]

    state.df_dc_sn_diff = state.df_dc_sn.copy()
    for model in models:
        state.df_dc_sn_diff[model] = state.df_dc_sn[f'{model}_BD'] - state.df_dc_sn[f'{model}_noBD']
    state.df_dc_sn_diff = state.df_dc_sn_diff[['Season','Hour']+models]

def plot_ts_diff(state, cm):
    # The difference between BD and noBD for each model
    cm.markdown('### BD minus noBD for each model')

    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )

    for model in models:
        icase = f'{model}_BD'
        diff = state.df_agg_sim[icase] - state.df_agg_sim[f'{model}_noBD']
        fig.add_trace(go.Scatter
            (   x=state.df_agg_sim['Time'],
                y=diff,
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

    for icase in all_simcases:
    
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
    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )
    
    for icase in all_cases:
        ldict = line_props[icase]
        # Set the line style
        if state.df_agg_merge is None:
            if icase=='Obs':
                continue
            fig.add_trace(go.Scatter(
                        x=state.df_agg_sim['Time'],
                        y=state.df_agg_sim[icase],
                        name=icase,
                        line=ldict,
                    ))
        else:
            fig.add_trace(go.Scatter(
                        x=state.df_agg_merge['ed'],
                        y=state.df_agg_merge[icase],
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
    state.par = cm.selectbox('Select component', list(par_dict.keys()))
    state.station = cm.selectbox('Select station', state.stnames)
    state.sname, state.sid = state.station.split('(')
    state.sname = state.sname.strip()
    state.sid = state.sid[:-1].strip()
    state.soid = state.soid_dict[state.sid]
    

    # Load data
    load_data(state)

def get_data_agg(state):
    state.time_agg = st.selectbox('Time interval', ['Hourly','Daily','Monthly'],1)
    state.tagg = state.time_agg[0]

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



def load_stations(state):
    infile = os.path.join(datapath, 'sites.csv')
    state.sdf = pd.read_csv(infile, header=0)
    state.sids = state.sdf['sid'].values
    # build a library based on 'sid' and 'obs_id'
    state.soid_dict = dict(zip(state.sdf['sid'],state.sdf['obs_id']))
    state.snames = state.sdf['sname'].values
    state.stnames = [f'{sname} ({sid})' for sid,sname in zip(state.sids,state.snames)]
    state.sdf['size'] = 5
    # print(state.stnames)

def very_mean(array_like):
    if any(pd.isnull(array_like)):
        return np.nan
    else:
        return array_like.mean()

def load_data(state):
    # Load simulation data
    simfile = os.path.join(datapath,'models', f'All_models_2019_{state.par}.csv')
    df = pd.read_csv(simfile, header=0)
    df_sel = df[['Time','Case',state.sid]]
    # df_sel[df_sel[state.sid]<0] = np.nan
    state.df_sim = df_sel.pivot(index='Time', columns='Case', values=state.sid).reset_index()
    state.df_sim['Time'] = np.array(pd.to_datetime(state.df_sim['Time']))
    # set all negative values to nan
    # state.df_sim[models][state.df_sim[models]<0] = np.nan
    

    # Load merged data
    infile = os.path.join(datapath, 'merged', f'All_merged_2019_{state.par}.csv')
    df = pd.read_csv(infile, header=0)
    # replace all negative values to nan
    # df[df[all_cases]<0] = np.nan
    if state.soid not in df.site.values:
        state.df_merge = None
        state.df_agg_merge = None
        state.no_obs = True
        return
    
    state.no_obs = False
    # print(state.soid)
    sel_df = df[df.site==state.soid]
    # print(sel_df)
    state.dths = sel_df['dtH'].unique()
    state.df_merge = sel_df.drop(columns=['site','dtH'], axis=1)
    state.df_merge['st'] = np.array(pd.to_datetime(state.df_merge['st']))
    state.df_merge['ed'] = np.array(pd.to_datetime(state.df_merge['ed']))
    


def load_metrics(state):
    infile = os.path.join(datapath, 'All_metrics_2019.csv')
    state.df_mtr = pd.read_csv(infile, header=0)


def initiate_state(state):
    if 'runs' not in state:
        state.runs = ['BD','noBD']
        
    if 'models' not in state:
        state.models = models
    
    # state.autoload = True
    load_stations(state)
    load_metrics(state)


if __name__ == "__main__":
    run()