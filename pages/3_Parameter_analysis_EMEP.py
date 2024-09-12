import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_funcs import common_params, common_funcs

st.set_page_config(layout="wide")
datapath = './post_process/data'
# models = ['DEHM', 'EMEP', 'MATCH']
sites = ['DK-09','NL10131','SE0022R']
years = [2018,2019]
cases = {'BDT': 'EMEP_BD', 'BDF': 'EMEP_noBD'}
par_dict = common_params.par_dict
seasons = common_params.seasons

def run():

    st.title('EMEP parameterization')

    state = st.session_state
    initiate_state(state)
    
    # Sidebar configuration
    select_data(state, st.sidebar)

    tab1, tab2 = st.tabs(['Time series','Diurnal cycle'])

    with tab1:
        get_data_agg(state,st)
        plot_ts(state,st)

    with tab2:
        get_data_diurnal(state)
        plot_diurnal(state,st)

def plot_diurnal(state,cm):
    c1, c2 = cm.columns(2)
    c1.markdown("<h2 style='text-align: center; color: black;'>Diurnal </h2>", unsafe_allow_html=True)
    c2.markdown("<h2 style='text-align: center; color: black;'>BDT minus BDF </h2>", unsafe_allow_html=True)

    # Get the common y ranges for all plots
    c1_y1_range, c1_y2_range = align_yaxis(np.array(state.df_diurnal_season[state.cases_1]).flatten(), np.array(state.df_diurnal_season[state.cases_2]).flatten())
    c2_y1_range, c2_y2_range = align_yaxis(np.array(state.df_diurnal_season[state.sel_1]).flatten(), np.array(state.df_diurnal_season[state.sel_2]).flatten())


    # Plot for the whole period
    lineplot(state, c1, state.df_diurnal, 'Hour', state.cases_all,y1_range=c1_y1_range,y2_range=c1_y2_range)
    lineplot(state, c2, state.df_diurnal, 'Hour', state.cases_diff,diff=True,y1_range=c2_y1_range,y2_range=c2_y2_range)

    # Plot for the seasons
    for iseason, sseason in seasons.items():
        c1.markdown(f"<h4 style='text-align: center; color: black;'>{sseason}</h4>", unsafe_allow_html=True)
        c2.markdown(f"<h4 style='text-align: center; color: black;'>{f'{sseason} diff'}</h4>", unsafe_allow_html=True)
        df = state.df_diurnal_season[state.df_diurnal_season['Season']==iseason]
        lineplot(state, c1, df, 'Hour', state.cases_all,y1_range=c1_y1_range,y2_range=c1_y2_range)
        lineplot(state, c2, df, 'Hour', state.cases_diff,diff=True,y1_range=c2_y1_range,y2_range=c2_y2_range)



def get_data_diurnal(state):
    state.df_merge['Hour'] = state.df_merge['Time'].dt.hour
    state.df_merge['Season'] = state.df_merge['Time'].dt.month.apply(lambda x: 'DJF' if x in [12,1,2] else 'MAM' if x in [3,4,5] else 'JJA' if x in [6,7,8] else 'SON')

    df = state.df_merge[['Hour']+state.cases_all+state.cases_diff]
    state.df_diurnal = df.groupby(['Hour']).mean().reset_index()

    df = state.df_merge[['Hour','Season']+state.cases_all+state.cases_diff]
    state.df_diurnal_season = df.groupby(['Hour','Season']).mean().reset_index()

def plot_ts(state,cm):
    # Plot the time series of all cases
    st.markdown('## Time series of all cases')
    lineplot(state, cm, state.df_agg_merge, 'Time', state.cases_all)

    # Plot the difference between BDT and BDF
    st.markdown('## Difference between BDT and BDF (BDT - BDF)')
    lineplot(state, cm, state.df_agg_merge, 'Time', state.cases_diff,diff=True)


def lineplot(state, cm, df, xlabel, cases,diff=False,title=None,y1_range=None,y2_range=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.update_layout(
        template = 'plotly_white'
    )

    y1cols,y2cols = [],[]

    for icase in cases:
        props = setup_props(state,icase)
        
        fig.add_trace(
            go.Scatter(
            x=df[xlabel],
            y=df[icase],
            name=icase,
            line=props['line']),
            secondary_y=props['secondary_y'],
        )
        
        if props['secondary_y']:
            y2cols.append(icase)
        else:
            y1cols.append(icase)
        # y2cols.append(icase) if props['secondary_y'] else y1cols.append(icase)

    if diff:
        # Plot a horizontal line at 0
        fig.add_shape(
            type="line",
            x0=df[xlabel].min(),
            y0=0,
            x1=df[xlabel].max(),
            y1=0,
            line=dict(
                color="black",
                width=1,
            ),
        )


    fig.update_layout(
        xaxis_title=xlabel,
        margin=dict(l=2, r=2, t=2, b=2),
        # width=1200,
        height=400,
        legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="left",
        x=0.01,
        font=dict(size=15)
        ),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        )
    
    if y1_range is None:
        y1_arr,y2_arr = np.array(df[y1cols]).flatten(),np.array(df[y2cols]).flatten()
        y1_range, y2_range = align_yaxis(y1_arr, y2_arr)

    y1_title_text = f'{state.sel_1} ({par_dict[state.sel_1]["units"]})' if state.sel_1 in par_dict.keys() else state.sel_1
    y2_title_text = f'{state.sel_2} ({par_dict[state.sel_2]["units"]})' if state.sel_2 in par_dict.keys() else state.sel_2
    # cunit = par_dict[state.sel_1]['units']
    fig.update_yaxes(title_text=y1_title_text, secondary_y=False,range=y1_range)
    fig.update_yaxes(title_text=y2_title_text, secondary_y=True,range=y2_range,\
                     showline=True, linewidth=1, linecolor='lightgrey',\
                     ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False)


    if title is not None:
        fig.update_layout(title=title,title_x=0.15, title_xanchor='left', title_y=0.95)
    
    cm.plotly_chart(fig, use_container_width=True)

    if cm.button('Download data'):
        cm.write(df)

    
def align_yaxis(y1_arr, y2_arr):
    y1_min, y1_max = y1_arr.min(), y1_arr.max()
    y1_max = 0 if y1_max < 0 else y1_max # set y1_max to 0
    y1_min = 0 if y1_min > 0 else y1_min # set y1_min to 0
    y1_padding = 0.1*(y1_max - y1_min)
    y1_max += y1_padding
    y1_min -= y1_padding

    y2_min, y2_max = y2_arr.min(), y2_arr.max()
    y2_max = 0 if y2_max < 0 else y2_max # set y2_max to 0
    y2_min = 0 if y2_min > 0 else y2_min # set y2_min to 0
    y2_padding = 0.1*(y2_max - y2_min)
    y2_max += y2_padding
    y2_min -= y2_padding

    y1_ratio = (0-y1_min)/(y1_max - y1_min)
    y2_ratio = (0-y2_min)/(y2_max - y2_min)

    # Update y1 and y2 range to align the zero
    if y1_ratio < y2_ratio:
        rdiff = y2_ratio - y1_ratio
        y1_min -= rdiff * (y1_max - y1_min)
        y2_max += rdiff * (y2_max - y2_min)
    else:
        rdiff = y1_ratio - y2_ratio
        y1_max += rdiff * (y1_max - y1_min)
        y2_min -= rdiff * (y2_max - y2_min)

    y1_range = [y1_min, y1_max]
    y2_range = [y2_min, y2_max]

    return y1_range, y2_range
    


def setup_props(state,icase):
    line_props = {}
    if 'BDT' in icase:
        line_props['width'] = 2
    elif 'BDF' in icase:
        line_props['width'] = 1
        line_props['dash'] = 'dash'
    else:
        line_props['width'] = 2

    if state.sel_2 in icase:
        line_props['color'] = 'blue'
        secondary_y = True
    else:
        line_props['color'] = 'red'
        secondary_y = False
    
    props = {'line': line_props, 'secondary_y': secondary_y}
    return props

def get_data_agg(state,cm):
    state.time_agg = cm.selectbox('Time interval', ['Hourly','Daily','Monthly'],1)
    state.tagg = state.time_agg[0]

    # state.df_merge['Time'] = pd.to_datetime(state.df_merge['Time'])
    # print(state.df_merge)
    if state.tagg == 'H':
        state.df_agg_merge = state.df_merge.copy()
    else:
        state.df_agg_merge = state.df_merge.resample(state.tagg, on='Time').mean().reset_index()

def select_data(state, cm):
    state.ysel = cm.selectbox('Select year', years+['All'])
    sname = cm.selectbox('Select site', state.stnames)
    state.sid = sname.split(' ')[-1][1:-1]
    state.sel_1 = cm.selectbox('Select first component (red)', list(par_dict.keys()) +state.cols_reorder)
    state.sel_2 = cm.selectbox('Select second component (blue)', state.cols_reorder+list(par_dict.keys()))
    state.cases_1, state.cases_2 = [f'{case}_{state.sel_1}' for case in cases.keys()] , [f'{case}_{state.sel_2}' for case in cases.keys()]
    state.cases_all = state.cases_1 + state.cases_2
    state.cases_diff = [state.sel_1, state.sel_2]

    load_data(state)

def load_data(state):
    # Load the concentrations
    df1 = load_data_each(state, state.sel_1)
    df2 = load_data_each(state, state.sel_2)
    # load_data_concentrations(state)

    # Load the parameters
    # load_data_parameters(state)

    state.df_merge = pd.merge(df1,df2,on='Time')

def load_data_each(state,par):
    if par in par_dict.keys():
        df = load_data_concentrations(state,par)
    else:
        df = load_data_parameters(state,par)
    return df

def load_data_concentrations(state,par):
    if state.ysel == 'All':
        # merge all years
        for year in years:
            df0 = read_concentration_year(state,year,par)
            if year == years[0]:
                state.df_conc = df0.copy()
            else:
                state.df_conc = pd.concat([state.df_conc,df0],axis=0)
    else:
        year = state.ysel
        state.df_conc = read_concentration_year(state,year,par)
    
    state.df_conc[par] = state.df_conc[f'BDT_{par}'] - state.df_conc[f'BDF_{par}']
    return state.df_conc
    
def load_data_parameters(state,par):
    if state.ysel == 'All':
        # merge all years
        for year in years:
            df0 = read_parameter_year(state,year,par)
            if year == years[0]:
                state.df_par = df0.copy()
            else:
                state.df_par = pd.concat([state.df_par,df0],axis=0)
    else:
        year = state.ysel
        state.df_par = read_parameter_year(state,year,par)
    
    state.df_par[par] = state.df_par[f'BDT_{par}'] - state.df_par[f'BDF_{par}']
    return state.df_par

def read_parameter_year(state,year,par):
    for icase in cases.keys():
        infile = os.path.join(datapath,'EMEP_CSV',f'{state.sid}_{icase}{year}.csv')
        df0 = pd.read_csv(infile,sep=',',skiprows=1,names=state.cols)
        # merge date and hour to datetime
        df0['Time'] = np.array(pd.to_datetime(df0['date'] + ' ' + df0['hour'],format='%d/%m/%Y %H:%M'))
        df0 = df0[['Time',par]]
        df0[par] = pd.to_numeric(df0[par],errors='coerce')
        df0 = df0.rename(columns={par: f'{icase}_{par}'})
        if icase == 'BDT':
            df = df0.copy()
        else:
            df = pd.merge(df,df0,on='Time')
    
    return df

def read_concentration_year(state,year,par):
    infile = os.path.join(datapath,'models',f'EMEP_{year}_{par}.csv')
    df = pd.read_csv(infile)
    df = df[['Time','Case',state.sid]]
    df = df.pivot(index='Time',columns='Case',values=state.sid).reset_index()
    df['Time'] = np.array(pd.to_datetime(df['Time']))
    df = df.rename(columns={'EMEP_BD': f'BDT_{par}', 'EMEP_noBD': f'BDF_{par}'})
    return df




def read_file(state,infile):

    df = pd.read_csv(infile,sep=',',skiprows=1,names=state.cols)
    return df

def initiate_state(state):
    infile = os.path.join(datapath,'EMEP_CSV','DK-09_BDF2019.csv')
    with open(infile,'r') as f:
        lines = f.readlines()
        state.cols = lines[0][:-1].split(';')

    # Reorder the parameters
    cols = state.cols[2:]
    pars1 = ['Vg3NH3','Rsur','Gns']
    pars2 = ['FstO3','Fphen','LC','NH3_ppb']
    pars12 = pars1 + pars2
    state.cols_reorder = pars1 + [i for i in cols if i not in pars12] + pars2

    # Get the station names
    sdf = pd.read_csv(os.path.join(datapath,'sites.csv'))
    stname_dict = dict(zip(sdf['sid'],sdf['sname']))
    state.stnames = [f'{stname_dict[i]} ({i})' for i in sites]


if __name__ == "__main__":
    run()