import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_funcs import common_params, common_funcs

st.set_page_config(layout="wide")
datapath = './post_process/data'
models = ['EMEP', 'MATCH']
sites = ['DK-09','NL10131','SE0022R']
years = [2018,2019]
cases = ['BD', 'noBD']
par_dict = common_params.par_dict
seasons = common_params.seasons
parameters_dict = common_params.parameters

def run():

    st.title('Parameterization')

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

    # update_selections(state)

def plot_diurnal(state,cm):
    c1, c2 = cm.columns(2)
    c1.markdown("<h2 style='text-align: center; color: black;'>Diurnal </h2>", unsafe_allow_html=True)
    c2.markdown("<h2 style='text-align: center; color: black;'>BD minus noBD </h2>", unsafe_allow_html=True)

    # Get the common y ranges for all plots
    if len(state.cases_1) > 0 and len(state.cases_2) > 0:
        c1_y1_range, c1_y2_range = align_yaxis(np.array(state.df_diurnal_season[state.cases_1]).flatten(), np.array(state.df_diurnal_season[state.cases_2]).flatten())
    else:
        c1_y1_range, c1_y2_range = None, None
    
    sel_y1_diff = [i for i in state.cases_diff if state.sel_1 in i]
    sel_y2_diff = [i for i in state.cases_diff if state.sel_2 in i]
    if len(sel_y1_diff) > 0 and len(sel_y2_diff) > 0:
        c2_y1_range, c2_y2_range = align_yaxis(np.array(state.df_diurnal_season[sel_y1_diff]).flatten(), np.array(state.df_diurnal_season[sel_y2_diff]).flatten())
    else:
        c2_y1_range, c2_y2_range = None, None

    # print(state.df_diurnal)
    # Plot for the whole period
    lineplot(state, c1, state.df_diurnal, 'Hour', state.cases_all,y1_range=c1_y1_range,y2_range=c1_y2_range, figkey='diurnal_all')
    lineplot(state, c2, state.df_diurnal, 'Hour', state.cases_diff,diff=True,y1_range=c2_y1_range,y2_range=c2_y2_range, figkey='diurnal_diff_all')

    # Plot for the seasons
    for iseason, sseason in seasons.items():
        c1.markdown(f"<h4 style='text-align: center; color: black;'>{sseason}</h4>", unsafe_allow_html=True)
        c2.markdown(f"<h4 style='text-align: center; color: black;'>{f'{sseason} diff'}</h4>", unsafe_allow_html=True)
        df = state.df_diurnal_season[state.df_diurnal_season['Season']==iseason]
        lineplot(state, c1, df, 'Hour', state.cases_all,y1_range=c1_y1_range,y2_range=c1_y2_range, figkey=f'diurnal_{iseason}')
        lineplot(state, c2, df, 'Hour', state.cases_diff,diff=True,y1_range=c2_y1_range,y2_range=c2_y2_range, figkey=f'diurnal_diff_{iseason}')



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
    lineplot(state, cm, state.df_agg_merge, 'Time', state.cases_all,figkey='ts_all',aligh_axis=False)

    # Plot the difference between BDT and BDF
    st.markdown('## Difference between BD and noBD (BD - noBD)')
    lineplot(state, cm, state.df_agg_merge, 'Time', state.cases_diff,diff=True,figkey='ts_diff')


def lineplot(state, cm, df, xlabel, cases,diff=False,title=None,aligh_axis=True,y1_range=None,y2_range=None, figkey=None):
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
            line=props['line'],
            opacity=0.7,
            ),
            
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
        margin=dict(l=20, r=20, t=20, b=20),
        # width=1200,
        height=600,
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="left",
        x=0.01,
        font=dict(size=15)
        ),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',mirror=True,\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',mirror=True,\
            ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False),
        )
    
    y1_title_text = f'{state.sel_1} ({par_dict[state.sel_1]["units"]})' if state.sel_1 in par_dict.keys() else state.sel_1
    y2_title_text = f'{state.sel_2} ({par_dict[state.sel_2]["units"]})' if state.sel_2 in par_dict.keys() else state.sel_2

    if len(y1cols) == 0 or len(y2cols) == 0:
        aligh_axis = False

    if aligh_axis:
        if y1_range is None:
            y1_arr,y2_arr = np.array(df[y1cols]).flatten(),np.array(df[y2cols]).flatten()
            y1_range, y2_range = align_yaxis(y1_arr, y2_arr)
        
        fig.update_yaxes(title_text=y1_title_text, secondary_y=False,range=y1_range,mirror=True)
        fig.update_yaxes(title_text=y2_title_text, secondary_y=True,range=y2_range,mirror=True,\
                        showline=True, linewidth=1, linecolor='lightgrey',\
                        ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False)

    else:
        fig.update_yaxes(title_text=y1_title_text, secondary_y=False,mirror=True)
        fig.update_yaxes(title_text=y2_title_text, secondary_y=True,mirror=True,\
                        showline=True, linewidth=1, linecolor='lightgrey',\
                        ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False)

    if title is not None:
        fig.update_layout(title=title,title_x=0.15, title_xanchor='left', title_y=0.95)
    
    cm.plotly_chart(fig, use_container_width=True, key=figkey+'_chart')

    cm.download_button(
        label='Download plot data',
        data = df.to_csv(index=False),
        key=figkey,
        file_name=f'Parameter_data_{figkey}.csv')

    
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

    if '_noBD' in icase:
        line_props['dash'] = 'dot'

    icase_split = icase.split('_')
    par = icase_split[-1]

    if par==state.sel_2:
        line_props['width'] = 3
        if 'MATCH' in icase:
            # line_props['color'] = '#b8e186'  # light green
            line_props['color'] = '#80cdc1'  # light cyan
        elif 'EMEP' in icase:
            # line_props['color'] = '#f1b6da' # light pink
            line_props['color'] = '#dfc27d' # light brown
        secondary_y = True
    elif par==state.sel_1:
        line_props['width'] = 2
        if 'MATCH' in icase:
            # line_props['color'] = '#4dac26'  # dark green
            line_props['color'] = '#018571' # dark cyan
        elif 'EMEP' in icase:
            # line_props['color'] = '#d01c8b' # dark pink
            line_props['color'] = '#a6611a' # dark brown
        secondary_y = False
    else:
        st.warning(f'Parameter {par} not found in the selection')

    # use different line styles for EMEP (solid) and MATCH (line with symbol) models
    # if 'MATCH' in icase:
    #     marker_props['symbol'] = 'circle'
    # elif 'EMEP' in icase:
    #     marker_props['symbol'] = 'line-ns'
    
    
    props = {'line': line_props, 'secondary_y': secondary_y}
    # print(icase,props)
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
    state.year_list = years+['All']
    cm.selectbox('Select year', state.year_list, key='ysel')
    # state.ysel_i = state.year_list.index(state.ysel)
    state.years = years if state.ysel == 'All' else [int(state.ysel)]
    
    cm.selectbox('Select site', state.stnames, key='p3_sname')
    # state.p3_sname_i = state.stnames.index(state.p3_sname)
    state.p3_sid = state.p3_sname.split(' ')[-1][1:-1]
    
    cm.multiselect('Select models', models, default=models, key='models')
    
    cm.selectbox('Select first component', state.allpars1, key='sel_1')
    # state.sel_1_i = state.allpars1.index(state.sel_1)
    if 'Note' in parameters_dict[state.sel_1].keys():
        cm.markdown(parameters_dict[state.sel_1]["Note"])
    
    cm.selectbox('Select second component', options=state.allpars, key='sel_2')
    # state.sel_2_0 = state.sel_2
    # cm.markdown(f'{state.sel_2}, {state.sel_2_0}')
    # cm.markdown(f'{state.count}')
    # state.sel_2_i = state.allpars.index(state.sel_2)
    if 'Note' in parameters_dict[state.sel_2].keys():
        cm.markdown(parameters_dict[state.sel_2]["Note"])

    load_data(state)

def load_data(state):
    # get state.cases_1
    state.cases_1, state.cases_2 = [],[]
    res = {}
    diff_counts = {'sel1':{}, 'sel2':{}}
    
    for imodel in state.models:
        diff_counts['sel1'][imodel] = 0
        diff_counts['sel2'][imodel] = 0
        for icase in cases:
            # Get all available parameters for the model
            allpars = []
            for par in [state.sel_1,state.sel_2]:
                if imodel in parameters_dict[par].keys():
                    allpars.append(parameters_dict[par][imodel])
            if len(allpars) == 0:
                continue

            # Get the data for the model and case
            if len(state.years) == 1:
                iyear = state.years[0]
                infile = os.path.join(datapath,'parameters',f'{iyear}_{state.p3_sid}_{imodel}_{icase}.csv')
                df = pd.read_csv(infile)
                # get the overlap parameters with df.columns
                overlap_pars = list(set(allpars).intersection(df.columns))
                df = df[['Time']+overlap_pars]
            else:
                for iyear in state.years:
                    infile = os.path.join(datapath,'parameters',f'{iyear}_{state.p3_sid}_{imodel}_{icase}.csv')
                    df0 = pd.read_csv(infile)
                    overlap_pars = list(set(allpars).intersection(df0.columns))
                    df0 = df0[['Time']+overlap_pars]
                    if iyear == state.years[0]:
                        df = df0.copy()
                    else:
                        df = pd.concat([df,df0],axis=0)

            if 'Time' not in res.keys():
                res['Time'] = df['Time']
            # check if the parameter is in the file for state.sel_1
            if imodel in parameters_dict[state.sel_1].keys():
                ipar = parameters_dict[state.sel_1][imodel]
                if ipar in df.columns:
                    state.cases_1.append(f'{imodel}_{icase}_{state.sel_1}')
                    res[f'{imodel}_{icase}_{state.sel_1}'] = df[ipar]
                    diff_counts['sel1'][imodel] += 1
            
            # check if the parameter is in the file for state.sel_2
            if imodel in parameters_dict[state.sel_2].keys():
                ipar = parameters_dict[state.sel_2][imodel]
                if ipar in df.columns:
                    state.cases_2.append(f'{imodel}_{icase}_{state.sel_2}')
                    res[f'{imodel}_{icase}_{state.sel_2}'] = df[ipar]
                    diff_counts['sel2'][imodel] += 1


    state.df_merge = pd.DataFrame(res)
    state.df_merge['Time'] = pd.to_datetime(state.df_merge['Time'])
    state.cases_all = state.cases_1 + state.cases_2
    
    # Get the difference between BDT and BDF
    state.cases_diff = []
    # diff_res = {'Time': state.df_merge['Time']}
    for i,ipar in enumerate([state.sel_1,state.sel_2]):
        for imodel in state.models:
            if diff_counts[f'sel{i+1}'][imodel] == 2:
                state.cases_diff.append(f'{imodel}_{ipar}')
                state.df_merge[f'{imodel}_{ipar}'] = state.df_merge[f'{imodel}_BD_{ipar}'] - state.df_merge[f'{imodel}_noBD_{ipar}']


def initiate_state(state):
    state.allpars1 = list(parameters_dict.keys())
    state.allpars1.remove('None')

    state.allpars = ['None'] + state.allpars1

    # Get the station names
    sdf = pd.read_csv(os.path.join(datapath,'sites.csv'))
    stname_dict = dict(zip(sdf['sid'],sdf['sname']))
    state.stnames = [f'{stname_dict[i]} ({i})' for i in sites]
    # if 'ysel_i0' not in state:
    #     state.ysel_i0 = 0
    # if 'sname_i0' not in state:
    #     state.sname_i0 = 0
    # if 'models_0' not in state:
    #     state.models_0 = ['EMEP', 'MATCH']
    # if 'sel_1_i0' not in state:
    #     state.sel_1_i0 = 0
    # if 'sel_2_i' not in state:
    #     state.sel_2_i = 0
    #     state.sel_2 = state.allpars[state.sel_2_i]
    # if 'sel_2_0' not in state:
    #     state.sel_2_0 = state.allpars[0]
    # if 'count' not in state:
    #     state.count = 0

# def update_selections(state):
#     state.ysel_i0 = state.ysel_i
#     state.p3_sname_i0 = state.p3_sname_i
#     state.models_0 = state.models
#     state.sel_1_i0 = state.sel_1_i
#     state.sel_2_0 = state.sel_2
    # state.sel_2_i0 = state.sel_2_i

if __name__ == "__main__":
    run()