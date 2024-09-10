import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

def run():
    state = st.session_state
    st.markdown('''
    <style>
    .katex-html {
        text-align: left;
    }
    </style>''',
    unsafe_allow_html=True
    )
    st.title('Non-stomatal Resistance Calculations')
    tab1, tab2 = st.tabs(['Calculations','Plots'])

    with tab1:
        c1, c2, c3 = st.columns([1,2,1])
        display_variables(state, c1)
        display_no_BD(state, c2)
        display_BD(state, c2)
        display_results(state, c3)
        display_notes(state,st)

    with tab2:
        plot_dependence(state)

def plot_dependence(state):
    plot_for_RH(state)
    plot_for_SAI(state)

def plot_for_RH(state):
    c1, c2 = st.columns([1,4])
    ts, asn, sai, ustar, hveg, rs = select_variables_for_RH(state,c1)

    # generate rh data from 50 to 100, step=0.2
    rhs = np.arange(50, 100, step=0.5)
    rns_olds = [calculate_old_rnc(ts,rh,asn) for rh in rhs]
    rns_news = [calculate_new_rnc(sai, rh, ustar, hveg, rs) for rh in rhs]
    pdata = {
        'x':{'label':'RH', 'data':rhs},
        'y2':{'label':'rns_old', 'data':rns_olds},
        'y1':{'label':'rns_new','data':rns_news},
    }
    lineplot(state,c2,pdata)
    

def select_variables_for_RH(state,cm):
    ts = cm.number_input('Ts (Kelvin)',key='ts1', value=280.0, format='%0.1f', step=1.0)
    # rh = cm.number_input('RH (%)', value=85.0, format='%0.1f', step=2.0)
    asn = cm.number_input('asn',key='asn1', value=0.2, format='%0.1f', step=0.1)
    sai = cm.number_input('SAI',key='sai1', value=2.0, format='%0.1f', step=0.5)
    hveg = cm.number_input('hveg (m)',key='hveg1', value=20.0, format='%0.1f', step=1.0)
    ustar = cm.number_input('ustar (m/s)',key='ustar1', value=1.0, format='%0.1f', step=0.1)
    rs = cm.number_input('r_soil',key='rs1', value=100, format='%d', step=30)
    return ts, asn, sai, ustar, hveg, rs

def select_variables_for_SAI(state,cm):
    ts = cm.number_input('Ts (Kelvin)', value=280.0,key='ts2', format='%0.1f', step=1.0)
    rh = cm.number_input('RH (%)', value=80.0,key='rh2', format='%0.1f', step=2.0)
    asn = cm.number_input('asn',key='asn2', value=0.2, format='%0.1f', step=0.1)
    # sai = cm.number_input('SAI (surface area index, dimensionless)', value=2.0, format='%0.1f', step=0.5)
    hveg = cm.number_input('hveg (m)',key='hveg2', value=20.0, format='%0.1f', step=1.0)
    ustar = cm.number_input('ustar (m/s)',key='ustar2', value=1.0, format='%0.1f', step=0.1)
    rs = cm.number_input('r_soil',key='rs2', value=100, format='%d', step=30)
    return ts, asn, rh, ustar, hveg, rs


def plot_for_SAI(state):
    c1, c2 = st.columns([1,4])
    ts, asn, rh, ustar, hveg, rs = select_variables_for_SAI(state,c1)

    sais = np.arange(2, 4, step=0.05)
    rns_olds = [calculate_old_rnc(ts,rh,asn) for sai in sais]
    rns_news = [calculate_new_rnc(sai,rh,ustar,hveg,rs) for sai in sais]
    pdata = {
        'x':{'label':'SAI', 'data':sais},
        'y1':{'label':'rns_new', 'data':rns_news},
        'y2':{'label':'rns_old','data':rns_olds},
    }

    lineplot(state,c2,pdata)

def lineplot(state, cm, data, diff=False):
    if diff:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    else:
        fig = make_subplots(specs=[[{'secondary_y': False}]])

    fig.update_layout(
        template = 'plotly_white'
    )

    for ys in ['y1', 'y2']:
        if ys in data.keys():
            fig.add_trace(
                go.Scatter(
                x=data['x']['data'],
                y=data[ys]['data'],
                name=data[ys]['label'],
                )
            )

    fig.update_layout(
        xaxis_title=data['x']['label'],
        margin=dict(l=2, r=2, t=30, b=2),
        # width=1200,
        height=600,
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

    # cunit = par_dict[state.sel_1]['units']
    fig.update_yaxes(title_text='r_ns',\
                     showline=True, linewidth=1, linecolor='lightgrey',\
                     ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False)
    
    cm.plotly_chart(fig, use_container_width=True)



def calculate_old_rnc(ts, rh,asn):
    f1 = 10*np.log(2+ts-273.15)*np.exp((100-rh)/7)/np.log(10)
    f2 = 10**(-1.1099*asn+1.6769)
    rns_old = min(max(10,0.0455*f1*f2),100)
    return rns_old

def calculate_new_rnc(sai, rh, ustar, hveg, rs):
    sai_haarweg = 3.5
    rext = sai_haarweg/sai*2*np.exp((100-rh)/12)
    if ustar > 0:
        rinc = min(14*sai*hveg/ustar, 1000)
    else:
        rinc = 1000
    rns_new = 1/(1/rext+1/(rs+rinc))

    return rns_new



def display_results(state, cm):
    cm.markdown('### Comparisons')
    # compare the state.rns_old and state.rns_new
    cm.latex(r'\text{Old } \large{r_{ns}} = %.1f' % state.rns_old)
    cm.latex(r'\text{New } \large{r_{ns}} = %.1f' % state.rns_new)
    # ratio of the two
    cm.latex(r'\color{red}\text{Old/New } \large= %.2f' % (state.rns_old/state.rns_new))

    cm.markdown('**In bi-dir:**')
    # ratio of state.rext and state.rns_new
    cm.latex(r'\frac{r_{ext}}{r_{ns}} \large= %.2f' % (state.rext/state.rns_new))


def display_variables(state, cm):
    cm.markdown('### Variables')
    state.ts = cm.number_input('Ts (Kelvin)', value=280.0, format='%0.1f', step=0.5)
    state.rh = cm.number_input('RH (%)', value=80.0, format='%0.1f', step=2.0)
    state.asn = cm.number_input('asn (acidity ratio)', value=0.2, format='%0.1f', step=0.1)
    state.sai = cm.number_input('SAI (surface area index, dimensionless)', value=2.0, format='%0.1f', step=0.5)
    state.hveg = cm.number_input('hveg (height of vegetation in m)', value=20.0, format='%0.1f', step=1.0)
    state.ustar = cm.number_input('ustar (friction velocity, m/s)', value=1.0, format='%0.1f', step=0.1)
    state.rs = cm.number_input('r_soil (soil resistance)', value=100, format='%d', step=30)

def display_no_BD(state, cm):
    cm.markdown('### no-BD (Old)')
    state.f1 = 10*np.log(2+state.ts-273.15)*np.exp((100-state.rh)/7)/np.log(10)
    cm.latex(r'f1=10\times\frac{\log(2+ts-273.15)\times\exp(\frac{100-rh}{7})}{\log(10)} \\ ~~~~~~ = \underline{%.2f}' % state.f1)
    state.f2 = 10**(-1.1099*state.asn+1.6769)
    cm.latex(r'f2=10^{-1.1099\times asn+1.6769} = \underline{%.2f}' % state.f2)
    state.rns_old = min(max(10,0.0455*state.f1*state.f2),100)
    cm.latex(r'\large{r_{ns}} = \min(\max(10,0.0455\times f1\times f2),100) \\ ~~~~~ = \underline{%.1f} ' % state.rns_old)

    
def display_BD(state, cm):
    cm.markdown('### BD (New)')
    state.sai_haarweg = 3.5
    cm.latex(r'SAI_{Haarweg} = 3.5')
    state.alpha_nh3 = 2
    cm.latex(r'\alpha = 2')

    state.rext = state.sai_haarweg/state.sai*state.alpha_nh3*np.exp((100-state.rh)/12)
    cm.latex(r'{r_{ext}} = \frac{SAI_{Haarweg}}{SAI}\times \alpha \times\exp(\frac{100-RH}{12}) \\ ~~~~~ = \underline{%.2f}' % state.rext)

    if state.ustar > 0:
        state.rinc = min(14*state.sai*state.hveg/state.ustar, 1000)
        cm.latex(r'{r_{inc}} = \min(14\times SAI\times hveg/ustar, 1000) \\ ~~~~~ = \underline{%.2f}' % state.rinc)
    else:
        state.rinc = 1000
        cm.latex(r'{r_{inc}} = 1000')

    state.rns_new = 1/(1/state.rext+1/(state.rs+state.rinc))
    cm.latex(r'\large{r_{ns}} = \frac{1}{\frac{1}{r_{ext}}+\frac{1}{r_{soil}+r_{inc}}} \\ ~~~~~ = \underline{%.2f}' % state.rns_new)


def display_notes(state,cm):
    cm.markdown('#### Notes:')
    cm.latex(r'asn = \min(30,\frac{0.6\times so2\_con}{\max(nh3\_con,1e-10)})')
    cm.markdown('hveg = 1m for arable land and 20m for forests')
    cm.markdown('r_soil = 1000 for frozen soil, 10 for water and non-frozen wet soil, 100 for dry soil')
    cm.markdown('r_inc in bi-dir is 1E10 for grassland.')
    cm.markdown('SAI=LAI+1 for forests, SAI = LAI for the grassland, sai is parameterized with LAI for crops in the growing season.')
    # cm.markdown('Applies when ustar > 0 ')


if __name__ == '__main__':
    run()