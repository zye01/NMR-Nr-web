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
    st.title('Non-stomatal Resistance Parameterization')
    tab1, tab2 = st.tabs(['Parameterization','Dependency'])

    with tab1:
        c1, c2, c3 = st.columns([1,2,1])
        display_variables(state, c1)
        display_no_BD(state, c2)
        display_BD(state, c2)
        display_results(state, c3)
        display_notes(state,st)

    with tab2:
        plot_dependence(state)

def calculate_old_rnc(ts, rh,asn, constant=0.0455):
    f1 = calc_f1(ts, rh)
    f2 = calc_f2(asn)
    rns_old = min(max(10, constant*f1*f2),100)
    # rns_old = min(max(10, constant*f1*f2),200)
    return rns_old

def calc_f1(ts, rh):
    f1 = 10*np.log(2+ts-273.15)*np.exp((100-rh)/7)/np.log(10)
    return f1

def calc_f2(asn):
    f2 = 10**(-1.1099*asn+1.6769)
    return f2

def calculate_new_rnc(sai, rh, ts, asn, X_a):
    Ra, Rb = 50, 30
    X_w = calc_Xw(ts,asn, X_a)
    rw_min = calc_rw_min(sai, rh)
    # rns_new = 1/(1/rext+1/(rs+rinc))
    rns_new = (X_a/(X_a-X_w))*(rw_min+Ra+Rb) - Ra - Rb
    return rns_new

def calc_rw_min(sai,rh):
    sai_haarweg = 3.5
    alpha_nh3 = 2
    rw_min = sai_haarweg/sai*alpha_nh3*np.exp((100-rh)/12)
    return rw_min

def calc_Xw(ts,asn, X_a):
    gamma_w = calc_gamma_w(ts, asn, X_a)
    X_w = (2.75*10**15/ts)*np.exp(-1.04*10**4/ts)*gamma_w
    return X_w


def calc_gamma_w(ts, asn, X_a):
    gamma_w = 1.84*10**3*np.exp(-0.11*(ts-273.15))*X_a-850
    fasn = calc_fasn(asn)
    gamma_w = gamma_w*max(0, fasn)
    return gamma_w


def calc_fasn(asn):
    fasn = 1.12-1.32*asn
    return fasn



def calc_rext(sai, rh):
    sai_haarweg = 3.5
    alpha_nh3 = 2
    rext = sai_haarweg/sai*alpha_nh3*np.exp((100-rh)/12)
    return rext

def calc_rinc(sai, hveg, ustar):
    if ustar > 0:
        rinc = min(14*sai*hveg/ustar, 1000)
    else:
        rinc = 1000
    return rinc

def display_no_BD(state, cm):
    cm.markdown('### DEHM no-BD (Old)')
    # state.f1 = 10*np.log(2+state.ts-273.15)*np.exp((100-state.rh)/7)/np.log(10)
    state.f1 = calc_f1(state.ts, state.rh)
    cm.latex(r'f1=10\times\frac{\log(2+Ts-273.15)\times\exp(\frac{100-RH}{7})}{\log(10)} \\ ~~~~~~ = \underline{%.2f}' % state.f1)
    # state.f2 = 10**(-1.1099*state.asn+1.6769)
    state.f2 = calc_f2(state.asn)
    cm.latex(r'f2=10^{-1.1099\times asn+1.6769} = \underline{%.2f}' % state.f2)
    state.rns_old = calculate_old_rnc(state.ts, state.rh, state.asn)
    cm.latex(r'\large{r_{ns}} = \min(\max(10,0.0455\times f1\times f2),100) \\ ~~~~~ = \underline{%.1f} ' % state.rns_old)
    # cm.latex(r'\large{r_{ns}} = \max(10, 0.0455\times f1\times f2) \\ ~~~~~ = \underline{%.1f} ' % state.rns_old)
    
def display_BD(state, cm):
    cm.markdown('### DEHM BD (New)')
    

    state.fasn = calc_fasn(state.asn)
    cm.latex(r'f_{asn} = 1.12-1.32\times asn = \underline{%.2f}' % state.fasn)

    state.gamma_w = calc_gamma_w(state.ts, state.asn, state.X_a)
    cm.latex(r'\Gamma_w = (1.84\times 10^3\times\exp(-0.11\times (Ts-273.15))\times X_a-850)\times f_{asn} = \underline{%.2f}' % state.gamma_w)

    state.X_w = calc_Xw(state.ts, state.asn, state.X_a)
    cm.latex(r'X_w = \frac{2.75\times 10^{15}}{Ts}\times\exp(-1.04\times 10^4/Ts)\times\Gamma_w = \underline{%.2f}' % state.X_w)

    cm.latex(r'SAI_{Haarweg} = 3.5')
    cm.latex(r'\alpha = 2')
    state.rw_min = calc_rw_min(state.sai, state.rh)
    cm.latex(r'r_{w,min} = \frac{SAI}{SAI_{Haarweg}}\times\alpha\times\exp(\frac{100-RH}{12}) = \underline{%.2f}' % state.rw_min)

    cm.latex(r'Ra = 50')
    cm.latex(r'Rb = 30')

    state.rns_new = calculate_new_rnc(state.sai, state.rh, state.ts, state.asn, state.X_a)
    # cm.latex(r'\large{r_{ns}} = \frac{X_w}{X_a-X_w}\times(Ra+Rb)+\frac{X_a}{X_a-X_w}\times r_{w,min} \\ ~~~~~ = \underline{%.2f}' % state.rns_new)
    cm.latex(r'\large{r_{ns}} = \frac{X_a}{X_a-X_w}\times(r_{w,min}+Ra+Rb)-Ra-Rb \\ ~~~~~ = \underline{%.2f}' % state.rns_new)

    # state.rext = calc_rext(state.sai, state.rh)
    # cm.latex(r'{r_{ext}} = \frac{SAI_{Haarweg}}{SAI}\times \alpha \times\exp(\frac{100-RH}{12}) \\ ~~~~~ = \underline{%.2f}' % state.rext)

    # state.rinc = calc_rinc(state.sai, state.hveg, state.ustar)
    # if state.ustar > 0:
    #     cm.latex(r'{r_{inc}} = \min(14\times SAI\times hveg/ustar, 1000) \\ ~~~~~ = \underline{%.2f}' % state.rinc)
    # else:
    #     cm.latex(r'{r_{inc}} = 1000')

    # state.rns_new = calculate_new_rnc(state.sai, state.rh, state.ts, state.asn, state.rext)
    # cm.latex(r'\large{r_{ns}} = \frac{1}{\frac{1}{r_{ext}}+\frac{1}{r_{soil}+r_{inc}}} \\ ~~~~~ = \underline{%.2f}' % state.rns_new)



def plot_dependence(state):
    plot_for_RH(state)
    plot_for_SAI(state)

def plot_for_RH(state):
    st.markdown('### Relationship with RH')
    c1, c2 = st.columns([1,3])
    ts, asn, sai, X_a = select_variables_for_RH(state,c1)

    # generate rh data from 50 to 100, step=0.2
    rhs = np.arange(50, 100, step=0.5)
    rns_olds = [calculate_old_rnc(ts,rh,asn) for rh in rhs]
    rns_news = [calculate_new_rnc(sai, rh, ts, asn, X_a) for rh in rhs]
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
    X_a = cm.number_input('X_a',key='X_a1', value=0.6, format='%0.1f', step=0.1)
    # hveg = cm.number_input('hveg (m)',key='hveg1', value=20.0, format='%0.1f', step=1.0)
    # ustar = cm.number_input('ustar (m/s)',key='ustar1', value=1.0, format='%0.1f', step=0.1)
    # rs = cm.number_input('r_soil',key='rs1', value=100, format='%d', step=30)
    return ts, asn, sai, X_a



def select_variables_for_SAI(state,cm):
    ts = cm.number_input('Ts (Kelvin)', value=280.0,key='ts2', format='%0.1f', step=1.0)
    rh = cm.number_input('RH (%)', value=80.0,key='rh2', format='%0.1f', step=2.0)
    asn = cm.number_input('asn',key='asn2', value=0.2, format='%0.1f', step=0.1)
    X_a = cm.number_input('X_a',key='X_a2', value=0.6, format='%0.1f', step=0.1)
    # sai = cm.number_input('SAI (surface area index, dimensionless)', value=2.0, format='%0.1f', step=0.5)
    # hveg = cm.number_input('hveg (m)',key='hveg2', value=20.0, format='%0.1f', step=1.0)
    # ustar = cm.number_input('ustar (m/s)',key='ustar2', value=1.0, format='%0.1f', step=0.1)
    # rs = cm.number_input('r_soil',key='rs2', value=100, format='%d', step=30)
    return ts, asn, rh, X_a


def plot_for_SAI(state):
    st.markdown('### Relationship with SAI')

    c1, c2 = st.columns([1,3])
    ts, asn, rh, X_a = select_variables_for_SAI(state,c1)

    sais = np.arange(2, 4, step=0.05)
    rns_olds = [calculate_old_rnc(ts,rh,asn) for sai in sais]
    rns_news = [calculate_new_rnc(sai,rh, ts, asn, X_a) for sai in sais]
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
                line=dict(width=3),
                )
            )

    fig.update_layout(
        xaxis_title=data['x']['label'],
        margin=dict(l=2, r=2, t=30, b=2),
        # width=1200,
        height=500,
        legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="left",
        x=0.01,
        font=dict(size=15)
        ),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=20),tickfont=dict(size=15), showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey',\
            ticks='inside',title_font=dict(size=26),tickfont=dict(size=15), showgrid=False),
        )

    # cunit = par_dict[state.sel_1]['units']
    fig.update_yaxes(title_text='r_ns',\
                     showline=True, linewidth=1, linecolor='lightgrey',\
                     ticks='inside',title_font=dict(size=18),tickfont=dict(size=15), showgrid=False)
    
    cm.plotly_chart(fig, use_container_width=True)






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
    state.ts = cm.number_input('Ts (Kelvin)',key='ts0', value=280.0, format='%0.1f', step=0.5)
    state.rh = cm.number_input('RH (%)',key='rh0', value=80.0, format='%0.1f', step=2.0)
    state.asn = cm.number_input('asn (acidity ratio)',key='asn0', value=0.2, format='%0.1f', step=0.1)
    state.sai = cm.number_input('SAI (surface area index, dimensionless)',key='sai0', value=2.0, format='%0.1f', step=0.5)
    # state.hveg = cm.number_input('hveg (height of vegetation in m)',key='hveg0', value=20.0, format='%0.1f', step=1.0)
    # state.ustar = cm.number_input('ustar (friction velocity, m/s)',key='ustar0', value=1.0, format='%0.1f', step=0.1)
    # state.rs = cm.number_input('r_soil (soil resistance)',key='rs0', value=100, format='%d', step=30)
    # state.ra = cm.number_input('r_a',key='ra0', value=50.0, format='%0.1f', step=5.0)
    # state.rb = cm.number_input('r_b',key='rb0', value=30.0, format='%0.1f', step=2.0)
    state.X_a = cm.number_input('X_a',key='X_a0', value=0.6, format='%0.1f', step=0.1)
    # state.const = cm.number_input('constant', value=0.0455, format='%0.4f', step=0.01)



def display_notes(state,cm):
    cm.markdown('#### Notes:')
    cm.latex(r'asn = \min(30,\frac{0.6\times so2\_con}{\max(nh3\_con,1e-10)})')
    cm.markdown('hveg = 1m for arable land and 20m for forests')
    cm.markdown('r_soil = 1000 for frozen soil, 10 for water and non-frozen wet soil, 100 for dry soil')
    cm.markdown('r_inc in bi-dir is 1E10 for grassland.')
    cm.markdown('SAI=LAI+1 for forests, SAI = LAI for the grassland, SAI is parameterized with LAI for crops in the growing season.')
    # cm.markdown('Applies when ustar > 0 ')


if __name__ == '__main__':
    run()