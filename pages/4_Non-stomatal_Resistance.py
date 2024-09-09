import os
import pandas as pd
import numpy as np
import streamlit as st

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
    c1, c2, c3 = st.columns([1,2,1])

    
    display_variables(state, c1)

    display_no_BD(state, c2)

    display_BD(state, c2)

    display_results(state, c3)

    display_notes(state,st)

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
    state.rh = cm.number_input('RH (%)', value=85.0, format='%0.1f', step=2.0)
    state.asn = cm.number_input('asn (acidity ratio)', value=0.2, format='%0.1f', step=0.1)
    state.lai = cm.number_input('LAI (leaf area index, dimensionless)', value=2.0, format='%0.1f', step=0.5)
    state.hveg = cm.number_input('hveg (height of vegetation in m)', value=20.0, format='%0.1f', step=1.0)
    state.ustar = cm.number_input('ustar (friction velocity, m/s)', value=1.0, format='%0.1f', step=0.1)
    state.rs = cm.number_input('r_soil (soil resistance)', value=100, format='%d', step=30)

def display_no_BD(state, cm):
    cm.markdown('### no-BD (Old)')
    state.f1 = 10*np.log(2+state.ts-273.15)*np.exp((100-state.rh)/7)/np.log(10)
    cm.latex(r'f1=10\times\frac{\log(2+ts-273.15)\times\exp(\frac{100-rh}{7})}{\log(10)} \\ ~~~~~~ = \underline{%.2f}' % state.f1)
    state.f2 = 10**(-1.1099*state.asn+1.6769)
    cm.latex(r'f2=10^{-1.1099\times asn+1.6769} = \underline{%.2f}' % state.f2)
    state.rns_old = min(max(10,0.0455*state.f1*state.f2),200)
    cm.latex(r'\large{r_{ns}} = \min(\max(10,0.0455\times f1\times f2),200) \\ ~~~~~ = \underline{%.1f} ' % state.rns_old)

    
def display_BD(state, cm):
    cm.markdown('### BD (New)')
    state.sai_haarweg = 3.5*state.lai/2
    cm.latex(r'SAI_{Haarweg} = 3.5\times LAI/2 = \underline{%.1f}' % state.sai_haarweg)

    state.rext = state.sai_haarweg*np.exp((100-state.rh)/12)
    cm.latex(r'{r_{ext}} = SAI_{Haarweg}\times\exp(\frac{100-RH}{12}) \\ ~~~~~ = \underline{%.2f}' % state.rext)

    if state.ustar > 0:
        state.rinc = 14*state.lai*state.hveg/state.ustar
        cm.latex(r'{r_{inc}} = 14\times lai\times hveg/ustar \\ ~~~~~ = \underline{%.2f}' % state.rinc)
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
    # cm.markdown('Applies when ustar > 0 ')


if __name__ == '__main__':
    run()