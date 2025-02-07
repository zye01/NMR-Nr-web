import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_funcs import common_params, common_funcs


st.set_page_config(layout="wide")
plotpath = './post_process/plots'
models = ['EMEP', 'MATCH', 'DEHM']
years = [2018, 2019]
cases = ['BD', 'noBD']
domains = ['Denmark', 'Europe']


def run():
    st.title('Annual mean spatial distributions')

    state = st.session_state()
    initiate_state(state)

    tab1, tab2, tab3 = st.tabs(['Concentrations','Deposition', 'Emissions'])
    with tab1:
        show_concentrations(state)

    with tab2:
        show_deposition(state)

    with tab3:
        show_emissions(state)

def show_concentrations(state):
    settings_conc(state)


def settings_conc(state):
    # Select for domain, models, and years
    c1, c2, c3 = st.columns(3)
    # select the domain
    c1.selectbox(domains, 


def initiate_state(state):
    pass


if __name__ == "__main__":
    run()