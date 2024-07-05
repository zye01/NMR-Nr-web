import pandas as pd
import numpy as np
import streamlit as st
from nh3_db import nh3_database


class st_nh3:

    def __init__(self):
        self.db = nh3_database()

        pass

    def read_nh3_data(self,state):
        state.sel_df = self.db.read_data(start_date=state.start_date,\
            end_date=state.end_date,sources=state.source)
        state.sel_df = state.sel_df.sort_values(by=['site', 'ed'])
        
    
    def get_nh3_stations(self,state):
        state.sel_sdf = self.db.get_stations(df=state.sel_df)
        state.sel_stlist = self.get_sitename_combine(state.sel_sdf)

    def get_sitename_combine(self,sdf):
        stlist = []
        for i in range(len(sdf.index)):
            try:
                sdf['sname'][i]
                stlist.append(sdf['sid'][i]+': '+sdf['sname'][i])
            except:
                stlist.append(sdf['sid'][i]+': '+sdf['sid'][i][3:])
        return stlist

    def stations_from_source(self,state):
        fil_df = state.sel_sdf[state.sel_sdf['source']==state.st_source].reset_index(drop=True)
        state.sel_source_stlist = self.get_sitename_combine(fil_df)

    def calculate_statistics(self,state):
        totdays = (state.end_date-state.start_date).days
        state.tot_sts,state.tot_perc = [], []
        for i in range(4):
            temp_sdf = state.sel_sdf[state.sel_sdf['source']==i+1].reset_index(drop=True)
            state.tot_sts.append(len(temp_sdf.index))
            percs=[]
            for ist in temp_sdf['sid'].values:
                temp_df = state.sel_df[state.sel_df['site']==ist]
                ndays = np.sum((temp_df['ed']-temp_df['st'])/ np.timedelta64(1, 'D'))
                # ndays = (temp_df['ed'].values.max()-temp_df['st'].values.min()).astype('timedelta64[D]').astype(int)
                iperc = ndays*100/totdays
                percs.append(np.int64(iperc))
            perc = (np.int64(np.array(percs).mean()))
            state.tot_perc.append(perc)

    
    def get_1site_data(self,state):
        state.s1_df = state.sel_df[state.sel_df['site']==state.selected_site].reset_index(drop=True)
        # state.s1_df = state.s1_df.rename(columns={'ed':'Time','val':'NH3 concentrations'})
        # print(state.s1_df)
        # state.s1_df['time'] = [(state.s1_df['st'][i]+state.s1_df['ed'][i])/2 for i in range(len(state.s1_df.index))]