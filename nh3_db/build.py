import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
# from mpl_toolkits.basemap import Basemap
import os

class nh3_database:
    tot_dks = 11
    units = 'ugN m-3'

    def __init__(self):
        self.basepath = '.'
        self.rawpath = os.path.join(self.basepath,'raw')
        self.dbpath = os.path.join(self.basepath,'nh3_db','data')
        self.db_sfile = os.path.join(self.dbpath,'sites_meta.csv')
        self.db_dfile = os.path.join(self.dbpath,'All_NH3.csv')
        

    def read_data(self,start_date=None,end_date=None,sources=None,methods=None):
        if not os.path.isfile(self.db_dfile):
            print('Database file not found, read from web')
            url = 'https://drive.google.com/file/d/18W5VeYeads31dL26KggTKjiZcG_HvcBE/view?usp=sharing'
            url='https://drive.google.com/uc?id=' + url.split('/')[-2]
            df = pd.read_csv(url)
        else:
            df = pd.read_csv(self.db_dfile,header=0)
        df = self.process_datetime(df)

        # Select from start and end date
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df = df[df.st>=start_date].reset_index(drop=True)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            df = df[df.ed<=end_date].reset_index(drop=True)
        
        if sources is not None:
            df = df[df.source.isin(sources)].reset_index(drop=True)

        if methods is not None:
            df = df[df.method.isin(methods)].reset_index(drop=True)
        return df

    def get_stations(self,df=None,start_date=None,end_date=None,sources=None,methods=None):
        if df is None:
            df = self.read_data(start_date=start_date,end_date=end_date,sources=sources,methods=methods)
        stlist = list(df['site'].unique())
        sdf = self.get_station_meta()
        fdf = sdf[sdf.sid.isin(stlist)].reset_index(drop=True)
        return fdf

    def get_station_meta(self):
        df = pd.read_csv(self.db_sfile,header=0)
        return df
    
    @staticmethod
    def process_datetime(df):
        df['st'] = pd.to_datetime(df['st'],format='mixed')
        df['ed'] = pd.to_datetime(df['ed'],format='mixed')
        return df


    # def plot_site_map(self):
    #     df = pd.read_csv(self.db_sfile,encoding = "ISO-8859-1", engine='python')
    #     # print(df)
    #     lats,lons = df['lat'].to_list(),df['lon'].to_list()
    #     z = df['source'].to_list()
    #     figname = self.basepath+'site_map.png'
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     map = self.get_basemap()
    #     x,y = map(lons,lats)
    #     map.scatter(x,y,c=z,marker='o',edgecolors='black',linewidth=0.1,s=10)
    #     fig.savefig(figname,dpi=300,bbox_inches='tight')
    #     plt.close(fig)
        

    # @staticmethod
    # def get_basemap():
    #     map = Basemap(projection='merc',llcrnrlon=-15.0,llcrnrlat=35.0,urcrnrlon=42.0,urcrnrlat=72,resolution = 'i')
    #     map.drawparallels(np.arange(-90.,90.,5.),labels=[True,False,False,False],color="grey",linewidth=0.3,fontsize=8)
    #     map.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],color='grey',linewidth=0.3,fontsize=8)
    #     map.drawmapboundary(fill_color='white')
    #     map.drawcoastlines(linewidth=0.12)
    #     map.drawcountries(linewidth=0.1)
    #     return map




