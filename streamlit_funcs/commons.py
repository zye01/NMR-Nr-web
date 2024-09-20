import numpy as np
# from scipy import stats

class common_params:

    source_id = {'EBAS':1,'Denmark':2,'France':3,'Holland':4}
    par_dict = {
        'NH3': {'units': 'μg N/m3', 'hol_convf': 14 / 17, 'sunit':'ugNm-3'},
        'O3': {'units': 'μg/m3', 'hol_convf': 1, 'sunit':'ugm-3'},
        'SO2': {'units': 'μg S/m3', 'hol_convf': 0.5, 'sunit':'ugSm-3'},
        'NO2': {'units': 'μg N/m3', 'hol_convf': 14 / 48, 'sunit':'ugNm-3'},
        'NH4': {'units': 'μg N/m3', 'hol_convf': 14 / 18, 'sunit':'ugNm-3'},
        'HNO3': {'units': 'μg N/m3', 'hol_convf': 14 / 63, 'sunit':'ugNm-3'},
        'SO4': {'units': 'μg S/m3', 'hol_convf': 32 / 96, 'sunit':'ugSm-3'},
    }
    seasons = {
        'MAM': 'Spring (MAM)',
        'JJA': 'Summer (JJA)',
        'SON': 'Fall (SON)',
        'DJF': 'Winter (DJF)',
    }
    parameters = {
        'PS':{'EMEP':'PS', 'MATCH':'PS'}, # surface pressure
        'T2':{'EMEP':'T2','MATCH':'T2',},
        'RH':{'EMEP':'RH','MATCH':'RH',}, #relative humidity'
        'UREF':{'EMEP':'UREF', 'MATCH':'UV10'}, # reference wind speed
        'LE':{'EMEP':'LE','MATCH':'LE',},   # latent heat flux
        'RAIN':{'EMEP':'RAIN','MATCH':'RAIN'}, # raining rate
        'SNOW':{'EMEP':'dsnow','MATCH':'dsnow'}, # snowing rate
        'USTmod':{'EMEP':'USTmod','MATCH':'USTmod'}, # ustar friction velocity modified by landuse etc.
        'PARsun':{'EMEP':'PARsun','MATCH':'PARsun'}, #photosynthetically active radiation in sunlit canopy
        'PARshade':{'EMEP':'PARshade','MATCH':'PARshade'}, #photosynthetically active radiation in shaded canopy
        'Ra':{'EMEP':'Ra','MATCH':'Ra'}, # aerodynamic resistance
        'Rb':{'EMEP':'Rb','MATCH':'Rb'}, # quasi-laminar boundary layer resistance
        'Gns':{'EMEP':'Gns','MATCH':'Gns'}, # non-stomatal conductance
        'LAI':{'EMEP':'LAI','MATCH':'LAI'}, # leaf area index
        'LAIsun':{'EMEP':'LAIsunfrac','MATCH':'LAIsun'}, # sunlit leaf area index
        'SAI':{'EMEP':'SAI','MATCH':'SAI'}, # surface area index
        'Fphen':{'EMEP':'Fphen','MATCH':'Fphen'}, # phenology parameter
        'Flight':{'EMEP':'Flight','MATCH':'Flight'}, # light response parameter
        'Ft':{'EMEP':'Ftemp','MATCH':'Ft'}, # temperature parameter do
        'Fvpd':{'EMEP':'Fvpd','MATCH':'Fvpd'}, # vapor pressure deficit parameter
        'Vg': {'EMEP':'Vg3NH3','MATCH':'Vg'}, # deposition velocity, EMEP is velocity at 3m
        'Rsur':{'EMEP':'Rsur','MATCH':'Rsur'}, # surface resistance
        'NH3_ppb':{'EMEP':'NH3_ppb','MATCH':'NH3_ppb'}, # ammonia concentration, MATCH is in ug/m3

        'SO2_ppb':{'MATCH':'SO2_ppb'}, # sulphur dioxide concentration, MATCH is in ug/m3
        'HF':{'MATCH':'H0',}, # sensible heat flux
        'HMIX':{'MATCH':'HMIX',}, # mixing height
        'MOL':{'MATCH':'MOL'}, # Monin-Obukhov length
        'Rinc':{'MATCH':'Rinc'}, # in-canopy resistance
        'Rext':{'MATCH':'Rext'}, # external leaf area resistance
        'X_tot':{'MATCH':'X_tot'}, # total compensation point

        'LC':{'EMEP':'LC'}, # land cover
        'VPD':{'EMEP':'VPD'}, # vapor pressure deficit
        'ZEN':{'EMEP':'ZEN'}, # solar zenith angle
        'PARdbh':{'EMEP':'PARdbh'}, # PAR above canopy
        'PARdif':{'EMEP':'PARdif'}, # PAR diffuse
        'Hd':{'EMEP':'Hd'}, # sensible heat flux
        'UST':{'EMEP':'UST'}, # ustar from met input
        'Fsun':{'EMEP':'Fsun'}, # sunlit fraction
        'FSW':{'EMEP':'FSW'}, # soil wetness
        'Fenv':{'EMEP':'Fenv'}, # environmental factor
        'gsto':{'EMEP':'gsto'}, # stomatal conductance
        'gsun':{'EMEP':'gsun'}, # sunlit stomatal conductance
        'FstO3':{'EMEP':'FstO3'}, # stomatal O3 flux
        'Gsto':{'EMEP':'Gsto'}, # stomatal conductance

        'None':{},
    }
    
class common_funcs:
    
    def __init__(self):
        pass
    
    @staticmethod
    def calc_metrics(sim,obs):
        outdict = {}
        # Remove nan values, can be removed as it is duplicated

        N = obs.shape[0]
        # outdict['N'] = N

        # meansim = np.nanmean(sim)
        # meanobs = np.nanmean(obs)
        # sigmasim = np.nanstd(sim)
        # sigmaobs = np.nanstd(obs)

        diff = sim - obs
        MB = np.nanmean(diff)
        outdict['MB'] = MB

        square_diff = np.square(diff)
        mean_square_diff = np.nanmean(square_diff)
        RMSE = np.sqrt(mean_square_diff)
        outdict['RMSE'] = RMSE

        addition = np.absolute(sim)+np.absolute(obs)
        division = np.where(addition==0, np.nan, np.true_divide(diff, addition))
        NMB = 2*np.nanmean(division)
        outdict['NMB'] = NMB

        # FGE = 2*np.nanmean(np.absolute(division))
        # outdict['FGE'] = FGE


        corr = np.corrcoef(sim,obs)[0,1]
        outdict['Pearsonr'] = corr
        
        # diffsim = sim - meansim
        # diffobs = obs - meanobs
        # multidiff = np.multiply(diffsim, diffobs)
        # CORR = np.nanmean(multidiff)/(sigmasim*sigmaobs)
        # print('For CORR calculations: ')
        # outdict['COV'] = np.nanmean(multidiff)
        
        # outdict['Corr'] = np.corrcoef(sim,obs)[0,1]


        # sum_square_diff = np.nansum(square_diff)
        # sum_square_obs = np.nansum(np.square(diffobs))
        # # outdict['SSR'] = sum_square_diff
        # # outdict['SST'] = sum_square_obs
        # R2 = 1 - sum_square_diff/sum_square_obs
        # # print('For R2 calculations:')
        # # print(sum_square_diff,sum_square_obs)
        # # outdict['R2_1'] = R2
        # # outdict['R2_2'] = r2_score(sim,obs)
        # slope, intercept, r_value, p_value, std_err = stats.linregress(sim, obs)
        # outdict['R2'] = r_value**2

        # df = pd.DataFrame(outdict)
        return outdict