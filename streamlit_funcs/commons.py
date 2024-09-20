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
        'NH3_ppb':{'EMEP':'NH3_ppb','MATCH':'NH3_ppb', 'Note':'NH3 concentration (ppbv)'}, # ammonia concentration, MATCH is in ug/m3
        'Vg': {'EMEP':'Vg3NH3','MATCH':'Vg', 'Note':'NH3 deposition velocity, EMEP is velocity at 3m'}, # deposition velocity, EMEP is velocity at 3m
        'Gns':{'EMEP':'Gns','MATCH':'Gns', 'Note':'non-stomatal conductance'}, # non-stomatal conductance
        'Rsur':{'EMEP':'Rsur','MATCH':'Rsur', 'Note':'surface resistance'}, # surface resistance
        'PS':{'EMEP':'PS', 'MATCH':'PS','Note':'surface pressure (hPa)'}, # surface pressure
        'T2':{'EMEP':'T2','MATCH':'T2','Note':'2m air temperature (degC)' }, # 2m temperature in degree C
        'RH':{'EMEP':'RH','MATCH':'RH','Note':'relative humidity'}, #relative humidity'
        'UREF':{'EMEP':'UREF', 'MATCH':'UV10','Note':'reference wind speed'}, # reference wind speed
        'LE':{'EMEP':'LE','MATCH':'LE','Note':'latent heat flux'},   # latent heat flux
        'RAIN':{'EMEP':'RAIN','MATCH':'RAIN'}, # raining rate
        'SNOW':{'EMEP':'dsnow','MATCH':'dsnow'}, # snowing rate
        'USTmod':{'EMEP':'USTmod','MATCH':'USTmod', 'Note':'ustar modified by landuse'}, # ustar friction velocity modified by landuse etc.
        'PARsun':{'EMEP':'PARsun','MATCH':'PARsun', 'Note': 'photoactive radiation in sunlight areas'}, #photoactive radiation in sunlight areas
        'PARshade':{'EMEP':'PARshade','MATCH':'PARshade', 'Note':'photoactive radiation in shaded areas'}, #photosynthetically active radiation in shaded canopy
        'Ra':{'EMEP':'Ra','MATCH':'Ra', 'Note':'aerodynamic resistance'}, # aerodynamic resistance
        'Rb':{'EMEP':'Rb','MATCH':'Rb', 'Note':'quasi-laminar boundary layer resistance'}, # quasi-laminar boundary layer resistance
        'LAI':{'EMEP':'LAI','MATCH':'LAI', 'Note':'leaf area index'}, # leaf area index
        'LAIsun':{'EMEP':'LAIsunfrac','MATCH':'LAIsun', 'Note':'sunlit leaf area index'}, # sunlit leaf area index
        'SAI':{'EMEP':'SAI','MATCH':'SAI', 'Note':'surface area index'}, # surface area index
        'Fphen':{'EMEP':'Fphen','MATCH':'Fphen', 'Note':'phenology parameter for gsto calculation'}, # phenology parameter for gsto calculation
        'Flight':{'EMEP':'Flight','MATCH':'Flight', 'Note':'light parameter'}, # light response parameter
        'Ft':{'EMEP':'Ftemp','MATCH':'Ft', 'Note':'temperature parameter'}, # temperature parameter do
        'Fvpd':{'EMEP':'Fvpd','MATCH':'Fvpd', 'Note':'vapor pressure parameter'}, # vapor pressure deficit parameter
        'HF':{'MATCH':'H0', 'EMEP':'Hd', 'Note':'sensible heat flux'}, # sensible heat flux

        'SO2_ppb':{'MATCH':'SO2_ppb', 'Note':'SO2 concentration (ppbv)'}, # sulphur dioxide concentration, MATCH is in ug/m3
        'HMIX':{'MATCH':'HMIX','Note':'mixing height'}, # mixing height
        'MOL':{'MATCH':'MOL','Note':'Monin-Obukhov length'}, # Monin-Obukhov length
        'Rinc':{'MATCH':'Rinc', 'Note':'in-canopy resistance'}, # in-canopy resistance
        'Rext':{'MATCH':'Rext','Note':'external leaf area resistance'}, # external leaf area resistance
        'X_tot':{'MATCH':'X_tot','Note':'total compensation point'}, # total compensation point

        # 'LC':{'EMEP':'LC','Note':'land use type'}, # land cover
        'VPD':{'EMEP':'VPD'}, # vapor pressure deficit
        'ZEN':{'EMEP':'ZEN'}, # solar zenith angle
        'PARdbh':{'EMEP':'PARdbh'}, # PAR above canopy
        'PARdif':{'EMEP':'PARdif'}, # PAR diffuse
        # 'Hd':{'EMEP':'Hd'}, # sensible heat flux
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