import numpy as np
# from scipy import stats

class common_params:

    source_id = {'EBAS':1,'Denmark':2,'France':3,'Holland':4}
    par_dict = {
        'NH3': {'units': 'ug N/m3', 'hol_convf': 14 / 17, 'sunit':'ugNm-3'},
        'O3': {'units': 'ug/m3', 'hol_convf': 1, 'sunit':'ugm-3'},
        'SO2': {'units': 'ug S/m3', 'hol_convf': 0.5, 'sunit':'ugSm-3'},
        'NO2': {'units': 'ug N/m3', 'hol_convf': 14 / 48, 'sunit':'ugNm-3'},
        'NH4': {'units': 'ug N/m3', 'hol_convf': 14 / 18, 'sunit':'ugNm-3'},
        'HNO3': {'units': 'ug N/m3', 'hol_convf': 14 / 63, 'sunit':'ugNm-3'},
        'SO4': {'units': 'ug S/m3', 'hol_convf': 32 / 96, 'sunit':'ugSm-3'},
    }
    seasons = {
        'MAM': 'Spring (MAM)',
        'JJA': 'Summer (JJA)',
        'SON': 'Fall (SON)',
        'DJF': 'Winter (DJF)',
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