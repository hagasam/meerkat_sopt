import numpy as np
import pandas as pd
import katpoint
import time
import warnings
warnings.filterwarnings('ignore')
import random
import datetime
from tqdm import tqdm
import sys
import json
import ephem
np.random.seed(40)
import utils


MAX_NUM_IDLES = 900
class Observation:
    def __init__(self, configs, args):
        '''
        Usage:
        
            Important note: before creating the object obs, make sure that the downloaded csv files from OPT are the latest ones
            
            obs = Observation(configs, args) # the arguments that are passed are config.yml file containing all the parameter
            values for the optimization and the file names (csv) for pulsar and imaging SBs from OPT. The csv file from the dashboard in katpaws is required in config.yml too.
            The "obs" is an instance of class Observation, and it will be used by the optimization algorithm(s).  
        '''
        self.configs = configs
        self.mid_time_opt_aproj  = katpoint.Timestamp(args.start).secs + 3600 * 24 * 54 # maximize the number of A-rank in the first ~ two months
        self.mid_time_opt_b1proj = katpoint.Timestamp(args.start).secs + 3600 * 24 * 115 # maximize the number of B1-rank in the first ~ three months
        self.filename = [self.configs['imaging'], self.configs['pulsar']]
        # assert isinstance(self.filename, list) # enforcing the argument to be a list
        if len(self.filename) > 1:
            self.data_ = pd.concat([pd.read_csv(self.filename[i]) for i in range(len(self.filename))], axis = 0)
            self.data_ = self.data_.reset_index(drop=True)
        else:
            self.data_ = pd.read_csv(self.filename[0])
        self.data_.dropna(subset=['lst_start'], inplace=True)
        self.data_.dropna(subset=['lst_start_end'], inplace=True)
        self.data = self.data_[(self.data_['state'] != 'DECLINED') & (self.data_['state'] != 'DRAFT')]
        self.data = self.data.dropna(subset=["simulated_duration"])
        self.data = self.data.reset_index(drop=True) # reset index
        self.data['lst_start_secs'] = self.data['lst_start'].map(lambda x: utils.convert_string_to_secs(x))
        self.data['lst_start_end_secs'] = self.data['lst_start_end'].map(lambda x: utils.convert_string_to_secs(x))
        self.antenna = katpoint.Antenna('Antenna_Position, -30:43:17.3, 21:24:38.5, 1038.0, 12.0')
        self.dashboard = pd.read_csv(self.configs['dashboard']) # this reads the csv file from Manager Katpaws
        self.rating = dict(zip(self.dashboard['Proposal Id'], self.dashboard['Grade'])) 
        self.data['Grade'] = self.data['proposal_id'].apply(lambda x: utils.get_grade(x, self.rating)) 
        self.dummy = json.load(open('/Users/Sam-Macbook/work/notebook/dummy.json'))
        self.data_avsrss = self.data[(self.data['avoid_sunrise_sunset'] == 'Yes') & (self.data['night_obs'] != 'Yes')]
        self.data_night = self.data[(self.data['night_obs'] == 'Yes')]
        self.data_day = self.data[(self.data['avoid_sunrise_sunset'] != 'Yes') & (self.data['night_obs'] != 'Yes')]
        
        
    def check_lst(self, day, lst_start, lst_end):
        convert_to_lst = self.antenna.local_sidereal_time(day)
        lst_now = utils.convert_string_to_secs(str(convert_to_lst))
        if lst_start > lst_end:
            return (lst_start <= lst_now) | (lst_end > lst_now)
        else:
            return (lst_start <= lst_now) & (lst_end >= lst_now)
    
    def get_obs_at_time(self, day, data, day_time, sunrise, sunset):
        '''
        This routine retrieves from a table (panda dataframe) all the observations
        that can be run at a given day time and returns it as a sub-table.
        Please note that the time is assumed to be UTC
            Parameters:
                    day (string): Y/M/D hour:min:sec [e.g. '2024-02-01 15:23:20']
                    data (dataframe): table

            Returns:
                    obs_run (dataframe): table
        '''
        convert_to_lst = self.antenna.local_sidereal_time(katpoint.Timestamp(day))
        lst_obs = utils.convert_string_to_secs(str(convert_to_lst))
        data_mid = data[data['lst_start_secs'] > data['lst_start_end_secs']]
        data_nor = data[data['lst_start_secs'] < data['lst_start_end_secs']]
        obs_run_nor = data_nor[(data_nor['lst_start_secs'] <= lst_obs) & (data_nor['lst_start_end_secs'] >= lst_obs)]
        if len(data_mid) > 0:
            obs_run_mid = data_mid[(data_mid['lst_start_secs'] <= lst_obs) | (data_mid['lst_start_end_secs'] > lst_obs)]
            obs_run = pd.concat([obs_run_mid, obs_run_nor], axis = 0)
            obs_run = obs_run.reset_index(drop=True)
        else: 
            obs_run = obs_run_nor.reset_index(drop=True)
        temp_day = day
        if day_time == 'daytime':
            max_length = katpoint.Timestamp(sunset).secs - temp_day - 1800 # buffer of 30 minutes
            obs_run = obs_run[(obs_run['simulated_duration'] <= max_length)].reset_index(drop=True)
        else:
            max_length = katpoint.Timestamp(sunrise).secs - temp_day - 1800 # buffer of 30 minutes
            obs_run = obs_run[(obs_run['simulated_duration'] <= max_length)].reset_index(drop=True)
        return obs_run
            
    def check_day_night(self, day):
        Obs = self.antenna
        Obs.observer.date = katpoint.Timestamp(day).to_string()
        sunrise = Obs.observer.previous_rising(ephem.Sun()) 
        sunset  = Obs.observer.next_setting(ephem.Sun())
        day_lst = Obs.local_sidereal_time(katpoint.Timestamp(day))
        sunrise_lst = Obs.local_sidereal_time(katpoint.Timestamp(sunrise))
        sunset_lst = Obs.local_sidereal_time(katpoint.Timestamp(sunset))
        sunrise_secs = utils.convert_string_to_secs(str(sunrise_lst))
        sunset_secs = utils.convert_string_to_secs(str(sunset_lst))
        day_lst_secs = utils.convert_string_to_secs(str(day_lst))
        if day_lst_secs > sunrise_secs and day_lst_secs < sunset_secs:
            day_time = 'daytime'
        else:
            day_time = 'nighttime'
            Obs.observer.date = katpoint.Timestamp(day).to_string()
            sunrise = Obs.observer.next_rising(ephem.Sun()) 
            sunset  = Obs.observer.previous_setting(ephem.Sun())
        return day_time, sunrise, sunset 
            
    def simulate_schedule(self, start = '2024-03-12 21:00:00', timespan = 24 * 3 * 60, 
                          method = 'greedy', sb_value = [4,3,2,1], plan = 'long', optim = True):
        assert plan == 'long' or plan == 'short', 'plan has only two values: long or short [as string]'
        dict_priority = utils.assign_ranking(sb_value)
        end = katpoint.Timestamp(start).secs + timespan * 3600
        day = katpoint.Timestamp(start).secs
        data_day = self.data_day.copy(deep=True)
        data_night = self.data_night.copy(deep=True)
        data_avsrss = self.data_avsrss.copy(deep=True)
        data_day = data_day.reset_index(drop=True)
        data_night = data_night.reset_index(drop=True)
        data_avsrss = data_avsrss.reset_index(drop=True)
        df_obs = pd.DataFrame(columns=['Start time (UTC)', 'Start time (LST)','id','description','owner','proposal_id', 'duration', 'product', 'Grade', 'priority', 'avoid_sunrise_sunset', 'night_obs'])
        if method == 'stochastic':
            while day <= end:
                table = self.get_obs_at_time(day, data)
                if len(table) > 0:
                    #table.loc[len(table.index)] = self.dummy # insert dummy row for idle state of telescope
                    table['priority'] = table.apply(lambda x: utils.assign_priority(x), axis = 1)
                    index = random.sample(range(len(table)), 1)[0]
                    convert_to_lst = self.antenna.local_sidereal_time(katpoint.Timestamp(day))
                    lst_obs = str(convert_to_lst)
                    df_obs.loc[len(df_obs.index)] = [katpoint.Timestamp(day).to_string(), lst_obs, int(table['id'].iloc[index]), 
                                                 table['description'].iloc[index], table['owner'].iloc[index], table['proposal_id'].iloc[index], 
                                                 str(datetime.timedelta(seconds=table['simulated_duration'].iloc[index])),
                                                 table['product'].iloc[index], table['Grade'].iloc[index], table['priority'].iloc[index]]
                    day += (table['simulated_duration'].iloc[index] + 1800)
                    data = data[(data['id'] != table['id'].iloc[index])].reset_index(drop=True)
                else:
                    table = self.dummy
                    convert_to_lst = self.antenna.local_sidereal_time(katpoint.Timestamp(day))
                    lst_obs = str(convert_to_lst)
                    df_obs.loc[len(df_obs.index)] = [katpoint.Timestamp(day).to_string(), lst_obs, int(2e13), 
                                                    table['description'], table['owner'], table['proposal_id'], 
                                                    str(datetime.timedelta(seconds=1800)),
                                                    table['product'], table['Grade'], -1, 
                                                    table['avoid_sunrise_sunset'], table['night_obs']]
                    day += 1800 # no-schedule instead?
        elif method == 'greedy':
            while day <= end:
                day_time, sunrise, sunset = self.check_day_night(day) # step 1
                if day_time == 'daytime': # step 2
                    data = pd.concat([data_day, data_avsrss], axis = 0)
                else:
                    data = pd.concat([data_night, data_avsrss], axis = 0)
                table = self.get_obs_at_time(day, data, day_time, sunrise, sunset) # step 3-4, all observations selected in table can be run at this level
                if len(table) > 0:
                    table['priority'] = table.apply(lambda x: utils.assign_priority(x, dict_priority = dict_priority), axis = 1)
                    index = table['priority'].idxmax()
                    convert_to_lst = self.antenna.local_sidereal_time(katpoint.Timestamp(day))
                    lst_obs = str(convert_to_lst)
                    df_obs.loc[len(df_obs.index)] = [katpoint.Timestamp(day).to_string(), lst_obs, table['id'].iloc[index], 
                                                 table['description'].iloc[index], table['owner'].iloc[index], table['proposal_id'].iloc[index], 
                                                 str(datetime.timedelta(seconds=table['simulated_duration'].iloc[index])),
                                                 table['product'].iloc[index], table['Grade'].iloc[index], table['priority'].iloc[index],
                                                    table['avoid_sunrise_sunset'].iloc[index], table['night_obs'].iloc[index]]
                    day += (table['simulated_duration'].iloc[index] + 1800) # this is to include the next build
                    if day_time == 'daytime':
                        data_day = data_day[(data_day['id'] != table['id'].iloc[index])].reset_index(drop=True).copy(deep=True)
                        data_avsrss = data_avsrss[(data_avsrss['id'] != table['id'].iloc[index])].reset_index(drop=True).copy(deep=True)
                    else:
                        data_night = data_night[(data_night['id'] != table['id'].iloc[index])].reset_index(drop=True).copy(deep=True)
                        data_avsrss = data_avsrss[(data_avsrss['id'] != table['id'].iloc[index])].reset_index(drop=True).copy(deep=True)
                else:
                    table = self.dummy
                    convert_to_lst = self.antenna.local_sidereal_time(katpoint.Timestamp(day))
                    lst_obs = str(convert_to_lst)
                    df_obs.loc[len(df_obs.index)] = [katpoint.Timestamp(day).to_string(), lst_obs, int(2e13), 
                                                    table['description'], table['owner'], table['proposal_id'], 
                                                    str(datetime.timedelta(seconds=1800)),
                                                    table['product'], table['Grade'], -1, 
                                                    table['avoid_sunrise_sunset'], table['night_obs']]
                    day += 1800
        if plan == 'long':
            maxim_grade_A = len(df_obs[(df_obs['Grade'] == 'A') & (df_obs['Start time (UTC)'].apply(lambda x: katpoint.Timestamp(x).secs) <= self.mid_time_opt_aproj)]) # optimize the number of A ranked proposal in the first 2 month.
            maxim_grade_B1 = len(df_obs[(df_obs['Grade'] == 'B1') & (df_obs['Start time (UTC)'].apply(lambda x: katpoint.Timestamp(x).secs) <= self.mid_time_opt_b1proj)]) # optimize the number of B1 ranked proposal in the first 3 months.
            minim_idles = len(df_obs[df_obs['priority'] == -1])
            if optim:
                return minim_idles / (100 * MAX_NUM_IDLES) + 1.0 / (maxim_grade_A + 1e-6) + 1.0 / (maxim_grade_B1 + 1e-6) # this works well
            else:
                return minim_idles, len(df_obs[df_obs['Grade'] == 'A']), len(df_obs[df_obs['Grade'] == 'B1']), len(df_obs[df_obs['Grade'] == 'B2']), df_obs
        elif plan == 'short':
            maxim_grade_A = len(df_obs[(df_obs['Grade'] == 'A')]) 
            maxim_grade_B1 = len(df_obs[(df_obs['Grade'] == 'B1')]) 
            minim_idles = len(df_obs[df_obs['priority'] == -1])
            if optim:
                return minim_idles + 1.0 / (maxim_grade_A + 1e-4) + 1.0 / (maxim_grade_B1 + 1e-4)
            else:
                return minim_idles, len(df_obs[df_obs['Grade'] == 'A']), len(df_obs[df_obs['Grade'] == 'B1']), len(df_obs[df_obs['Grade'] == 'B2']), df_obs
        