import numpy as np
import katpoint

NORMALIZE_TIME = 3600 * 8
RANKING = ['A', 'B1', 'B2', None]
def assign_priority(x, dict_priority = {'A': 20, 'B1': 14, 'B2': 10, None: 1}):
    return dict_priority[x.Grade] * (x.simulated_duration / NORMALIZE_TIME)

def assign_ranking(x):
    dict_priority = {}
    for i in range(len(RANKING)):
        dict_priority[RANKING[i]] = x[i]
    return dict_priority

def convert_string_to_secs(lst_str):
    '''
    This routine converts lst in string format to integer
    '''
    hours = lst_str.split(':')[0]
    mins = lst_str.split(':')[1]
    return 3600 * int(hours) + 60 * int(mins)

def get_grade(x, dict_):
    '''
    This routine is used to read the value of a dictionary dict_ corresponding to a key x.
    The "try ... except" is used to bypass cases where the key x is not part of the dictionary keys.
    '''
    try:
        return dict_[x]
    except:
        return None

def get_duration_secs(timestr):
    ftr = [3600,60,1]
    return sum([a*b for a,b in zip(ftr, map(int,timestr.split(':')))])

def get_schedule_score(table):
    '''
    This routine is used to compute the score of a proposed schedule 
    within a given timespan
    '''
    len_table = len(table)
    # compute the total duration of an entire table
    timestr = table['duration'].iloc[-1]
    initial_time = katpoint.Timestamp(table['Start time (UTC)'].iloc[0])
    end_time = katpoint.Timestamp(table['Start time (UTC)'].iloc[-1]) + get_duration_secs(timestr)
    total_duration = end_time - initial_time - 1800 # to account for the build of the first SB in the table
    # normalize the length of an SB with the total length from the table
    normalize_length_sb = {int(table['id'].iloc[i]): (get_duration_secs(table['duration'].iloc[i]) / total_duration, table['priority'].iloc[i]) for i in range(len_table) if table['Grade'].iloc[i] != 'Idle'}
    # get the no-schedule score
    score_nosched = sum([-1800/total_duration for i in range(len_table) if table['Grade'].iloc[i] == 'Idle'])
    # get the total score
    score = sum([v[0]*v[1] for k,v in normalize_length_sb.items()])
    # COMPUTING THE TELESCOPE USAGE
    # number of actual obs SBs
    number_sbs = len(table)# this should include Idle since they are in the list....len(table[table['Grade'] != 'Idle'])
    number_idle = len(table[table['Grade'] == 'Idle'])
    #print(number_idle, number_sbs)
    total_time_loss = number_sbs * 1800 + number_idle * 3600
    usage = 100 * (1 - total_time_loss / total_duration)
    return score + score_nosched, usage

