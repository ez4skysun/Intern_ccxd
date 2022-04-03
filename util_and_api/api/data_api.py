'''
author: ah
date：2021-07-13 
'''
from util_and_api.api.ccxd_utils_dir_util import *
import util_and_api.trading_date.trading_date as tdate
import util_and_api.util.parser as parser
import util_and_api.util.mptool as mt 

import subprocess as sub
import wrds
import os
import gzip
import time
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import datetime
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataAPI:
    def __init__(self, market, asset_class):
        self.__market = market
        self.__asset_class = asset_class
        pass 
    
    def get_events_between(self, domain, event_name, begin_date, end_date, suffix, delay):
        print('get data between: ', begin_date, end_date)
        calendar = tdate.TradingDate('WIND', self.__market, self.__asset_class, 'Basic')
        real_end_date = calendar.prev_day(end_date, delay)
        trading_date = calendar.get_tradingdates(begin_date, real_end_date)
        df_list = []
        for date in trading_date:
            filename = DirLocator(self.__market, self.__asset_class).cooked_daily_file_name(domain, date, event_name + suffix)
            if os.path.exists(filename):
                # print(filename)
                with gzip.open(filename) as filename_file:
                    df = pd.read_csv(filename_file, encoding='gbk', low_memory=False)
                    df_list.append(df)
        # print(df_list)
        if len(df_list) == 0:
            return None
        else:
            return pd.concat(df_list, sort=False).reset_index(drop=True)
    
    def get_events_rolling(self, domain, event_name, today, window, suffix, delay):
        calendar = tdate.TradingDate('WIND', self.__market, self.__asset_class, 'Basic')
        begin_date = calendar.prev_day(today, window)
        return self.get_events_between(domain, event_name, begin_date, today, suffix, delay)