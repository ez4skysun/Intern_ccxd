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

class DataAPIV2:
    def __init__(self, market, asset_class):
        self.__market = market
        self.__asset_class = asset_class
        self.__data = None 
        pass 
    
    def get_events_between(self, domain, event_name, begin_date, end_date, suffix, delay):
        print('* ' * 30)
        calendar = tdate.TradingDate('WIND', self.__market, self.__asset_class, 'Basic')
        real_end_date = calendar.prev_day(end_date, delay)
        trading_date = calendar.get_tradingdates(begin_date, real_end_date)
        print('wanted: ', trading_date[0], trading_date[-1], len(trading_date))
        df_list = []
        for date in trading_date:
            filename = DirLocator(self.__market, self.__asset_class).cooked_daily_file_name(domain, date, event_name + suffix)
            if os.path.exists(filename):
                # print(filename)
                with gzip.open(filename) as filename_file:
                    df = pd.read_csv(filename_file, encoding='gbk')
                    df['dtl'] = date 
                    df_list.append(df)
        print('got: ', trading_date[0], trading_date[-1], len(trading_date))
        if len(df_list) == 0:
            return None 
        else:
            return pd.concat(df_list, sort=False).reset_index(drop=True)

    def get_events_between_v2(self, domain, event_name, begin_date, end_date, suffix, delay):
        print('* ' * 30)
        calendar = tdate.TradingDate('WIND', self.__market, self.__asset_class, 'Basic')
        real_end_date = calendar.prev_day(end_date, delay)
        trading_date = calendar.get_tradingdates(begin_date, real_end_date)
        print('wanted: ', trading_date[0], trading_date[-1], len(trading_date))

        if self.__data is None:
            df_list = []
            for date in trading_date:
                filename = DirLocator(self.__market, self.__asset_class).cooked_daily_file_name(domain, date, event_name + suffix)
                if os.path.exists(filename):
                    # print(filename)
                    with gzip.open(filename) as filename_file:
                        df = pd.read_csv(filename_file, encoding='gbk')
                        df['dtl'] = date 
                        df_list.append(df)
            print('got: ', trading_date[0], trading_date[-1], len(trading_date))
            if len(df_list) == 0:
                return None 
            else:
                self.__data = pd.concat(df_list, sort=False).reset_index(drop=True)
                return self.__data 

        else:
            dtl_list = sorted(list(set(self.__data['dtl'].values)))
            b = []
            e = []
            for date in dtl_list:
                if date not in trading_date:
                    b.append(date)
            for date in trading_date:
                if date not in dtl_list:
                    e.append(date)
            print('deleted: ', b)
            print('added: ', e)
            
            if len(b) > 0:
                self.__data = self.__data[self.__data['dtl'] > b[-1]]
            df_list = [self.__data]
            for date in e:
                filename = DirLocator(self.__market, self.__asset_class).cooked_daily_file_name(domain, date, event_name + suffix)
                if os.path.exists(filename):
                    # print(filename)
                    with gzip.open(filename) as filename_file:
                        df = pd.read_csv(filename_file, encoding='gbk')
                        df['dtl'] = date 
                        df_list.append(df)
            self.__data = pd.concat(df_list, sort=False).reset_index(drop=True)

            dtl_list = sorted(list(set(self.__data['dtl'].values)))
            print('got: ', dtl_list[0], dtl_list[-1], len(dtl_list))
            return self.__data
    
    def get_events_rolling(self, domain, event_name, today, window, suffix, delay):
        calendar = tdate.TradingDate('WIND', self.__market, self.__asset_class, 'Basic')
        begin_date = calendar.prev_day(today, window)
        return self.get_events_between(domain, event_name, begin_date, today, suffix, delay)

    def get_events_rolling_v2(self, domain, event_name, today, window, suffix, delay):
        calendar = tdate.TradingDate('WIND', self.__market, self.__asset_class, 'Basic')
        begin_date = calendar.prev_day(today, window)
        return self.get_events_between_v2(domain, event_name, begin_date, today, suffix, delay)