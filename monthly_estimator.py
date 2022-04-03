import os
import traceback
import warnings
from configparser import ConfigParser
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV

import util_and_api.trading_date.trading_date as tdate
from ccxd_dbtool import conn
from ccxd_dbtool.conn import ConnCCXD, ConnJY_mysql
from data_dir_tools import check_lastupdate, collect, write_datedir

warnings.filterwarnings('ignore')

fn = 'industry_sector.config'
config = ConfigParser()
parent_dir = os.path.dirname(os.path.abspath(fn))
config.read(os.path.join(parent_dir ,fn), encoding='UTF-8')   #读取配置文件采用绝对路径


class Estimator():

    def __init__(self, funds, start, end=None, report=False,window=50):
        self.funds = funds
        self.window = window
        if end is not None:
            date_list = tdate.TradingDate(asset_class='stock').get_tradingdates(start, end)
        else:
            date_list = [start]
        dl = pd.to_datetime(date_list)
        dl = pd.DataFrame(dl).rename(columns={0:'DATE'})
        dl['YEAR'] = dl.DATE.apply(lambda x:x.year)
        dl['MONTH'] = dl.DATE.apply(lambda x:x.month)
        date_list = list(dl.groupby(['YEAR','MONTH']).max().DATE)
        self.date_list = pd.to_datetime(date_list)
        self.fund_data = self._get_fund_data()
        self.index_data = self._get_index_data().dropna()
        self.stock_hold_data = self._get_stock_hold_data()
        self.items = [('金融', '银行,非银金融,房地产'),
                ('资源', '采掘,有色金属,钢铁'),
                ('材料', '化工,轻工制造,建筑材料'),
                ('消费', '食品饮料,家用电器,医药生物,休闲服务'),
                ('制造', '汽车,机械设备,电气设备'),
                ('tmt', '计算机,通信,传媒,电子'),
                ('公共', '交通运输,建筑装饰,公用事业'),
                ('综合', '纺织服装,综合,商业贸易'),
                ('农林', '农林牧渔'),
                ('军工', '国防军工')]
        index_data = self.index_data.copy(deep=True)
        for item in self.items:
            index_data[item[0]] = self.index_data[item[1].split(',')].mean(axis=1)
        self.sector_data = index_data[['金融','资源','材料','消费','制造','tmt','公共','综合','农林','军工']]
        del index_data
        self.industry_data = self._get_industry()
        
    def _get_fund_data(self):
        date_recent = self.date_list[0].date()
        date_need = (date_recent + timedelta(days=-self.window*5)).strftime('%Y-%m-%d')
        """
        下载各个基金在估计窗口前一段时间内的复权净值增长率
        """
        sql = '''
            select mf.TradingDay ,mf.NVRDailyGrowthRate ,sc.SecuCode as fundcode,sc.SecuMarket as fundmkt
            from MF_FundNetValueRe as mf
            left join (select InnerCode, SecuCode, SecuMarket from secumain s 
                        union select InnerCode, SecuCode, SecuMarket from hk_secumain hs ) sc 
            on mf.InnerCode = sc.InnerCode
            where mf.TradingDay >= str_to_date('{}', '%Y-%m-%d') 
            '''.format(date_need)
        db = ConnJY_mysql()
        df = db.read(sql)
        df.index = pd.RangeIndex(len(df))

        # 基金后缀
        df['fundcode_wind'] = df['fundcode'].apply(lambda x: x + '.OF')
        df['fund_suf'] = ''
        df.loc[df['fundmkt'].isna(), 'fund_suf'] = '.OF'
        df.loc[df['fundmkt'] == 90, 'fund_suf'] = '.SZ'
        df.loc[df['fundmkt'] == 83, 'fund_suf'] = '.SH'
        df['fundcode_mkt'] = df['fundcode'] + df['fund_suf']
        
        y = pd.pivot_table(df, values='NVRDailyGrowthRate', index='TradingDay', columns='fundcode_mkt')
        dl = tdate.TradingDate(asset_class='stock').get_tradingdates(date_need, self.date_list[-1])
        y = y.loc[dl]
        fp = set(y.columns).intersection(set(self.funds))
        self.funds = list(fp)
        y = y[fp]
        return y.dropna(axis=0,how='all')

    def _get_index_data(self):
        """
        下载申万一级行业的指数增长率
        """
        date_recent = self.date_list[0].date()
        date_need = (date_recent + timedelta(days=-self.window*5)).strftime('%Y-%m-%d')
        sql = '''
        select tb.* ,qi.TradingDay ,qi.ClosePrice ,qi.TurnoverValue
        from (select distinct lc.IndexCode ,ci.FirstIndustryName as INDUSTRY,lc.UpdateTime 
            from lc_corrindexindustry lc 
            left join ct_industrytype ci 
            on lc.IndustryStandard = ci.Standard and lc.IndustryCode = ci.IndustryCode
            where lc.IndustryStandard = 24 and lc.UpdateTime >= str_to_date('2021-1-1', '%Y-%m-%d') ) tb
        left join qt_indexquote qi 
        on tb.Indexcode = qi.InnerCode 
        where qi.TradingDay >= str_to_date('{}', '%Y-%m-%d')
        '''.format(date_need)
        db = ConnJY_mysql()
        df = db.read(sql)
        df.index = pd.RangeIndex(len(df))
        X = pd.pivot_table(df, values='ClosePrice', index='TradingDay', columns='INDUSTRY')
        X = X.pct_change(axis=0)
        return X.dropna()

    def _get_stock_hold_data(self):
        """
        下载各个基金报告期股票持仓，返回的列包括
        'FUNDCODE', 'INFOPUBLDATE', 'REPORTDATE', 'STOCKCODE', 'RATIOINNV'
        """
        year_recent = self.date_list[0].date().year - 2
        sql = '''
            select tb.*, sc.SecuCode as stockcode, sc.SecuMarket
            from 
                (select 
                    s.SecuCode as fundcode, s.SecuMarket as fundmkt, ms.InnerCode,
                    ms.InfoPublDate, ms.ReportDate, ms.InvestType, ms.StockInnerCode, ms.RatioInNV
                from mf_stockportfoliodetail ms left join secumain s 
                on ms.InnerCode = s.InnerCode 
                where 
                    ms.InvestType != 3 and 
                    ms.ReportDate >= str_to_date('{}-01-01', '%Y-%m-%d'))tb 
                left join
                    (select InnerCode, SecuCode, SecuMarket from secumain s 
                        union select InnerCode, SecuCode, SecuMarket from hk_secumain hs ) sc
            on tb.StockInnerCode = sc.InnerCode
        '''.format(year_recent)
        db = ConnJY_mysql()
        df = db.read(sql)
        df = df.sort_values(['ReportDate', 'fundcode', 'RatioInNV'], ascending=[True, True, False])
        df.index = pd.RangeIndex(len(df))

        # 定期报告公布日期交易日匹配
        today = date.today().strftime('%Y-%m-%d')
        date_list = tdate.TradingDate(asset_class='stock').get_tradingdates('{}-01-01'.format(year_recent), today)
        date_list = pd.to_datetime(date_list)
        pubdate = list(set(df['InfoPublDate']))
        pubdate.sort()
        pubdate_td = [x if x in date_list else date_list[date_list >= x][0] for x in pubdate]
        pubdate_map = dict(zip(pubdate, pubdate_td))
        df['InfoPublDate_td'] = [pubdate_map[x] for x in df['InfoPublDate']]

        # 基金后缀
        df['fundcode_wind'] = df['fundcode'].apply(lambda x: x + '.OF')
        df['fund_suf'] = ''
        df.loc[df['fundmkt'].isna(), 'fund_suf'] = '.OF'
        df.loc[df['fundmkt'] == 90, 'fund_suf'] = '.SZ'
        df.loc[df['fundmkt'] == 83, 'fund_suf'] = '.SH'
        df['fundcode_mkt'] = df['fundcode'] + df['fund_suf']

        # 股票后缀
        df['stock_suf'] = ''
        df.loc[df['SecuMarket'] == 72, 'stock_suf'] = '.HK'
        df.loc[df['SecuMarket'] == 83, 'stock_suf'] = '.SH'
        df.loc[df['SecuMarket'] == 90, 'stock_suf'] = '.SZ'
        df.loc[df['SecuMarket'] == 81, 'stock_suf'] = '.BJ'
        df.loc[df['stockcode'] == '400080', 'stock_suf'] = '.NQ'
        df['stockcode_wind'] = df['stockcode'] + df['stock_suf']

        # 返回的列
        df = df[['fundcode_mkt', 'InfoPublDate_td', 'ReportDate', 'stockcode_wind', 'RatioInNV']]
        df.columns = ['FUNDCODE', 'INFOPUBLDATE', 'REPORTDATE', 'STOCKCODE', 'RATIOINNV']
        return df
        
    def _get_industry(self):
        sql = '''
            select distinct ci.FirstIndustryCode as SW2014F ,ci.FirstIndustryName as INDUSTRY
            from ct_industrytype ci 
            where ci.Standard = 24
            '''
        db = ConnJY_mysql()
        industry_map = db.read(sql)
        industry_map.index = pd.RangeIndex(len(industry_map))
        df = self.stock_hold_data
        indmap = collect(fn='Industry.txt', rootpath=Path('/data/cooked/Industry'), start='2008-01-01', end='2022-02-08', sep='|')
        indmap = indmap[['SECU_CODE','TRADINGDAY','SW2014F']]
        indmap = indmap[indmap['SW2014F']!='None']
        indmap = indmap.rename(columns={'SECU_CODE':'STOCKCODE'})
        indmap = indmap.merge(industry_map,how='left',on='SW2014F')
        indmap['YEAR'] = indmap.TRADINGDAY.apply(lambda x:str(x)[:4])
        df['YEAR'] = df.REPORTDATE.apply(lambda x:str(x)[:4])
        indmap.pop('TRADINGDAY')
        indmap.drop_duplicates(inplace=True)
        indmap.dropna(inplace=True)
        data= df.merge(indmap,how='left',on=['STOCKCODE','YEAR']).dropna()
        data.pop('SW2014F')

        industry_data = {}
        for fund in set(self.funds):
            hold = data[data.FUNDCODE==fund]
            temp = {}
            for i in range(len(self.date_list)):
                date = self.date_list[i]
                hold_temp = hold[(hold.INFOPUBLDATE<=date.strftime("%Y%m%d"))&(hold.INFOPUBLDATE>=(date+timedelta(days=-730)).strftime("%Y%m%d"))]
                ind = list(hold_temp.groupby('INDUSTRY').sum().sort_values(by='RATIOINNV',ascending=False).T.columns)
                temp[date.date()] = ind[:4*len(ind)//5]
                industry_data[fund] = temp
        return industry_data
        
    def history_industry_estimate(self,window=40,njobs=50):
        def subjob_func(fund):
            X = self.index_data
            allind = list(X.columns)
            y = self.fund_data[fund]
            industry = self.industry_data[fund]
            p = np.zeros_like(X[0:len(self.date_list)])
            for i in range(len(self.date_list)):
                date = self.date_list[i].date()
                prevdate = pd.to_datetime(tdate.TradingDate(asset_class='stock').prev_tradingday(date,count=window)).date()
                X_temp = X[(X.index.date<=date)&(X.index.date>=prevdate)]
                if industry[date]:
                    X_temp = X_temp[industry[date]]
                    partind = list(X_temp.columns)
                    y_temp = y.loc[y.index.isin(set(X_temp.index))].dropna()
                    if y_temp.shape[0] >= 3*window//4:
                        X_temp = X_temp.loc[y_temp.index]
                        enet = ElasticNetCV(alphas=[0.001,0.0005,0.005], 
                                            l1_ratio=[0.9,0.95,0.85], 
                                            #n_jobs=-1,
                                            positive=True,fit_intercept=True,max_iter=5000)
                        enet.fit(X_temp, y_temp, sample_weight=np.linspace(0,1,len(y_temp)))
                        weights = enet.coef_/enet.coef_.sum()
                        for j in range(len(weights)):
                            p[i][allind.index(partind[j])] = weights[j]
                else:
                    p[i] = np.full(X.shape[1],np.nan)
            p_temp = pd.DataFrame(p)
            p_temp.columns = X.columns
            p_temp.index = self.date_list
            return p_temp
        res = Parallel(n_jobs=njobs)(delayed(subjob_func)(fund) for fund in self.funds)
        pe = {self.funds[i]:res[i] for i in range(len(self.funds))}
        return pe
    
    def all_industry_estimate(self,window=60,njobs=50):
        def subjob_func(fund):
            X = self.index_data
            y = self.fund_data[fund]
            p = np.zeros_like(X[0:len(self.date_list)])
            for i in range(len(self.date_list)):
                date = self.date_list[i].date()
                prevdate = pd.to_datetime(tdate.TradingDate(asset_class='stock').prev_tradingday(date,count=window)).date()
                X_temp = X[(X.index.date<=date)&(X.index.date>=prevdate)]
                y_temp = y.loc[y.index.isin(set(X_temp.index))].dropna()
                if y_temp.shape[0] >= 5*window//6:
                    X_temp = X_temp.loc[y_temp.index]
                    enet = ElasticNetCV(alphas=[0.001,0.0005,0.005], 
                                        l1_ratio=[0.9,0.95,0.85], 
                                        #n_jobs=-1,
                                        positive=True,fit_intercept=True,max_iter=5000)
                    enet.fit(X_temp, y_temp, sample_weight=np.linspace(0,1,len(y_temp)))
                    weights = enet.coef_/enet.coef_.sum()
                    p[i] = weights
                else:
                    p[i] = np.full(X.shape[1],np.nan)
            p_temp = pd.DataFrame(p)
            p_temp.columns = X.columns
            p_temp.index = self.date_list
            
            return p_temp
        res = Parallel(n_jobs=njobs)(delayed(subjob_func)(fund) for fund in self.funds)
        pe = {self.funds[i]:res[i] for i in range(len(self.funds))}
        return pe

    def sector_estimate(self,window=25,njobs=50):
        def subjob_func(fund):
            X = self.sector_data
            y = self.fund_data[fund]
            p = np.zeros_like(X[0:len(self.date_list)])
            for i in range(len(self.date_list)):
                date = self.date_list[i].date()
                prevdate = pd.to_datetime(tdate.TradingDate(asset_class='stock').prev_tradingday(date,count=window)).date()
                X_temp = X[(X.index.date<=date)&(X.index.date>=prevdate)]
                y_temp = y.loc[y.index.isin(set(X_temp.index))].dropna()
                if y_temp.shape[0] >= 4*window//5:
                    X_temp = X_temp.loc[y_temp.index]
                    enet = ElasticNetCV(alphas=[0.001,0.0005,0.005], 
                                        l1_ratio=[0.9,0.95,0.85], 
                                        #n_jobs=-1,
                                        positive=True,fit_intercept=True,max_iter=5000)
                    enet.fit(X_temp, y_temp, sample_weight=np.linspace(0,1,len(y_temp)))
                    weights = enet.coef_/enet.coef_.sum()
                    p[i] = weights
                else:
                    p[i] = np.full(X.shape[1],np.nan)
            p_temp = pd.DataFrame(p)
            p_temp.columns = X.columns
            p_temp.index = self.date_list
            
            return p_temp
        res = Parallel(n_jobs=njobs)(delayed(subjob_func)(fund) for fund in self.funds)
        pe = {self.funds[i]:res[i] for i in range(len(self.funds))}
        return pe

def main():
    fp = pd.read_parquet('/work/ysun/fof_project_for_zhaoyin/equityfund_analyzer_data/fund_style_test_pool.parquet')
    fp.index = pd.to_datetime(fp.index)
    fp = fp.loc[fp.index>=datetime(2014,6,1)]
    pool = list(fp.loc[:, (fp).any(axis=0)].columns)
    ES = Estimator(pool,start='2014-06-30',end='2022-02-28')
    def ans(position):
        pos = {}
        for fund in set(position.keys()):
            temp = position[fund].reset_index()
            temp['FUNDCODE'] = fund
            pos[fund] = temp.rename(columns={'index':'DATE'})
        da = pd.concat([pos[fund] for fund in set(position.keys())],join='outer')
        return da.round(5).set_index('FUNDCODE')

    sec = ES.sector_estimate(njobs=60)
    sector = ans(sec)
    sector.to_csv('/home/qiantianyang/work/' + 'sector.csv')
    print('sec finish')

    # his = ES.history_industry_estimate(njobs=24)
    # hisind = ans(his)
    # hisind.to_csv('/home/qiantianyang/work/' + 'hisind.csv')
    # print('his finish')

    al = ES.all_industry_estimate(njobs=60)
    allind = ans(al)
    allind.to_csv('/home/qiantianyang/work/' + 'allind.csv')
    print('all finish')
    

if __name__ == '__main__':
	main()

