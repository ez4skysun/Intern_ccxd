import datetime
import os
import traceback
import warnings
from configparser import ConfigParser

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV

import util_and_api.trading_date.trading_date as tdate
from ccxd_dbtool.conn import ConnCCXD, ConnJY_mysql

warnings.filterwarnings('ignore')


class IndustryEstimator():

    def __init__(self, funds, start, end=None, report=False,window=50):
        self.funds = funds
        self.window = window
        if end is not None:
            date_list = tdate.TradingDate(asset_class='stock').get_tradingdates(start, end)
        else:
            date_list = [start]
        self.date_list = pd.to_datetime(date_list)
        self.fund_data = self._get_fund_data()
        self.index_data = self._get_index_data()
        self.report_date = self._get_info()[1]
        self.info = self._get_info()[0]
        self.industry = self._get_industry()
        self.report = self._get_report()
        self.position = self._estimate_position()
        if report:
            self.esreport = self._estimate_report_position()
        
    def _get_fund_data(self):
        date_recent = self.date_list[0].date()
        date_need = (date_recent + datetime.timedelta(days=-self.window*5)).strftime('%Y-%m-%d')
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
        y = y[self.funds]
        #y = df.dropna(axis=0)
        return y

    def _get_index_data(self):
        """
        下载申万一级行业的指数增长率
        """
        date_recent = self.date_list[0].date()
        date_need = (date_recent + datetime.timedelta(days=-self.window*5)).strftime('%Y-%m-%d')
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
        # 聚源给出的2021申万数据有重复行业
        #X = df.drop(['化工','黑色金属','休闲服务','商业贸易','电气设备','纺织服装','餐饮旅游'],axis = 1)
        X = X.pct_change(axis=0)
        return X.dropna()

    def _get_stock_hold_data(self):
        """
        下载各个基金报告期股票持仓，返回的列包括
        'FUNDCODE', 'INFOPUBLDATE', 'REPORTDATE', 'STOCKCODE', 'RATIOINNV'
        """
        year = datetime.datetime.today().year - 2
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
        '''.format(year)
        db = ConnJY_mysql()
        df = db.read(sql)
        df = df.sort_values(['ReportDate', 'fundcode', 'RatioInNV'], ascending=[True, True, False])
        df.index = pd.RangeIndex(len(df))

        # 定期报告公布日期交易日匹配
        today = datetime.date.today().strftime('%Y-%m-%d')
        date_list = tdate.TradingDate(asset_class='stock').get_tradingdates('2021-01-01', today)
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
        df['INFOPUBLDATE'] = df['INFOPUBLDATE'].apply(lambda x: x.strftime('%Y%m%d'))
        df['REPORTDATE'] = df['REPORTDATE'].apply(lambda x: x.strftime('%Y%m%d'))
        hold = {}
        for fund in self.funds:
            temp = df[df['FUNDCODE']==fund]
            hold[fund] = temp
        return hold

    def _get_industry_map(self):
        # 下载股票代码与股票所属行业的对应map
        sql = '''
            select distinct ci.FirstIndustryCode as SW2014F ,ci.FirstIndustryName as INDUSTRY
            from ct_industrytype ci 
            where ci.Standard = 24
            '''
        db = ConnJY_mysql()
        df = db.read(sql)
        df.index = pd.RangeIndex(len(df))
        return df
        
    def _get_info(self):
        """
        通过各个基金的股票持仓与股票行业对应表，计算各个基金于最近报告期的行业投资比例
        """
        stock = self._get_stock_hold_data()
        industry = self._get_industry_map()
        # yd = (datetime.datetime.today()+datetime.timedelta(days=-1)).strftime('%Y/%m/%d')
        # data = pd.read_csv('/data/cooked/Industry/{}/Industry.txt'.format(yd),sep='|')
        data = pd.read_csv('/data/cooked/Industry/2022/2/23/Industry.txt',sep='|')
        data = data[['SECU_CODE','SW2014F']]
        data = data[data['SW2014F']!='None']
        data = data.rename(columns={'SECU_CODE':'STOCKCODE'})
        data = data.merge(industry,how='left',on='SW2014F')
        dic = {}
        report_date = {}
        for fund in self.funds:
            hold = stock[fund]
            report_date[fund] = max(hold.REPORTDATE)
            hold = hold.merge(data,how='left',on='STOCKCODE')
            dic[fund] = hold
        return dic,report_date
    
    def _get_industry(self,report=False):
        """
        计算基金2020年起的历史重仓行业（投资比例占前60%的行业并集）
        """
        ind = {}
        for fund in self.funds:
            df = self.info[fund]
            date_list = list(df.REPORTDATE.unique())
            if report:
                date_list.remove(self.report_date[fund]) 
            ind_temp = set()
            for date in date_list:
                di = df[df.REPORTDATE==date].groupby('INDUSTRY').sum().sort_values(by='RATIOINNV',ascending=False).T
                #ind_temp = ind_temp.union(set(di.columns))
                ind_temp = ind_temp.union(set(di.columns[:di.shape[1]*3//5]))
            ind[fund] = list(ind_temp) if list(ind_temp) else list(self._get_industry_map().INDUSTRY.values)
        return ind
        
    def _estimate_position(self):
        """
        通过ElasticNet基于历史投资比例高的行业的指数增长率对基金增长率进行回归，将处理后的回归系数作为估计的行业持仓比例
        """
        def subjob_func(fund):
            fund_industry = self.industry[fund]
            X = self.index_data[fund_industry]
            y = self.fund_data[fund]
            p = np.zeros_like(X[0:len(self.date_list)])
            for i in range(len(self.date_list)):
                date = pd.to_datetime(self.date_list[i]).date()
                y_temp = y[y.index.date<=date]
                y_temp.dropna(inplace=True)
                y_temp = y_temp[-self.window:]
                X_temp = X.loc[y_temp.index]
                enet = ElasticNetCV(alphas=[0.001,0.0005], 
                                    l1_ratio=[0.9,0.95], 
                                    #n_jobs=-1,
                                    positive=True,fit_intercept=True)
                #enet.fit(X_temp, y_temp, sample_weight=[0.95**i for i in range(self.window)])
                enet.fit(X_temp, y_temp, sample_weight=np.linspace(0,1,len(y_temp)))
                weights = enet.coef_/enet.coef_.sum()
                p[i] = weights
            p_temp = pd.DataFrame(p)
            p_temp.columns = X_temp.columns
            p_temp.index = self.date_list
            p_temp = p_temp.loc[:, (p_temp != 0).any(axis=0)]
            return p_temp
        res = Parallel(n_jobs=60)(delayed(subjob_func)(fund) for fund in self.funds)
        pe = {self.funds[i]:res[i] for i in range(len(self.funds))}
        return pe

    def _estimate_report_position(self):
        """
        估计上一个报告期时的持仓比例
        """
        def subjob_func(fund):
            fund_industry = self._get_industry(report=True)[fund]
            y = self.fund_data[fund]
            X = self.index_data[fund_industry]
            rd = datetime.datetime.strptime(self.report_date[fund],'%Y%m%d').date()
            #date_need = rd + datetime.timedelta(days=-self.window*3)
            y = y[y.index.date<=rd]
            y.dropna(inplace=True)
            y = y[-self.window:]
            X = X.loc[y.index]
            enet = ElasticNetCV(alphas=[0.001,0.0005], 
                                l1_ratio=[0.9,0.95], 
                                n_jobs=-1,
                                positive=True,fit_intercept=True)
            #enet.fit(X, y, sample_weight=[0.95**i for i in range(len(y))]) # 等比衰减
            enet.fit(X, y, sample_weight=np.linspace(0,1,len(y))) # 等差衰减
            p_temp = pd.DataFrame(enet.coef_/enet.coef_.sum()).T
            p_temp.columns = X.columns
            p_temp.index = pd.DatetimeIndex([self.report_date[fund]])
            p_temp = p_temp.loc[:, (p_temp != 0).any(axis=0)]
            return p_temp
        res = Parallel(n_jobs=60)(delayed(subjob_func)(fund) for fund in self.funds)
        pe = {self.funds[i]:res[i] for i in range(len(self.funds))}
        return pe

    def _get_report(self):
        """
        通过行业持仓数据计算归一化后的行业持仓比例
        """
        rep = {}
        for fund in self.funds:
            df = self.info[fund]
            df = df[df.REPORTDATE==self.report_date[fund]]
            df = df.groupby('INDUSTRY').sum().sort_values(by='RATIOINNV',ascending=False).T
            df = df/float(df.sum(axis=1))
            df.index = pd.DatetimeIndex([self.report_date[fund]])
            rep[fund] = df
        return rep

    def deviation(self,report=False):
        """
        计算模型给出的持仓比例与报告期比例的绝对仓位偏差，0.6以下代表良好
        Args:
            report:default=False

        Returns:
            A dict mapping keys to the corresponding fund. 
            Each row is represented as the absolute deviation. 
        """
        def abs_dev(x,y):
            subset = set(x.columns).intersection(set(y.columns))
            x = x.reset_index(drop=True)
            y = y.reset_index(drop=True)
            ans = (x-y).abs().sum(axis=1).values
            for col in x.columns:
                if col not in subset:
                    ans += x[col].abs().values
            for col in y.columns:
                if col not in subset:
                    ans += y[col].abs().values
            return float(ans)
        deviation = {}
        predict = self.position if not report else self.esreport
        hold = self.report
        for fund in self.funds:
            temp = []
            for i in range(predict[fund].shape[0]):
                x = pd.DataFrame(predict[fund].iloc[i]).T
                y = hold[fund].copy(deep=True)
                temp.append(abs_dev(x,y))
            temp = pd.Series(temp,index=predict[fund].index)
            deviation[fund] = temp
        df = pd.DataFrame(deviation)
        return df

class SectorEstimator(IndustryEstimator):
    
    def _get_index_data(self):
        path = 'industry_sector.config'
        config = ConfigParser()
        config.read(path, encoding='UTF-8')
        date_recent = self.date_list[0].date()
        date_need = (date_recent + datetime.timedelta(days=-self.window*5)).strftime('%Y-%m-%d')
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
        for item in config.items('sector'):
            X[item[0]] = X[item[1].split(',')].mean(axis=1)
        return X[config.options('sector')].dropna()

    def _get_report(self):
        """
        通过行业持仓数据计算归一化后的行业聚类持仓比例
        """
        path = 'industry_sector.config'
        config = ConfigParser()
        config.read(path, encoding='UTF-8')
        rep = {}
        for fund in self.funds:
            df = self.info[fund]
            df = df[df.REPORTDATE==self.report_date[fund]]
            df = df.groupby('INDUSTRY').sum().sort_values(by='RATIOINNV',ascending=False)
            for item in config.items('sector'):
                df.loc[item[0]] = df.loc[df.index.isin(item[1].split(','))].sum()
            df = df.loc[df.index.isin(config.options('sector'))].sort_values(by='RATIOINNV',ascending=False)
            df = (df/float(df.sum(axis=0))).T
            df.index = pd.DatetimeIndex([self.report_date[fund]])
            rep[fund] = df
        return rep
    
    def _estimate_position(self):
        """
        通过ElasticNet基于分类后类内行业指数平均增长率对基金增长率进行回归，将处理后的回归系数作为估计的分类持仓比例
        """
        def subjob_func(fund):
            X = self.index_data
            y = self.fund_data[fund]
            p = np.zeros_like(X[0:len(self.date_list)])
            for i in range(len(self.date_list)):
                date = pd.to_datetime(self.date_list[i]).date()
                y_temp = y[y.index.date<=date]
                y_temp.dropna(inplace=True)
                y_temp = y_temp[-self.window:]
                X_temp = X.loc[y_temp.index]
                # p = np.zeros_like(X_temp[0:X_temp.shape[0]-self.window])
                enet = ElasticNetCV(alphas=[0.001,0.0005], 
                                    l1_ratio=[0.9,0.95], 
                                    #n_jobs=-1,
                                    positive=True,fit_intercept=True)
                #enet.fit(X_temp, y_temp, sample_weight=[0.95**i for i in range(self.window)])
                enet.fit(X_temp, y_temp, sample_weight=np.linspace(0,1,len(y_temp)))
                weights = enet.coef_/enet.coef_.sum()
                p[i] = weights
            p_temp = pd.DataFrame(p)
            p_temp.columns = X_temp.columns
            p_temp.index = self.date_list
            p_temp = p_temp.loc[:, (p_temp != 0).any(axis=0)]
            return p_temp
        res = Parallel(n_jobs=60)(delayed(subjob_func)(fund) for fund in self.funds)
        pe = {self.funds[i]:res[i] for i in range(len(self.funds))}
        return pe

    def _estimate_report_position(self):
        """
        估计上一个报告期时的分类持仓比例
        """
        def subjob_func(fund):
            y = self.fund_data[fund]
            X = self.index_data
            rd = datetime.datetime.strptime(self.report_date[fund],'%Y%m%d').date()
            #date_need = rd + datetime.timedelta(days=-self.window*3)
            y = y[y.index.date<=rd]
            y.dropna(inplace=True)
            y = y[-self.window:]
            X = X.loc[y.index]
            enet = ElasticNetCV(alphas=[0.001,0.0005], 
                                l1_ratio=[0.9,0.95], 
                                n_jobs=-1,
                                positive=True,fit_intercept=True)
            #enet.fit(X, y, sample_weight=[0.95**i for i in range(len(y))]) # 等比衰减
            enet.fit(X, y, sample_weight=np.linspace(0,1,len(y))) # 等差衰减
            p_temp = pd.DataFrame(enet.coef_/enet.coef_.sum()).T
            p_temp.columns = X.columns
            p_temp.index = pd.DatetimeIndex([self.report_date[fund]])
            p_temp = p_temp.loc[:, (p_temp != 0).any(axis=0)]
            return p_temp
        res = Parallel(n_jobs=60)(delayed(subjob_func)(fund) for fund in self.funds)
        pe = {self.funds[i]:res[i] for i in range(len(self.funds))}
        return pe


def main():
	pass


if __name__ == '__main__':
	main()
