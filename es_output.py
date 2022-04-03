import calendar
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

import util_and_api.trading_date.trading_date as tdate
from ccxd_dbtool import conn
from ccxd_dbtool.conn import ConnCCXD, ConnJY_mysql
from data_dir_tools import check_lastupdate, collect, write_datedir

warnings.filterwarnings('ignore')

def get_stock_hold_data():
    """
    下载各个基金报告期股票持仓，返回的列包括
    'FUNDCODE', 'INFOPUBLDATE', 'REPORTDATE', 'STOCKCODE', 'RATIOINNV'
    """
    year = 2014
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
    today = date.today().strftime('%Y-%m-%d')
    date_list = tdate.TradingDate(asset_class='stock').get_tradingdates('2014-01-01', today)
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

def get_position():
    sql = '''
            select distinct ci.FirstIndustryCode as SW2014F ,ci.FirstIndustryName as INDUSTRY
            from ct_industrytype ci 
            where ci.Standard = 24
            '''
    db = ConnJY_mysql()
    industry_map = db.read(sql)
    industry_map.index = pd.RangeIndex(len(industry_map))
    indmap = collect(fn='Industry.txt', rootpath=Path('/data/cooked/Industry'), start='2008-01-01', end='2022-02-08', sep='|')
    indmap = indmap[['SECU_CODE','TRADINGDAY','SW2014F']]
    indmap = indmap[indmap['SW2014F']!='None']
    indmap = indmap.rename(columns={'SECU_CODE':'STOCKCODE'})
    indmap = indmap.merge(industry_map,how='left',on='SW2014F')
    indmap['YEAR'] = indmap.TRADINGDAY.apply(lambda x:str(x)[:4])
    df = get_stock_hold_data()
    df.REPORTDATE.apply(lambda x:str(x.year))
    df['YEAR'] = df.REPORTDATE.apply(lambda x:str(x)[:4])
    df['YM'] = df.REPORTDATE.apply(lambda x:str(x.year)+str(x.month))
    indmap.pop('TRADINGDAY')
    indmap.drop_duplicates(inplace=True)
    indmap.dropna(inplace=True)
    data= df.merge(indmap,how='left',on=['STOCKCODE','YEAR']).dropna()
    dic = {}
    items = [('金融', '银行,非银金融,房地产'),
                ('资源', '采掘,有色金属,钢铁'),
                ('材料', '化工,轻工制造,建筑材料'),
                ('消费', '食品饮料,家用电器,医药生物,休闲服务'),
                ('制造', '汽车,机械设备,电气设备'),
                ('tmt', '计算机,通信,传媒,电子'),
                ('公共', '交通运输,建筑装饰,公用事业'),
                ('综合', '纺织服装,综合,商业贸易'),
                ('农林', '农林牧渔'),
                ('军工', '国防军工')]
    for item in items:
        for ind in item[1].split(','):
            dic[ind] = item[0]
    data['SECTOR'] = data.INDUSTRY.apply(lambda x:dic[x])
    data.pop('SW2014F')
    data.pop('YEAR')
    print('get position finish')
    return data

def abs_dev(a,b):
    x = a.reset_index(drop=True)
    y = b.reset_index(drop=True)
    ans = (x-y).abs().dropna(axis=1).sum(axis=1).values
    for col in set(y.columns):
        if col not in set(x.columns):
            ans += y[col].values
    ans = float(ans)
    if ans != 0:
        return ans
    else:
        return np.nan


if __name__ == '__main__':
    sec = pd.read_csv('/home/qiantianyang/work/sector.csv')
    his = pd.read_csv('/home/qiantianyang/work/hisind.csv')
    al = pd.read_csv('/home/qiantianyang/work/allind.csv')
    sec = sec.loc[(sec.iloc[:,2:] != 0).any(axis=1)]
    his = his.loc[(his.iloc[:,2:] != 0).any(axis=1)]
    al = al.loc[(al.iloc[:,2:] != 0).any(axis=1)]
    industry = al.columns[2:]
    sector = sec.columns[2:]
    data = get_position()
    for m in [sec,his,al]:
        m.DATE = m.DATE.apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
        m['YM'] = m.DATE.apply(lambda x:str(x.year)+str(x.month))
    pool = list(sec.FUNDCODE.unique())
    def subjob_func(fund):
        temp_data = data[data.FUNDCODE == fund]
        temp_sec = sec[sec.FUNDCODE == fund]
        temp_his = his[his.FUNDCODE == fund]
        temp_al = al[al.FUNDCODE == fund]
        score = pd.DataFrame(columns=['SEC', 'HIS', 'ALL'], index=temp_data.YM.unique())
        score['DATE'] = temp_data.YM.unique()
        for date in data.YM.unique():
            real_ind = temp_data[temp_data.YM == date].groupby('INDUSTRY').sum().apply(lambda x:x/sum(x)).T
            real_sec = temp_data[temp_data.YM == date].groupby('SECTOR').sum().apply(lambda x:x/sum(x)).T
            rp = temp_data[temp_data.YM == date].REPORTDATE
            score.loc[date,'SEC'] = abs_dev(temp_sec[temp_sec.YM == date][sector],real_sec)
            score.loc[date,'HIS'] = abs_dev(temp_his[temp_his.YM == date][industry],real_ind)
            score.loc[date,'ALL'] = abs_dev(temp_al[temp_al.YM == date][industry],real_ind)
        return score
    res = Parallel(n_jobs=5)(delayed(subjob_func)(fund) for fund in pool)
    table = {pool[i]:res[i] for i in range(len(pool))}
    pos = {}
    for fund in pool:
        temp = table[fund].reset_index(drop=True).dropna(subset=['DATE'])
        temp['FUNDCODE'] = fund
        pos[fund] = temp
    score = pd.concat([pos[fund] for fund in pool],join='outer')
    score = score.round(5).replace(0,np.nan)
    score.replace(1,np.nan,inplace=True)
    score.replace(2,np.nan,inplace=True)
    score = score.dropna(subset=['SEC','HIS','ALL'],how='all').set_index('FUNDCODE')
    score.to_csv('/home/qiantianyang/work/' + 'model_score.csv')

    sec.pop('YM')
    his.pop('YM')
    al.pop('YM')
    sec = sec.set_index(['FUNDCODE','DATE']).stack().reset_index()
    sec.rename(columns={'level_2':'PART',0:'WEIGHT'},inplace=True)
    sec = sec.loc[~(sec==0).any(axis=1)]
    his = his.set_index(['FUNDCODE','DATE']).stack().reset_index()
    his.rename(columns={'level_2':'PART',0:'WEIGHT'},inplace=True)
    his = his.loc[~(his==0).any(axis=1)]
    al = al.set_index(['FUNDCODE','DATE']).stack().reset_index()
    al.rename(columns={'level_2':'PART',0:'WEIGHT'},inplace=True)
    al = al.loc[~(al==0).any(axis=1)]
    sec['MODEL'] = '板块回归'
    al['MODEL'] = '全行业回归'
    his['MODEL'] = '历史行业回归'
    aio = pd.concat([sec,his,al])
    aio.to_csv('/home/qiantianyang/work/' + 'three_models_estimations.csv')

