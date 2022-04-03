import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from typing import List
from ccxd_dbtool import conn

import util_and_api.trading_date.trading_date as tdate

DATE_FORMAT = "%Y%m%d"
DASHED_DATE_FORMAT = "%Y-%m-%d"
DATAPATH = Path('/data/public_transfer/ysun/fund_test/')


def convert_to_date(file_date):
    if isinstance(file_date, date):
        return file_date
    elif isinstance(file_date, str):
        if len(file_date) == 8:
            return datetime.strptime(file_date, DATE_FORMAT).date()
        elif len(file_date) == 10:
            return datetime.strptime(file_date, DASHED_DATE_FORMAT).date()
        else:
            raise Exception(f"invalid format of date {file_date}")
    else:
        raise Exception(f"invalid format of date {file_date}")


def date_to_date_sub_path(file_date):
    _date = convert_to_date(file_date)
    return f"{_date.year}/{_date.month}/{_date.day}"


def walk_dir(rtpath: Path) -> List[Path]:
    years = [x for x in rtpath.iterdir() if x.parts[-1].isdigit() and x.is_dir()]
    months = []
    for x in years:
        months += list(x.iterdir())
    months = [x for x in months if x.parts[-1].isdigit() and x.is_dir()]
    days = []
    for x in months:
        days += list(x.iterdir())
    days = [x for x in days if x.parts[-1].isdigit() and x.is_dir()]
    return days


def collect(fn, rootpath: Path=DATAPATH, start: str=None, end: str=None, **kwargs):
    datedirs = walk_dir(rootpath)
    datedirs.sort(key=lambda x: date(int(x.parts[-3]), int(x.parts[-2]), int(x.parts[-1])))
    dirdates = [date(int(x.parts[-3]), int(x.parts[-2]), int(x.parts[-1])) for x in datedirs]
    datedirs = np.array(datedirs)
    dirdates = np.array(dirdates)
    mark = np.ones(len(datedirs), dtype=bool)
    if start is not None:
        start = convert_to_date(start)
        mark = mark & (dirdates >= start)
    if end is not None:
        end = convert_to_date(end)
        mark = mark & (dirdates <= end)
    datedirs = datedirs[mark]
    
    res = []
    for d in datedirs:
        df = pd.read_csv(d / fn, **kwargs)
        res.append(df)
    res = pd.concat(res, ignore_index=True)
    return res

def write_datedir(data: pd.DataFrame, datecol: str, fn: str, write_path: Path):
    grouped = data.groupby(datecol)
    for _date, group in grouped:
        _dt = convert_to_date(_date)
        fpath = write_path / date_to_date_sub_path(_dt)
        if not fpath.is_dir():
            fpath.mkdir(parents=True)
        group.to_csv(fpath / fn, index=False)

def check_lastupdate(fn, datapath):
    datedirs = walk_dir(datapath)
    datedirs.sort(key=lambda x: date(int(x.parts[-3]), int(x.parts[-2]), int(x.parts[-1])))
    datefns = [datedir / fn for datedir in datedirs]
    updated = [x for x in datefns if x.is_file()]
    if len(updated) == 0:
        return None
    else:
        parts = updated[-1].parts
        update_date = date(int(parts[-4]), int(parts[-3]), int(parts[-2]))
        return update_date.strftime('%Y-%m-%d')
    