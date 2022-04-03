import datetime
import logging
import os
import traceback

import pandas as pd
import pymysql

logger = logging.getLogger(__name__)

class DBConn:
    def __init__(self, *, 
            host=None, port=None, user=None, passwd=None, 
            sid=None, database=None, dbtype=None):
        os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.sid = sid
        self.database = database
        self.dbtype = dbtype
        self.connection = None

    def connect(self):
        if self.dbtype == 'ORACLE':
            if self.port is None:
                self.port = '1521'
            conn_addr = '{}/{}@{}:{}/{}'.format(
                self.user, self.passwd, self.host, self.port, self.sid)
            self.connection = cx_Oracle.connect(conn_addr)
        elif self.dbtype == 'MYSQL':
            if self.port is None:
                self.port = '3306'
            self.connection = pymysql.connect(
                host=self.host, user=self.user, password=self.passwd, database=self.database)
        else:
            raise NotImplementedError('Database type %s is not implemented!' % self.dbtype)

    def read(self, sql: str, close=True) -> pd.DataFrame:
        if self.connection is None:
            self.connect()
        data = pd.read_sql(sql, self.connection)
        if close:
            self.close()
        return data

    def close(self) -> None:
        if self.connection is None:
            print('No connection exists!')
            return
        self.connection.close()
        self.connection = None

class DBOpr(DBConn):
    def insert_dataframe(self, dat: pd.DataFrame, tbl_name: str, method: str='insert'):
        if self.connection is None:
            self.connect()
        cursor = self.connection.cursor()

        rt = len(dat)
        dat = dat.where(pd.notnull(dat), None)
        #处理时间
        dt_col_ns = dat.select_dtypes(include=['datetime']).columns.to_list()
        dat[dt_col_ns] = dat[dt_col_ns].astype(str)
        #添加操作时间和备注字段
        dat['operate_time'] = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')
        dat['operate_mode'] = 1
        col_names = '({})'.format(','.join(dat.columns))
        args = [tuple(x) for x in list(dat.values)]
        insert_sql = 'INSERT INTO {} {} VALUES ({})'.format(
            tbl_name, col_names, ','.join(['%s']*dat.shape[1]))
        if method == 'replace':
            insert_sql = 'REPLACE INTO {} {} VALUES ({})'.format(
                tbl_name, col_names, ','.join(['%s']*dat.shape[1]))
        elif method == 'ignore':
            insert_sql = 'INSERT IGNORE INTO {} {} VALUES ({})'.format(
                tbl_name, col_names, ','.join(['%s']*dat.shape[1]))

        try:
            cursor.executemany(insert_sql, args)
            self.connection.commit()
            logger.info('insert dat to {} succeed.'.format(tbl_name))
        except:
            traceback.print_exc()
            self.connection.rollback()
            logger.info('insert dat to {} failed.'.format(tbl_name))
            rt = -1
        cursor.close()
        self.close()
        return rt

    def drop_tbl(self, tbl_name: str):
        if self.connection is None:
            self.connect()
        db = self.connection
        cursor = db.cursor()
        rt = 0
        try:
            cursor.execute("DROP TABLE IF EXISTS {}".format(tbl_name))
            db.commit()
            logger.info('drop {} succeed.'.format(tbl_name))
        except:
            db.rollback()
            logger.info('drop {} failed.'.format(tbl_name))
            rt=-1
        cursor.close()
        self.close()
        return rt

    def clear_tbl(self, tbl_name: str):
        if self.connection is None:
            self.connect()
        db = self.connection
        cursor = db.cursor()
        rt = 0
        try:
            cursor.execute("TRUNCATE TABLE {}".format(tbl_name))
            db.commit()
            logger.info('clear {} succeed.'.format(tbl_name))
        except:
            db.rollback()
            logger.info('clear {} failed.'.format(tbl_name))
            rt = -1
        cursor.close()
        self.close()
        return rt

    def create_tbl(self, crt_sql: str=''):
        if self.connection is None:
            self.connect()
        db = self.connection
        cursor = db.cursor()
        rt = 0

        tbl_name = crt_sql.split()[2]
        try:
            cursor.execute(crt_sql)
            db.commit()
            logger.info('create {} succeed.'.format(tbl_name))
        except:
            db.rollback()
            logger.info('create {} failed.'.format(tbl_name))
            rt = -1
        cursor.close()
        self.close()
        return rt

    def delete_record(self, sql: str, tbl_name: str=''):
        if self.connection is None:
            self.connect()
        db = self.connection
        cursor = db.cursor()
        rt = 0
        try:
            rt=cursor.execute(sql)
            db.commit()
            logger.info('del {} succeed.'.format(tbl_name))
        except:
            db.rollback()
            logger.info('del {} failed.'.format(tbl_name))
            rt=-1
        cursor.close()
        self.close()
        return rt

class ConnWind(DBConn):
    def __init__(self) -> None:
        super().__init__()
        self.host = '10.200.100.198'
        self.port = '1521'
        self.user = 'wind_readonly'
        self.passwd = 'wind_readonly'
        self.sid = 'orcl'
        self.dbtype = 'ORACLE'

class ConnWind_mysql(DBConn):
    def __init__(self) -> None:
        super().__init__()
        self.host = '10.200.100.199'
        self.port = '3306'
        self.user = 'wind_readonly'
        self.passwd = 'wind_readonly'
        self.database = 'wind'
        self.dbtype = 'MYSQL'

class ConnJY(DBConn):
    def __init__(self) -> None:
        super().__init__()
        self.host = '10.200.100.198'
        self.port = '1521'
        self.user = 'jydb_readonly'
        self.passwd = 'jydb_readonly'
        self.sid = 'orcl'
        self.dbtype = 'ORACLE'

class ConnJY_mysql(DBConn):
    def __init__(self) -> None:
        super().__init__()
        self.host = '10.200.100.199'
        self.port = '3306'
        self.user = 'jydb_readonly'
        self.passwd = 'jydb_readonly'
        self.database = 'jydb'
        self.dbtype = 'MYSQL'

class ConnDZH(DBConn):
    def __init__(self) -> None:
        super().__init__()
        self.host = '47.92.52.147'
        self.port = '29782'
        self.user = 'dzh_shuke_bshao'
        self.passwd = 'R1B2CUtxjx1Uyn87HTf'
        self.database = 'fcdb'
        self.dbtype = 'MYSQL'

class ConnZYYX(DBConn):
    def __init__(self) -> None:
        super().__init__()
        self.host = '10.200.100.199'
        self.port = '3306'
        self.user = 'utsdaemon_readonly'
        self.passwd = 'utsdaemon_readonly'
        self.database = 'utsdaemon_data'
        self.dbtype = 'MYSQL'

class ConnTL(DBConn):
    def __init__(self) -> None:
        super().__init__()
        self.host = '10.200.100.199'
        self.port = '3306'
        self.user = 'hermes_readonly'
        self.passwd = 'hermes_readonly'
        self.database = 'hermes'
        self.dbtype = 'MYSQL'

class OprFOF(DBOpr):
    def __init__(self) -> None:
        super().__init__()
        self.host = '10.200.100.206'
        self.port = '3306'
        self.user = 'ods_fof'
        self.passwd = 'fofods98Hts'
        self.database = 'ods_fof'
        self.dbtype = 'MYSQL'

class ConnCCXD(DBConn):
    def __init__(self) -> None:
        super().__init__()
        self.host = '10.200.100.206'
        self.port = '3306'
        self.user = 'ods_readonly'
        self.passwd = 'ods_readonly'
        self.dbtype = 'MYSQL'
