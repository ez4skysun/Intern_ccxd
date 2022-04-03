# 数据字典

| 数据库     | 用户名      | 密码        | 地址                                         |
| :--------- | :---------- | :---------- | :------------------------------------------- |
| 聚源       | zcxszsjk01  | gildata@123 | https://dd.gildata.com/                      |
| 大智慧财汇 | ccxi        | ccxi123     | https://datadict.finchina.com/               |
| 朝阳永续   | E000090602  | wXS5Ju      | http://gogoaldata.go-goal.cn/html/Login.html |
| 通联       | 18817818993 | CCXd1234    | https://app.datayes.com                      |
| 万得       | 15810733147 | CCXDwind123 | http://wds.wind.com.cn/                      |

# FOF数据库表创建示例
```python
from conn import OprFOF

create_fund_barra_exposure_sql = '''
create table fund_barra_exposure
(
    id BIGINT(19) NOT NULL AUTO_INCREMENT COMMENT '自增id',
    WINDCODE varchar(50) NOT NULL COMMENT '基金代码',
    TRADE_DT varchar(50) NOT NULL COMMENT '交易日',
    BETA float COMMENT 'Beta',
    MOM float COMMENT '动量',
    SIZE float COMMENT '对数市值',
    EARN float COMMENT '收益率',
    RESVOL float COMMENT '剩余波动率',
    GROWTH float COMMENT '成长',
    BP float COMMENT '账面市值比',
    LEV float COMMENT '杠杆',
    LIQ float COMMENT '流动性',
    NLSIZE float COMMENT '非线性市值',
    operate_time datetime,
    operate_mode int,
    PRIMARY KEY (`id`) USING BTREE,
    index WINDCODE(WINDCODE),
    index TRADE_DT(TRADE_DT)
)
    COMMENT='基金每日Barra CNE5模型下的因子暴露'
    COLLATE='utf8mb4_0900_ai_ci'
    ENGINE=InnoDB
    AUTO_INCREMENT=1
'''

fofdb = OprFOF()
fofdb.create_tbl(create_fund_barra_exposure_sql)
```