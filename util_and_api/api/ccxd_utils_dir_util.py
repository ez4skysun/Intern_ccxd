import os
from datetime import datetime, date


# /data/data_platform/{market}/{asset_class}/
#                                        .../raw/{data_source}/{domain}/
#                                                                   .../full/{static_files}
#                                                                   .../{year}/{month}/day/{daily_files}
#                                        .../cooked/{domain}/
#                                                        .../full/{static_files}
#                                                        .../{year}/{month}/day/{daily_files}
# /data/data_platform/ccxd/{env}/{domain}/{domain_specific_paths?}

DP_ROOT_DIR = "/data-platform"
LOG_DIR_TEMPLATE = os.path.join(DP_ROOT_DIR, "log/{market}/{asset_class}")
VENDOR_DIR_TEMPLATE = os.path.join(DP_ROOT_DIR, "{market}/{asset_class}")
VENDOR_RAW_DIR_TEMPLATE = os.path.join(VENDOR_DIR_TEMPLATE, "raw/{data_source}/{domain}")
VENDOR_COOKED_DIR_TEMPLATE = os.path.join(VENDOR_DIR_TEMPLATE, "cooked/{domain}")
CCXD_FILE_TEMPLATE = os.path.join(DP_ROOT_DIR, "ccxd/{env}/{domain}")

DEFAULT_MARKET = "CHN"
DATE_FORMAT = "%Y%m%d"
DASHED_DATE_FORMAT = "%Y-%m-%d"
ACCEPTED_CCXD_ENVS = ("prod", "dev")


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


class DirLocator:
    def __init__(self, market, asset_class):
        self.__market = market
        self.__asset_class = asset_class

    def log_dir(self):
        path_template = LOG_DIR_TEMPLATE
        return path_template.format(market=self.__market, asset_class=self.__asset_class)
        
    def vendor_raw_full_file_name(self, data_source, domain, file_name):
        path_template = os.path.join(VENDOR_RAW_DIR_TEMPLATE, "full/{file_name}")
        return path_template.format(market=self.__market, asset_class=self.__asset_class, data_source=data_source, domain=domain, file_name=file_name)

    def vendor_raw_daily_file_name(self, data_source, domain, file_date, file_name):
        path_template = os.path.join(VENDOR_RAW_DIR_TEMPLATE, "{year}/{month}/{day}/{file_name}")
        _date = convert_to_date(file_date)
        return path_template.format(market=self.__market, asset_class=self.__asset_class, data_source=data_source, domain=domain,
                                    year=_date.year, month=_date.month, day=_date.day, file_name=file_name)
        pass

    def cooked_full_file_name(self, domain, file_name):
        path_template = os.path.join(VENDOR_COOKED_DIR_TEMPLATE, "full/{file_name}")
        return path_template.format(market=self.__market, asset_class=self.__asset_class, domain=domain, file_name=file_name)

    def cooked_daily_file_name(self, domain, file_date, file_name):
        path_template = os.path.join(VENDOR_COOKED_DIR_TEMPLATE, "{year}/{month}/{day}/{file_name}")
        _date = convert_to_date(file_date)
        return path_template.format(market=self.__market, asset_class=self.__asset_class, domain=domain,
                                    year=_date.year, month=_date.month, day=_date.day, file_name=file_name)

    def ccxd_file_name(self, env, domain, sub_path):
        if env not in ACCEPTED_CCXD_ENVS:
            raise Exception(f"invalid env {env}. Should have env in {ACCEPTED_CCXD_ENVS}")
        path_template = os.path.join(CCXD_FILE_TEMPLATE, "{sub_path}")
        return path_template.format(market=self.__market, env=env, domain=domain, sub_path=sub_path)


def main():
    asset_class, data_source, domain, file_name = "bond", "WIND", "Basic", "test.csv"
    file_date = '2021-01-01'
    CHNDirLocator = DirLocator('CHN', asset_class)
    print(CHNDirLocator.vendor_raw_full_file_name(data_source=data_source, domain=domain, file_name=file_name))
    print(CHNDirLocator.vendor_raw_daily_file_name(data_source=data_source, domain=domain, file_date=file_date, file_name=file_name))
    print(CHNDirLocator.cooked_full_file_name(domain=domain, file_name=file_name))
    print(CHNDirLocator.cooked_daily_file_name(domain=domain, file_date=file_date, file_name=file_name))
    print(CHNDirLocator.ccxd_file_name(env="prod", domain="valuation", sub_path="test/subpath/test.txt"))
    print(CHNDirLocator.log_dir())
    try:
        print(CHNDirLocator.ccxd_file_name(env="test", domain=domain, sub_path="test/subpath/test.txt"))
    except:
        print("error")
    print(convert_to_date("20200304"))
    print(convert_to_date("2020-03-04"))
    try:
        print(convert_to_date("2020304"))
    except:
        print("error")
    print(convert_to_date(date(2020, 3, 4)))
    print(date_to_date_sub_path("20200506"))

    pass


if __name__ == '__main__':
    main()
