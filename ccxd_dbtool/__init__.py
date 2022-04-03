
import pkgutil
import logging.config
import yaml

log_cfg = yaml.safe_load(pkgutil.get_data(__package__, 'config/log_conf.yaml'))
logging.config.dictConfig(log_cfg)