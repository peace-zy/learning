
# -*- coding: UTF-8 -*-
"""
本文件实现了log函数。
Date:    2020/03/03 13:06:46
"""

import os
import logging
import logging.handlers

class Logger(object):
    level_dict = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self, filename, log_path="log", level="info", when="D", backup=7,
             fmt="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
             datefmt="%m-%d %H:%M:%S"):
        """
         init_log - initialize log module

         Args:
         log_path - Log file path prefix.
         Log data will go to two files: log_path.log and log_path.log.wf
         Any non-exist parent directories will be created automatically
         level - msg above the level will be displayed
         DEBUG < INFO < WARNING < ERROR < CRITICAL
         the default value is logging.INFO
         when - how to split the log file by time interval
         'S' : Seconds
         'M' : Minutes
         'H' : Hours
         'D' : Days
         'W' : Week day
         'MIDNIGHT' 每天凌晨
         default value: 'D'
         fmt - format of the log
         default format:
         %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
         INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
         backup - how many backup file to keep
         default value: 7

         Raises:
         OSError: fail to create log directories
         IOError: fail to open log file
        """
        self.level = self.level_dict.get(level) 
        formatter = logging.Formatter(fmt, datefmt)
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level)

        log_file = os.path.join(log_path, filename)
        dir_name = os.path.dirname(log_file)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        handler = logging.handlers.TimedRotatingFileHandler(filename=log_file + ".log",
                                                            interval=1,
                                                            when=when,
                                                            backupCount=backup,
                                                            encoding="utf-8")
        handler.setLevel(self.level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        handler = logging.handlers.TimedRotatingFileHandler(filename=log_file + ".log.wf",
                                                            interval=1,
                                                            when=when,
                                                            backupCount=backup,
                                                            encoding="utf-8")
        handler.setLevel(logging.WARNING)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        handler = logging.StreamHandler()#往控制台输出
        handler.setFormatter(formatter) #设置控制台上显示的格式
        self.logger.addHandler(handler)

#logger = Logger(__file__, level="info")
