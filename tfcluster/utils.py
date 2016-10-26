import logging

def setup_logger(logger):
    logger.setLevel(logging.DEBUG)

    FORMAT = '%(levelname).1s%(asctime)-11s.%(msecs)06d %(process)d %(filename)s:%(lineno)d] %(message)s'
    formatter = logging.Formatter(FORMAT, datefmt = '%m%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

