from CandleNet import Logger


def dummy_log(log_type, origin, caller, msg):
    Logger().log(log_type, origin, caller, msg)
