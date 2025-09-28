from enum import Enum


class LogType(Enum):
    EVENT = "EVENT"
    WARNING = "WARNING"
    ERROR = "ERROR"
    STATUS = "STATUS"


class OriginType(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
