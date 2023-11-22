# file to handle different form of errors and exceptions


class ValueOverflowingError(Exception):
    def __init__(self, msg:str) -> None:
        super().__init__(msg)
