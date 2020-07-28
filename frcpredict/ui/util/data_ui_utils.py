from enum import Enum


def getEnumEntryName(enumEntry: Enum) -> str:
    return enumEntry.name.capitalize().replace("_", " ")
