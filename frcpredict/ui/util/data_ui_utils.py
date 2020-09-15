from enum import Enum


def snakeCaseToName(snakeCaseString: str) -> str:
    name = snakeCaseString[:1].upper() + snakeCaseString[1:]  # Convert first character to uppercase
    name = name.replace("_", " ")
    return name


def getEnumEntryName(enumEntry: Enum) -> str:
    return snakeCaseToName(enumEntry.name)
