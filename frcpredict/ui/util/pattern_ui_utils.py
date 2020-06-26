from frcpredict.model import PatternType


def getPatternTypeName(patternType: PatternType) -> str:
    if patternType == PatternType.gaussian:
        return "Gaussian"
    elif patternType == PatternType.doughnut:
        return "Doughnut"
    elif patternType == PatternType.airy:
        return "Airy"
    elif patternType == PatternType.digital_pinhole:
        return "Digital pinhole"
    elif patternType == PatternType.physical_pinhole:
        return "Physical pinhole"
    else:
        return patternType.name
