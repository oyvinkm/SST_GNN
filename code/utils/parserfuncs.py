from loguru import logger

def none_or_str(value):
    if value.lower() == "none":
        return None
    return value

def none_or_int(value):
    if value == "None":
        return None
    return int(value)

def none_or_float(value):
    if value == "None":
        return None
    return float(value)
    
def t_or_f(value):
    ua = str(value).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       logger.CRITICAL("boolean argument incorrect")