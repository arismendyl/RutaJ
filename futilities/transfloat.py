import pandas as pd

def commatoperiod(column):
    trans = column.replace(',','.',regex=True)
    return trans

def strtofloat(column):
    column = pd.to_numeric(column, downcast="float")
    return column