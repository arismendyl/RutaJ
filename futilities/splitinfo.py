
def listToSplit(info_table,overFlow):
    overflow = info_table[overFlow]
    toSplit = overflow.index.tolist()
    return toSplit