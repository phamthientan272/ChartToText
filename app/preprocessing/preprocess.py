import re
import pandas as pd
from preprocessing.tokenizer import word_tokenize

def getChartType(x):
    if x.lower() == 'year':
        return 'line_chart'
    else:
        return 'bar_chart'

def cleanAxisLabel(label):
    cleanLabel = re.sub('\s', '_', label)
    cleanLabel = cleanLabel.replace('%', '').replace('*', '')
    return cleanLabel

def cleanAxisValue(value):
    #print(value)
    if value == '-' or value == 'nan':
        return '0'
    cleanValue = re.sub('\s', '_', value)
    cleanValue = cleanValue.replace('|', '').replace(',', '').replace('%', '').replace('*', '')
    return cleanValue

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def openMultiColumnData(data_file):
    df = pd.read_csv(data_file)
    cols = df.columns
    size = df.shape[0]
    chartType = getChartType(cols[0])
    return df, cols, size, chartType

def refactor_titles(title: str):
        clean_title = word_tokenize(title)
        clean_title = ' '.join(clean_title).replace('*', '')
        return clean_title


class Preprocessor:
    def __init__(self, raw_data_file, title):
        self.__raw_data_file = raw_data_file
        self.__title = title
        
    def preprocess_data(self):
        dataArr = []
        titleArr = []
        
        df, cols, size, chartType = openMultiColumnData(self.__raw_data_file)
        cleanCols = [cleanAxisLabel(axis) for axis in cols]
        dataLine = ''
        colData = []
        for col in df:
            vals = df[col].values
            cleanVals = [cleanAxisValue(str(value)) for value in vals]
            colData.append(cleanVals)
        # iterate through each table row
        for m in range(0, size):
            axisTypes = []
            for axis, n in zip(cols, range(cols.size)):
                if is_number(axis[0]):
                    axisTypes.append('numerical')
                else:
                    axisTypes.append('categorical')
                value = str(df.at[m, axis])
                cleanValue = cleanAxisValue(value)
                #rowData.append(cleanValue)
                record = f"{cleanCols[n]}|{cleanValue}|{n}|{chartType}"
                dataLine += f'{record} '
        
        title = refactor_titles(self.__title)
        
        
        dataArr.append(dataLine)
        titleArr.append(title)
        
        return dataArr, titleArr
        

