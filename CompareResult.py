"""
Usage: 
    CompareResult.py <year> (--SP500 | --SP1500 | --all)

"""
import sys
from os.path import join
#import sklearn.cluster as cluster
import numpy as np
import math
import json
from docopt import docopt
import pandas as pd
import pyarrow.parquet as pq

if __name__ == "__main__":

    # Read search fraction from EDGAR log
    print("Reading Search fraction")
    opt = docopt(__doc__)
    year = opt["<year>"]
    if opt["--SP500"]:
        fileName = "./EDGARLog/Result/dict"+year+"SP500.json"
    if opt["--SP1500"]:
        fileName = "./EDGARLog/Result/dict"+year+"SP1500.json"
    if opt["--all"]:
        fileName = "./EDGARLog/Result/dict"+year+"all.json"

    with open(fileName, "r") as f:
        searchFractionDict = json.load(f)

    #print(searchFractionDict["1163165"][3][1])

    # Read in Financial ratios
    print("Reading Financial Ratios")
    table1 = pq.read_table('ChengLun.parquet-2.gz')
    df1 = table1.to_pandas()
    df1 = df1.rename(columns = {"CUSIP {t, historical}":"CUSIP"})
    df1["CUSIP"] = df1["CUSIP"].str.decode("utf-8")
    
    df1.drop(df1[df1["Date EOM {t}"].dt.month != 12].index, inplace=True)
    
    df1["Year"] = df1["Date EOM {t}"].dt.year
    df1.drop(df1[df1["Year"] != int(year)].index, inplace=True)
    
    #df1.to_csv(path_or_buf='./df1.csv')
    #print(df1)

    # Read CIK_CUSIP table and get the dictionary
    df2 = pd.read_csv(filepath_or_buffer = "./CIK_CUSIP.csv", usecols = [0, 1, 3])
    #print(df2)

    merged = pd.merge(df1, df2, on=["CUSIP", "Year"])
    merged["PB"] =merged["PRC {t-1}"] / merged["AT {t, y-1}"] * merged["SHROUT {t-1}"]
    #merged["PE"] =merged["PRC {t-1}"] / merged["NI {t, y-1}"] * merged["SHROUT {t-1}"]
    merged["OIS"] =merged["OIADP {t, y-1}"] / merged["SALE {t, y-1}"] 
    #merged["OIAT"] =merged["OIADP {t, y-1}"] / merged["AT {t, y-1}"] 
    #merged["ROE"] = merged["ROA {t, y-1}"] * merged["AT {t, y-1}"] / merged["SEQ {t, y-1}"]
    #merged["NIS"] = merged["NI {t, y-1}"] / merged["SALE {t, y-1}"]
    financialDict = merged.set_index("CIK").to_dict()
    #usedRatio = "RNOA {t, y-1}"
    usedRatio = "PB"
    #usedRatio = "ROE"
    #usedRatio = "OIAT"
    #merged.to_csv(path_or_buf='./merged.csv')
    #print(financialDict["PRC {t-1}"][785787])
    
    # Contains entries of S&P500 & S&P1500 at the year
    sp = pd.read_csv("./EDGARLog/sp500_constituents.csv", delimiter=',', header=0, usecols=[4,5,6,7,12,13,15,16], dtype={'co_cik':'Int64'}, parse_dates = ['from', 'thru'], date_parser = pd.to_datetime)
    if opt["--SP500"]:
        sp.drop(sp[(sp['conm']!='S&P 500 Comp-Ltd')].index, inplace=True)
    if opt["--SP1500"]:
        sp.drop(sp[(sp['conm']!='S&P 1500 Super Composite') & (sp['conm']!='S&P 500 Comp-Ltd')].index, inplace=True)
    #if opt["--all"]:
    spDate = sp.drop(sp[(sp['from'] > pd.Timestamp(year + "0101")) | (sp['thru'] < pd.Timestamp(year + "1231"))].index)
    #spDate.to_csv(path_or_buf='./spDate.csv')
     
    # Read similarity matrix from 10-K (similarity in word usage)
    print("Reading similarity matrix")
    
    #inpath = sys.argv[1]
    inpath = "./EDGAR_Parsing/edgar-10k-mda/form10k" + str(int(year)%100).zfill(2) + "_item1_dict/"
    ciks = [] # ciks in Similarity Matrix

    with open(join(inpath, "similarityMatrix.txt"), 'r') as f:
        ciks = f.readline().split(",")[:-1]
        similarityMatrix = [[0 for x in range(len(ciks))] for y in range(len(ciks))]
        i = 0
        for line in f:
            similarityMatrix[i] = [float(k) for k in line.split(",")[:-1]]
            i += 1
   
    ciksOK = len(ciks)*[False]
    for i in range(len(ciks)):
        if int(ciks[i]) in spDate["co_cik"].values:
            ciksOK[i] = True
    
    similarityList = []
    # Find the suitable threshold from similarity Matrix 
    print("Find the suitable threshold from similarity Matrix")
    for i in range(len(ciks)):
        if not ciksOK[i]:
            continue
        #similarityMatrix[i][i] = 1
        for j in range(i):
            if not ciksOK[j]:
                continue
            #similarityMatrix[i][j] = similarityMatrix[j][i]
            similarityList.append(similarityMatrix[j][i])

    similarityList.sort(reverse=True)
    thresholdForm = similarityList[int(0.01*len(similarityList))] # top 1%

    # Get CIK to SIC lookup table
    print("Getting CIK to SIC lookup table")

    
    cik2sic = dict() # CIK to SIC
    cik2conm = dict() # CIK to Company Name
    sicSet = dict() # SIC to set of CIKs
    
    for row in spDate.itertuples():        
        co_cik = str(getattr(row, 'co_cik'))
        co_conm = str(getattr(row, 'co_conm'))
        co_sic = str(getattr(row, 'co_sic'))[0:2] # sic2
        if co_cik == 'nan':
            #print(co_cik + " " + co_conm + " " + co_sic)
            continue
        cik2sic[co_cik] = co_sic
        cik2conm[co_cik] = co_conm
        if co_sic in sicSet:
            sicSet[co_sic].add(co_cik)
        else:
            sicSet[co_sic] = set()
            sicSet[co_sic].add(co_cik)

    # Iterating through CIKs in search fraction input file
    print("Iterating through CIKs in search fraction input file")
    logForm = 0
    formLog = 0
    same = 0

    logSic = 0
    sicLog = 0
    same2 = 0

    formSic = 0
    sicForm = 0
    same3 = 0

    ratioSBP = []
    ratioTNIC = []
    ratioSIC = []

    for cik in searchFractionDict:
        if cik not in ciks:
            continue

        peerLog = [] # peers of cik from search fraction (SBP)
        tmpRatio = 0
        count = 0
        i = 0
        while i < len(searchFractionDict[cik]) and i < 10: #searchFractionDict[cik][i][1] > thresholdLog:
            peerLog.append(str(searchFractionDict[cik][i][0]))
            if int(searchFractionDict[cik][i][0]) in financialDict[usedRatio] and not math.isnan(financialDict[usedRatio][int(searchFractionDict[cik][i][0])]):
                tmpRatio += financialDict[usedRatio][int(searchFractionDict[cik][i][0])]
                count += 1
            i += 1
        if count != 0:
            ratioSBP.append(tmpRatio/count)

        peerForm = [] # peers of cik from Form 10K (TNIC)
        index = ciks.index(cik)
        tmpRatio = 0
        count = 0
        for i in range(len(ciks)):
            if not ciksOK[i]:
                continue
            if i != index and similarityMatrix[index][i] > thresholdForm:
                peerForm.append((ciks[i], similarityMatrix[index][i]))
                if int(ciks[i]) in financialDict[usedRatio] and not math.isnan(financialDict[usedRatio][int(ciks[i])]):
                    tmpRatio += financialDict[usedRatio][int(ciks[i])]
                    count += 1
        peerForm.sort(key=lambda tup: tup[1])
        peerForm = [x[0] for x in peerForm]
        if count != 0:
            ratioTNIC.append(tmpRatio/count)

        tmpRatio = 0
        count = 0
        if cik in cik2sic and cik2sic[cik] in sicSet:
            for i in sicSet[cik2sic[cik]]:
                #print("SIC:" + str(cik2sic[cik]) + " cik:" + str(i))
                if int(i) in financialDict[usedRatio] and not math.isnan(financialDict[usedRatio][int(i)]):
                    tmpRatio += financialDict[usedRatio][int(i)]
                    count += 1
        
        if count != 0:
            ratioSIC.append(tmpRatio/count)
        
        '''
        if cik == "320193":
            print("TNIC Peers ===========")
            for i in range(len(peerForm)):
                if peerForm[i] in cik2conm:
                    print(peerForm[i] + " " + cik2conm[peerForm[i]])

            print("SBP Peers ===========")
            for i in range(len(peerLog)):
                print(peerLog[i] + " " + cik2conm[peerLog[i]])
            
            print("SIC 2 ========")
            for i in sicSet[cik2sic[cik]]:
                print(i + " " + cik2conm[i])
            #for i in range(len(sicSet[cik2sic[cik]])):
            #    print(sicSet[cik2sic[cik]][i] + " " + cik2conm[sicSet[cik2sic[cik]][i]])'''

        logForm += len(list(set(peerLog) - set(peerForm)))
        formLog += len(list(set(peerForm) - set(peerLog)))
        same += len(list(set(peerForm).intersection(set(peerLog))))
       
        if cik in cik2sic and cik2sic[cik] in sicSet:
            logSic += len(list(set(peerLog) - sicSet[cik2sic[cik]]))
            sicLog += len(list(sicSet[cik2sic[cik]] - set(peerLog)))
            same2 += len(list(sicSet[cik2sic[cik]].intersection(set(peerLog))))
            
            formSic += len(list(set(peerForm) - sicSet[cik2sic[cik]]))
            sicForm += len(list(sicSet[cik2sic[cik]] - set(peerForm)))
            same3 += len(list(sicSet[cik2sic[cik]].intersection(set(peerForm))))


    #for ratio in ratioSBP:
    #    print(str(ratio), end = ' ')
    print(usedRatio + " Variance of SBP: " + str(np.var(ratioSBP)))
    print(usedRatio + " Variance of TNIC: " + str(np.var(ratioTNIC)))
    print(usedRatio + " Variance of SIC: " + str(np.var(ratioSIC)))

    ''' 
    print("Fraction of SBP intersect with TNIC: " + str(same/(logForm+same)))
    print("Fraction of TNIC intersect with SBP: " + str(same/(formLog+same)))
    print("Fraction of SBP intersect with SIC: " + str(same2/(logSic+same2)))
    print("Fraction of SIC intersect with SBP: " + str(same2/(sicLog+same2)))
    print("Fraction of TNIC intersect with SIC: " + str(same3/(formSic+same3)))
    print("Fraction of SIC intersect with TNIC: " + str(same3/(sicForm+same3)))
    '''
