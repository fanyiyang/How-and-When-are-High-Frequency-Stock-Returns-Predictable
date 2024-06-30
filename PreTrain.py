import pandas as pd
from MatchingEngine import Engine
import time
import os
import numpy as np
from MatchingEngine import Side


def LoadData(stockname):
    """
    Load the data of the stock with the stockname
    """
    print(stockname,'LoadData')
    start=time.time()
    files = os.listdir("../../data/HFData/HS300_data/%s.XSHE/2020/" %stockname)
    files = sorted(files)
    filesnumber=files
    for i in range(len(files)):
        filesnumber[i]=[int(files[i][:2]),int(files[i][2:])]
    file_pathT='/data/HFData/HS300_data'
    stock="%s.XSHE" %stockname
    year=2020
    datadict={}
    datelist=[]
    for part in filesnumber:
        month=part[0]
        day=part[1]
        try:
            engine=Engine(stock=stock,year=year,month=month,day=day,file_path=file_pathT)
            engine.main_matching_process(execute_flag=True,execute_rule="trade_before",execute_level_num=1)
            execute_total_df = pd.DataFrame(engine.order_book.execute_total_list)
            date=pd.to_datetime('2020'+'-'+str(month)+'-'+str(day))
            datadict[date]=execute_total_df
            datelist.append(date)
        except:
            print(month,day,stock,'no data')
    fulldata=pd.concat(datadict)
    fulldata.to_csv("/data/work/yangsq/dataset_all/%sfulldata.csv" %stockname)
    use=time.time()-start
    print("usetime:%f" %use)

def PreTrain(stockname):
    """
    PreTrain the data of the stock with the stockname
    """
    print(stockname,'PreTrain')
    # global today
    start=time.time()
    fulldata=pd.read_csv("/data/work/yangsq/dataset_all/%sfulldata.csv" %stockname,parse_dates=True)
    fulldata=fulldata.rename(columns={'Unnamed: 0':'TradingDay','Unnamed: 1':'TradesNum'})
    fulldata=fulldata.set_index('TradingDay')
    fulldata.index=pd.to_datetime(fulldata.index)
    fulldata=fulldata.loc['2020-01-01':'2020-12-31']  #Change the time period to be calculated here

    amountdict={}
    for i in range(1,10):
        amountdata=pd.read_csv("/data/work/yangsq/数据/20200%s.csv" %i,encoding='utf-8',encoding_errors='ignore')
        amountdict[i]=amountdata.loc[amountdata.证券代码=='%s.SZ' %stockname].T.iloc[2:]
    for i in range(10,13):
        amountdata=pd.read_csv("/data/work/yangsq/数据/2020%s.csv" %i,encoding='utf-8',encoding_errors='ignore')
        amountdict[i]=amountdata.loc[amountdata.证券代码=='%s.SZ' %stockname].T.iloc[2:]
    amount2020=pd.concat(amountdict).droplevel([1])
    amount2020.index=pd.date_range('2020-01-01','2020-12-31',freq='D')
    dateindex=fulldata.index
    today=0

    ResponseVariable=['Return5s','Return30s','Return10trades','Return200trades','Return1000Volumes','Return20000Volumes']
    Predictorlabel=['Breadth','Immediacy','VolumeAll','VolumeAvg','VolumeMax','Lambda','LobImbalance','TxnImbalance','PastReturn','Turnover','AutoCov','QuotedSpread','EffectiveSpread']
    Predictorname=[]
    for i in range(9):
        for j in range(13):
            Predictorname.append(Predictorlabel[j]+str(i))
    time_cal=[(0,0.1),(0.1,0.2),(0.2,0.4),(0.4,0.8),(0.8,1.6),(1.6,3.2),(3.2,6.4),(6.4,12.8),(12.8,25.6)]
    time_tran=[(0,1),(1,2),(2,4),(4,8),(8,16),(16,32),(32,64),(64,128),(128,256)]
    time_vol=[(0,100),(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12800),(12800,25600)]

    # Function to calculate return
    def ResponseCalculator(data):
        data[ResponseVariable]=0
        data=data.reset_index().set_index('TradesNum')
        data.Return10trades=data.Price.rolling(10).mean()/data.Price-1
        data.Return200trades=data.Price.rolling(200).mean()/data.Price-1
        data=data.reset_index().set_index('Volumecum')
        data.Return1000Volumes=data.Price.rolling(1000).mean()/data.Price-1
        data.Return20000Volumes=data.Price.rolling(20000).mean()/data.Price-1
        data=data.reset_index().set_index('Time')
        data.index=pd.to_datetime(data.index,format='%H%M%S%f')
        data.Return5s=data.Price.rolling('5s').mean()/data.Price-1
        data.Return30s=data.Price.rolling('30s').mean()/data.Price-1
        return data

    # Function to calculate the predictor
    def PredictorCalculator(data,timetype,timescale):
        # global today
        if timetype=='calendar':
            window_left='%sms' %time_cal[timescale][0]*1000
            window_right='%sms' %time_cal[timescale][1]*1000
            time=data.Time
        if timetype=='transaction':
            window_left=time_tran[timescale][0]
            window_right=time_tran[timescale][1]
            time=pd.Series(np.arange(len(data)),index=data.index)
        if timetype=='volume':
            window_left=time_vol[timescale][0]
            window_right=time_vol[timescale][1]
            time=data.Volumecum
        data.index=time
        # Construct the predictors
        Breadth=data.Price.rolling(window_right,closed='left').count()-data.Price.rolling(window_left,closed='left').count()
        Immediacy=(time_cal[timescale][1]-time_cal[timescale][0])/Breadth
        VolumeAll=data.TradeQty.rolling(window_right,closed='left').sum()-data.TradeQty.rolling(window_left,closed='left').sum()
        VolumeAvg=VolumeAll/Breadth
        Plt=data.Price-data.Price.shift(1)
        Lambda=(Plt.rolling(window_right,closed='left').sum()-Plt.rolling(window_left,closed='left').sum())/VolumeAll
        IS=(data.OfferSize1-data.BidSize1)/(data.OfferSize1+data.BidSize1)
        LobImbalance=(IS.rolling(window_right,closed='left').sum()-IS.rolling(window_left,closed='left').sum())/Breadth
        Dir=pd.Series(np.where(data.Direction==Side.BUY,1.0,-1.0),index=data.index)
        TxnImbalance=((data.TradeQty.multiply(Dir)).rolling(window_right,closed='left').sum()-(data.TradeQty.multiply(Dir)).rolling(window_left,closed='left').sum())/VolumeAll
        PastReturn=1-(data.Price.rolling(window_right).sum()-data.Price.rolling(window_left).sum())/Breadth/data.Price
        Turnover=VolumeAll/amount2020.loc[dateindex[today]].iloc[0]
        today=today+1
        logPlt1=(data.Price/data.Price.shift(1)).apply(np.log)
        logPlt2=(data.Price.shift(1)/data.Price.shift(2)).apply(np.log)
        AutoCov=((logPlt1.multiply(logPlt2)).rolling(window_right,closed='left').sum()-(logPlt1.multiply(logPlt2)).rolling(window_left,closed='left').sum())/Breadth
        Spread=(data.OfferPX1-data.BidPX1)/(data.OfferPX1+data.BidPX1)
        QuotedSpread=(Spread.rolling(window_right,closed='left').sum()-Spread.rolling(window_left,closed='left').sum())/Breadth
        WSpread=logPlt1.multiply(Dir).multiply(data.TradeQty).multiply(data.Price)
        Weight=data.TradeQty.multiply(data.Price)
        EffectiveSpread=(WSpread.rolling(window_right,closed='left').sum()-WSpread.rolling(window_left,closed='left').sum())/(Weight.rolling(window_right,closed='left').sum()-Weight.rolling(window_left,closed='left').sum())
        dict={Predictorname[13*timescale+0]:Breadth,Predictorname[13*timescale+1]:Immediacy,Predictorname[13*timescale+2]:VolumeAll,Predictorname[13*timescale+3]:VolumeAvg,Predictorname[13*timescale+5]:Lambda,Predictorname[13*timescale+6]:LobImbalance,Predictorname[13*timescale+7]:TxnImbalance,Predictorname[13*timescale+8]:PastReturn,Predictorname[13*timescale+9]:Turnover,Predictorname[13*timescale+10]:AutoCov,Predictorname[13*timescale+11]:QuotedSpread,Predictorname[13*timescale+12]:EffectiveSpread}
        Predictor=pd.concat(dict,axis=1)
        return Predictor

    # Function of complete calculation
    def PreRegress(dataO,type='calendar'):
        """
        Calculate the predictors and response variables for the data
        """
        data=pd.DataFrame(dataO)
        data=data.dropna(how='any')
        data=data.sort_values('Time',ascending=True)
        data['Volumecum']=data['TradeQty'].cumsum()
        data=data.sort_values('Time',ascending=False)
        data=ResponseCalculator(data)
        data=data.sort_values('Time',ascending=True).reset_index()
        Predictordict={}
        if type=='calendar':
            for i in range(9):
                Predictordict[i]=PredictorCalculator(data,'calendar',i)
            x=pd.concat(Predictordict,axis=1)
            y=data[['Return5s','Return30s']]
        elif type=='transaction':
            for i in range(9):
                Predictordict[i]=PredictorCalculator(data,'transaction',i)
            x=pd.concat(Predictordict,axis=1)
            y=data[['Return10trades','Return200trades']].reset_index()
        elif type=='volume':
            for i in range(9):
                Predictordict[i]=PredictorCalculator(data,'volume',i)
            x=pd.concat(Predictordict,axis=1)
            y=data.reset_index().set_index('Volumecum')[['Return1000Volumes','Return20000Volumes']]
        trainset=x.join(y,how='left',sort=True).dropna(how='any',axis=0)
        return trainset

    trainset=fulldata.groupby('TradingDay').apply(PreRegress)
    print(trainset)

    end=time.time()
    use=end-start
    print("usetime:%f" %use)

    trainset.to_csv("/data/work/yangsq/trainset_all/%strainset.csv" %stockname)



files = os.listdir("../../data/HFData/HS300_data/")
files = sorted(files)
# We choose 12 representative stocks in HS300 to pretrain
stockname_list =['002352', '002044', '002142', '002304', '002027', '002032', '002120', '002129', '002311', '002271', '002157', '002064']

for stockname in stockname_list:
    print(stockname)
    try:
        LoadData(stockname)
        PreTrain(stockname)
    except:
        pass