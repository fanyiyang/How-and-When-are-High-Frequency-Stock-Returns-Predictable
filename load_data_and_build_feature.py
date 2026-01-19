import os
import time

import numpy as np
import pandas as pd
from MatchingEngine import Engine, Side


def load_raw_data(stockname):
    """Load raw tick data and store merged trades/quotes for a single stock."""
    print(stockname, "load_raw_data")
    start = time.time()
    files = os.listdir(f"../../data/HFData/HS300_data/{stockname}.XSHE/2020/")
    files = sorted(files)
    filesnumber = files
    for i in range(len(files)):
        filesnumber[i] = [int(files[i][:2]), int(files[i][2:])]
    file_path = "/data/HFData/HS300_data"
    stock = f"{stockname}.XSHE"
    year = 2020
    datadict = {}
    for month, day in filesnumber:
        try:
            engine = Engine(stock=stock, year=year, month=month, day=day, file_path=file_path)
            engine.main_matching_process(
                execute_flag=True, execute_rule="trade_before", execute_level_num=1
            )
            execute_total_df = pd.DataFrame(engine.order_book.execute_total_list)
            date = pd.to_datetime(f"2020-{month}-{day}")
            datadict[date] = execute_total_df
        except Exception:
            print(month, day, stock, "no data")
    fulldata = pd.concat(datadict)
    fulldata.to_csv(f"/data/work/yangsq/dataset_all/{stockname}fulldata.csv")
    use = time.time() - start
    print(f"usetime:{use:f}")


def build_feature_set(stockname):
    """Build predictors + response variables for a single stock and save training set."""
    print(stockname, "build_feature_set")
    start = time.time()
    fulldata = pd.read_csv(
        f"/data/work/yangsq/dataset_all/{stockname}fulldata.csv", parse_dates=True
    )
    fulldata = fulldata.rename(columns={"Unnamed: 0": "TradingDay", "Unnamed: 1": "TradesNum"})
    fulldata = fulldata.set_index("TradingDay")
    fulldata.index = pd.to_datetime(fulldata.index)
    fulldata = fulldata.loc["2020-01-01":"2020-12-31"]

    amountdict = {}
    for i in range(1, 10):
        amountdata = pd.read_csv(
            f"/data/work/yangsq/数据/20200{i}.csv",
            encoding="utf-8",
            encoding_errors="ignore",
        )
        amountdict[i] = amountdata.loc[amountdata.证券代码 == f"{stockname}.SZ"].T.iloc[2:]
    for i in range(10, 13):
        amountdata = pd.read_csv(
            f"/data/work/yangsq/数据/2020{i}.csv",
            encoding="utf-8",
            encoding_errors="ignore",
        )
        amountdict[i] = amountdata.loc[amountdata.证券代码 == f"{stockname}.SZ"].T.iloc[2:]
    amount2020 = pd.concat(amountdict).droplevel([1])
    amount2020.index = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    dateindex = fulldata.index
    today = 0

    response_variable = [
        "Return5s",
        "Return30s",
        "Return10trades",
        "Return200trades",
        "Return1000Volumes",
        "Return20000Volumes",
    ]
    predictor_label = [
        "Breadth",
        "Immediacy",
        "VolumeAll",
        "VolumeAvg",
        "VolumeMax",
        "Lambda",
        "LobImbalance",
        "TxnImbalance",
        "PastReturn",
        "Turnover",
        "AutoCov",
        "QuotedSpread",
        "EffectiveSpread",
    ]
    predictor_name = []
    for i in range(9):
        for j in range(13):
            predictor_name.append(predictor_label[j] + str(i))
    time_cal = [
        (0, 0.1),
        (0.1, 0.2),
        (0.2, 0.4),
        (0.4, 0.8),
        (0.8, 1.6),
        (1.6, 3.2),
        (3.2, 6.4),
        (6.4, 12.8),
        (12.8, 25.6),
    ]
    time_tran = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]
    time_vol = [
        (0, 100),
        (100, 200),
        (200, 400),
        (400, 800),
        (800, 1600),
        (1600, 3200),
        (3200, 6400),
        (6400, 12800),
        (12800, 25600),
    ]

    def response_calculator(data):
        data[response_variable] = 0
        data = data.reset_index().set_index("TradesNum")
        data.Return10trades = data.Price.rolling(10).mean() / data.Price - 1
        data.Return200trades = data.Price.rolling(200).mean() / data.Price - 1
        data = data.reset_index().set_index("Volumecum")
        data.Return1000Volumes = data.Price.rolling(1000).mean() / data.Price - 1
        data.Return20000Volumes = data.Price.rolling(20000).mean() / data.Price - 1
        data = data.reset_index().set_index("Time")
        data.index = pd.to_datetime(data.index, format="%H%M%S%f")
        data.Return5s = data.Price.rolling("5s").mean() / data.Price - 1
        data.Return30s = data.Price.rolling("30s").mean() / data.Price - 1
        return data

    def predictor_calculator(data, timetype, timescale):
        nonlocal today
        if timetype == "calendar":
            window_left = f"{time_cal[timescale][0] * 1000}ms"
            window_right = f"{time_cal[timescale][1] * 1000}ms"
            time_window = data.Time
        if timetype == "transaction":
            window_left = time_tran[timescale][0]
            window_right = time_tran[timescale][1]
            time_window = pd.Series(np.arange(len(data)), index=data.index)
        if timetype == "volume":
            window_left = time_vol[timescale][0]
            window_right = time_vol[timescale][1]
            time_window = data.Volumecum
        data.index = time_window
        breadth = data.Price.rolling(window_right, closed="left").count() - data.Price.rolling(
            window_left, closed="left"
        ).count()
        immediacy = (time_cal[timescale][1] - time_cal[timescale][0]) / breadth
        volume_all = data.TradeQty.rolling(window_right, closed="left").sum() - data.TradeQty.rolling(
            window_left, closed="left"
        ).sum()
        volume_avg = volume_all / breadth
        plt = data.Price - data.Price.shift(1)
        lambd = (plt.rolling(window_right, closed="left").sum() - plt.rolling(window_left, closed="left").sum()) / volume_all
        ispread = (data.OfferSize1 - data.BidSize1) / (data.OfferSize1 + data.BidSize1)
        lob_imbalance = (
            ispread.rolling(window_right, closed="left").sum()
            - ispread.rolling(window_left, closed="left").sum()
        ) / breadth
        direction = pd.Series(np.where(data.Direction == Side.BUY, 1.0, -1.0), index=data.index)
        txn_imbalance = (
            (data.TradeQty.multiply(direction)).rolling(window_right, closed="left").sum()
            - (data.TradeQty.multiply(direction)).rolling(window_left, closed="left").sum()
        ) / volume_all
        past_return = 1 - (data.Price.rolling(window_right).sum() - data.Price.rolling(window_left).sum()) / breadth / data.Price
        turnover = volume_all / amount2020.loc[dateindex[today]].iloc[0]
        today += 1
        log_plt1 = (data.Price / data.Price.shift(1)).apply(np.log)
        log_plt2 = (data.Price.shift(1) / data.Price.shift(2)).apply(np.log)
        auto_cov = (
            (log_plt1.multiply(log_plt2)).rolling(window_right, closed="left").sum()
            - (log_plt1.multiply(log_plt2)).rolling(window_left, closed="left").sum()
        ) / breadth
        spread = (data.OfferPX1 - data.BidPX1) / (data.OfferPX1 + data.BidPX1)
        quoted_spread = (
            spread.rolling(window_right, closed="left").sum()
            - spread.rolling(window_left, closed="left").sum()
        ) / breadth
        w_spread = log_plt1.multiply(direction).multiply(data.TradeQty).multiply(data.Price)
        weight = data.TradeQty.multiply(data.Price)
        effective_spread = (
            w_spread.rolling(window_right, closed="left").sum()
            - w_spread.rolling(window_left, closed="left").sum()
        ) / (weight.rolling(window_right, closed="left").sum() - weight.rolling(window_left, closed="left").sum())
        predictor = pd.concat(
            {
                predictor_name[13 * timescale + 0]: breadth,
                predictor_name[13 * timescale + 1]: immediacy,
                predictor_name[13 * timescale + 2]: volume_all,
                predictor_name[13 * timescale + 3]: volume_avg,
                predictor_name[13 * timescale + 5]: lambd,
                predictor_name[13 * timescale + 6]: lob_imbalance,
                predictor_name[13 * timescale + 7]: txn_imbalance,
                predictor_name[13 * timescale + 8]: past_return,
                predictor_name[13 * timescale + 9]: turnover,
                predictor_name[13 * timescale + 10]: auto_cov,
                predictor_name[13 * timescale + 11]: quoted_spread,
                predictor_name[13 * timescale + 12]: effective_spread,
            },
            axis=1,
        )
        return predictor

    def build_dataset(data_original, mode="calendar"):
        data = pd.DataFrame(data_original)
        data = data.dropna(how="any")
        data = data.sort_values("Time", ascending=True)
        data["Volumecum"] = data["TradeQty"].cumsum()
        data = data.sort_values("Time", ascending=False)
        data = response_calculator(data)
        data = data.sort_values("Time", ascending=True).reset_index()
        predictor_dict = {}
        if mode == "calendar":
            for i in range(9):
                predictor_dict[i] = predictor_calculator(data, "calendar", i)
            xset = pd.concat(predictor_dict, axis=1)
            yset = data[["Return5s", "Return30s"]]
        elif mode == "transaction":
            for i in range(9):
                predictor_dict[i] = predictor_calculator(data, "transaction", i)
            xset = pd.concat(predictor_dict, axis=1)
            yset = data[["Return10trades", "Return200trades"]].reset_index()
        elif mode == "volume":
            for i in range(9):
                predictor_dict[i] = predictor_calculator(data, "volume", i)
            xset = pd.concat(predictor_dict, axis=1)
            yset = data.reset_index().set_index("Volumecum")[["Return1000Volumes", "Return20000Volumes"]]
        trainset = xset.join(yset, how="left", sort=True).dropna(how="any", axis=0)
        return trainset

    trainset = fulldata.groupby("TradingDay").apply(build_dataset)
    print(trainset)

    end = time.time()
    use = end - start
    print(f"usetime:{use:f}")

    trainset.to_csv(f"/data/work/yangsq/trainset_all/{stockname}trainset.csv")


if __name__ == "__main__":
    stockname_list = ["002352"]  # Add more stock codes here if needed.
    for stockname in stockname_list:
        print(stockname)
        try:
            load_raw_data(stockname)
            build_feature_set(stockname)
        except Exception:
            pass
