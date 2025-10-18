import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
# import grs_test

tvl = pd.read_pickle("../../data/crypto_factors/tvl/FF_96_all_tvl/tvl_to_mktcap/tvl_quartiles.pkl")
tvl.describe()
##               hml          q1          q2          q3          q4
## count  112.000000  112.000000  112.000000  112.000000  112.000000
## mean    -0.359318    1.798688    1.683076    1.597705    1.439370
## std      5.654532    6.947088   11.018484    9.114407    8.645252
## min    -14.176822  -14.756142  -22.722244  -24.099774  -18.197445
## 25%     -3.368743   -2.168258   -4.244034   -3.201708   -3.994051
## 50%     -0.767612    0.285020   -0.425590    0.604415    0.224120
## 75%      1.795587    6.010557    5.992684    7.787926    6.009528
## max     19.745780   27.062163   44.410854   31.966032   28.316809

## This has old crypto factor portfolios (SMB, HML). 
## market_and_RF = pd.read_pickle("../../data/aggregated_for_fama_macbeth/data_tvl.pkl")
factors_5 = pd.read_csv("../../data/fama_french/F-F_Research_Data_5_Factors_2x3_daily.csv", skiprows=3)
factors_5 = factors_5.drop(factors_5.index[-1])
factors_5 = factors_5.set_index(pd.to_datetime(factors_5['Unnamed: 0'], format='%Y%m%d'))
factors_5 = factors_5.drop('Unnamed: 0', axis=1)
resample_weekly_on_day = 'W-MON'
weekly_factors_5 = factors_5.resample(resample_weekly_on_day).sum()

crypto_market = pd.read_pickle("../../data/crypto_factors/crypto_market_cap/coinmarketcap/crypto_market_rets.pkl")
crypto_market = crypto_market * 100
crypto_market = pd.concat([crypto_market, weekly_factors_5], axis=1, join="inner")

crypto_market = crypto_market["marketCap"] - crypto_market['RF']
smb = pd.read_pickle("../../data/crypto_factors/crypto_size/four_week_smb_monday_top_100.pkl")
mom = pd.read_pickle("../../data/crypto_factors/crypto_momentum/four_week_mom_monday_top_100.pkl")


data = pd.concat([tvl, crypto_market, smb, weekly_factors_5['RF'], weekly_factors_5['Mkt-RF'], mom], axis=1)
data = data.dropna()
data
data.columns = ['hml', 'q1', 'q2', 'q3', 'q4', 'crypto_market', 'smb', 'rf', 'stock_market', 'mom']
data['q1'] = data['q1'] - data['rf']
data['q2'] = data['q2'] - data['rf']
data['q3'] = data['q3'] - data['rf']
data['q4'] = data['q4'] - data['rf']
data.describe()

# 	hml	q1	q2	q3	q4	crypto_market	smb	rf	stock_market	mom
# count	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000
# mean	-0.254651	1.804691	1.746097	1.721038	1.550040	1.455698	-9.680845	0.094220	0.367339	0.788395
# std	5.671821	7.002269	11.103608	9.053976	8.625470	6.345350	32.909180	0.012394	1.906726	7.639105
# min	-14.176822	-14.864142	-22.817244	-24.207774	-18.292445	-15.422433	-206.528441	0.064000	-5.720000	-30.138574
# 25%	-3.355590	-2.134310	-4.310935	-3.221310	-3.568168	-1.890443	-6.409042	0.085000	-0.640000	-4.310581
# 50%	-0.762102	0.336389	-0.487374	0.513137	0.416144	0.252700	-0.865672	0.100000	0.460000	0.545163
# 75%	1.872155	5.923126	6.339473	7.910417	6.060966	5.725746	2.181655	0.105000	1.480000	4.480412
# max	19.745780	26.982163	44.310854	31.866032	28.216809	18.699366	18.538288	0.110000	5.620000	24.925163

stats.ttest_1samp(data["hml"], popmean=0)
# TtestResult(statistic=np.float64(-0.25636278473366486), pvalue=np.float64(0.7981771516661971), df=np.int64(104))
stats.ttest_1samp(data["q1"], popmean=0)
# TtestResult(statistic=np.float64(2.5820903715978796), pvalue=np.float64(0.011212235787415004), df=np.int64(104))
stats.ttest_1samp(data["q2"], popmean=0)
# TtestResult(statistic=np.float64(1.6835940983023436), pvalue=np.float64(0.09525957383376984), df=np.int64(104))
stats.ttest_1samp(data["q3"], popmean=0)
# TtestResult(statistic=np.float64(1.8615826192835392), pvalue=np.float64(0.06548572719965799), df=np.int64(104))
stats.ttest_1samp(data["q4"], popmean=0)
# TtestResult(statistic=np.float64(1.940715478094536), pvalue=np.float64(0.05499993284504132), df=np.int64(104))

### ADF tests on all variables
adfuller(data['hml'])
# (np.float64(-9.67426788452152),  np.float64(1.2529311638418632e-16), 1, 107
## reject null of unit root
## reject null for all at 0.1% => no unit roots

### Now merge in Algo excess returns------------------

algo = pd.read_csv("../old_data/ALGO_USD_daily.csv")
algo = algo.set_index(pd.to_datetime(algo["timestamp"]))
algo = algo.drop(["timestamp"], axis=1)
algo = algo["close"]
algo = algo.resample(resample_weekly_on_day).last()
algo = algo.pct_change()[1:] * 100
algo.index
data.index = pd.to_datetime(data.index)
data = pd.concat([algo, data], axis=1).dropna()
data
data.columns.values[0] = 'algo'
## excess algo returns----
data['algo'] = data['algo'] - data['rf']

data.columns.values[1] = 'tvl_hml'

data


#### data ready for markov switching model-------------------------





