import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import arch
from arch import arch_model
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
data.index = pd.to_datetime(data.index)

# 	hml	q1	q2	q3	q4	crypto_market	smb	rf	stock_market	mom
# count	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000	109.000000
# mean	-0.254651	1.804691	1.746097	1.721038	1.550040	1.455698	-9.680845	0.094220	0.367339	0.788395
# std	5.671821	7.002269	11.103608	9.053976	8.625470	6.345350	32.909180	0.012394	1.906726	7.639105
# min	-14.176822	-14.864142	-22.817244	-24.207774	-18.292445	-15.422433	-206.528441	0.064000	-5.720000	-30.138574
# 25%	-3.355590	-2.134310	-4.310935	-3.221310	-3.568168	-1.890443	-6.409042	0.085000	-0.640000	-4.310581
# 50%	-0.762102	0.336389	-0.487374	0.513137	0.416144	0.252700	-0.865672	0.100000	0.460000	0.545163
# 75%	1.872155	5.923126	6.339473	7.910417	6.060966	5.725746	2.181655	0.105000	1.480000	4.480412
# max	19.745780	26.982163	44.310854	31.866032	28.216809	18.699366	18.538288	0.110000	5.620000	24.925163

# stats.ttest_1samp(data["hml"], popmean=0)
# # TtestResult(statistic=np.float64(-0.25636278473366486), pvalue=np.float64(0.7981771516661971), df=np.int64(104))
# stats.ttest_1samp(data["q1"], popmean=0)
# # TtestResult(statistic=np.float64(2.5820903715978796), pvalue=np.float64(0.011212235787415004), df=np.int64(104))
# stats.ttest_1samp(data["q2"], popmean=0)
# # TtestResult(statistic=np.float64(1.6835940983023436), pvalue=np.float64(0.09525957383376984), df=np.int64(104))
# stats.ttest_1samp(data["q3"], popmean=0)
# # TtestResult(statistic=np.float64(1.8615826192835392), pvalue=np.float64(0.06548572719965799), df=np.int64(104))
# stats.ttest_1samp(data["q4"], popmean=0)
# # TtestResult(statistic=np.float64(1.940715478094536), pvalue=np.float64(0.05499993284504132), df=np.int64(104))

### ADF tests on all variables
# adfuller(data['hml'])
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

final_data = pd.concat([algo, data], axis=1).dropna()
final_data
# final_data.columns.values[0] = 'algo'
# final_data.columns.values[1] = 'tvl_hml'
## excess algo returns----
# final_data['algo'] = final_data['algo'] - final_data['rf']
final_data['close'] = final_data['close'] - final_data['rf']



final_data.to_csv("final_data.csv")
final_data.to_pickle("final_data.pkl")


#### data ready for markov switching model-------------------------

exog = final_data[['crypto_market', 'hml', 'smb']]
# Fit the model
mod_areturns = sm.tsa.MarkovRegression(
    final_data['close'],
    k_regimes=2,
    exog=exog, #weekly_data_3_factor['Mkt-RF']
    switching_variance=True,
)
res_areturns = mod_areturns.fit()

res_areturns.summary()

plt.clf()

res_areturns.smoothed_marginal_probabilities[1].plot(
    title="Probability of high-variance regime: Crypto 3 Factor Model", figsize=(12, 3)
)

# plt.show()
plt.savefig("two_regimes_3_factor.pdf")
# plt.savefig("two_regimes_3_factor.png")

#                         Markov Switching Model Results                        
# ==============================================================================
# Dep. Variable:                  close   No. Observations:                  109
# Model:               MarkovRegression   Log Likelihood                -385.417
# Date:                Mon, 20 Oct 2025   AIC                            812.835
# Time:                        19:12:57   BIC                            869.353
# Sample:                    01-02-2023   HQIC                           835.755
#                          - 01-27-2025                                         
# Covariance Type:               approx                                         
#                              Regime 0 parameters                              
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -5.0794      0.862     -5.891      0.000      -6.769      -3.389
# x1             0.3181      0.073      4.381      0.000       0.176       0.460
# x2             0.1822      0.060      3.059      0.002       0.065       0.299
# x3            -0.0336      0.012     -2.888      0.004      -0.056      -0.011
# sigma2         1.7364      0.675      2.571      0.010       0.412       3.060
#                              Regime 1 parameters                              
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -1.1577      1.411     -0.820      0.412      -3.923       1.608
# x1             1.2633      0.190      6.654      0.000       0.891       1.635
# x2             0.0469      0.372      0.126      0.900      -0.683       0.776
# x3            -0.0268      0.030     -0.899      0.368      -0.085       0.032
# sigma2        36.1407     44.104      0.819      0.413     -50.302     122.584
#                              Regime 2 parameters                              
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          4.6974     11.611      0.405      0.686     -18.059      27.454
# x1             2.3030      0.802      2.873      0.004       0.732       3.874
# x2             0.3924      0.827      0.474      0.635      -1.229       2.014
# x3            -0.0178      0.179     -0.099      0.921      -0.368       0.333
# sigma2       427.9292    595.704      0.718      0.473    -739.629    1595.488
#                          Regime transition parameters                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# p[0->0]        0.0628      0.270      0.232      0.816      -0.467       0.593
# p[1->0]        0.1864      0.078      2.402      0.016       0.034       0.338
# p[2->0]        0.1334      1.220      0.109      0.913      -2.258       2.525
# p[0->1]        0.9355        nan        nan        nan         nan         nan
# p[1->1]        0.7694      0.165      4.664      0.000       0.446       1.093
# p[2->1]        0.0001      2.491   5.57e-05      1.000      -4.881       4.881
# ==============================================================================


##### market model


exog = final_data[['crypto_market']]
# Fit the model
mod_areturns = sm.tsa.MarkovRegression(
    final_data['close'],
    k_regimes=2,
    exog=exog, #weekly_data_3_factor['Mkt-RF']
    switching_variance=True,
)
res_areturns = mod_areturns.fit()

res_areturns.summary()

plt.clf()

res_areturns.smoothed_marginal_probabilities[1].plot(
    title="Probability of high-variance regime: Crypto Market Model", figsize=(12, 3)
)

# plt.show()
plt.savefig("two_regimes_market_model.pdf")
plt.savefig("two_regimes_market_model.jpeg")
plt.savefig("two_regimes_market_model.png")

#                         Markov Switching Model Results                        
# ==============================================================================
# Dep. Variable:                  close   No. Observations:                  109
# Model:               MarkovRegression   Log Likelihood                -394.679
# Date:                Mon, 20 Oct 2025   AIC                            805.358
# Time:                        19:22:59   BIC                            826.888
# Sample:                    01-02-2023   HQIC                           814.089
#                          - 01-27-2025                                         
# Covariance Type:               approx                                         
#                              Regime 0 parameters                              
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -1.6282      0.740     -2.201      0.028      -3.078      -0.179
# x1             1.0391      0.125      8.296      0.000       0.794       1.285
# sigma2        43.6980      9.668      4.520      0.000      24.748      62.648
#                              Regime 1 parameters                              
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          3.9621      5.637      0.703      0.482      -7.087      15.011
# x1             2.4169      0.730      3.310      0.001       0.986       3.848
# sigma2       460.1312    178.208      2.582      0.010     110.850     809.412
#                          Regime transition parameters                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# p[0->0]        0.9626      0.026     36.529      0.000       0.911       1.014
# p[1->0]        0.1570      0.123      1.277      0.202      -0.084       0.398
# ==============================================================================


## one regime CAPM

exog = final_data[['crypto_market']]

# Fit and summarize OLS model
mod = sm.OLS(final_data['close'], sm.add_constant(exog))

res = mod.fit()

print(res.summary())

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                  close   R-squared:                       0.351
# Model:                            OLS   Adj. R-squared:                  0.345
# Method:                 Least Squares   F-statistic:                     57.83
# Date:                Wed, 22 Oct 2025   Prob (F-statistic):           1.17e-11
# Time:                        20:59:23   Log-Likelihood:                -425.17
# No. Observations:                 109   AIC:                             854.3
# Df Residuals:                     107   BIC:                             859.7
# Df Model:                           1                                         
# Covariance Type:            nonrobust                                         
# =================================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# const            -0.4312      1.187     -0.363      0.717      -2.784       1.921
# crypto_market     1.3922      0.183      7.604      0.000       1.029       1.755
# ==============================================================================
# Omnibus:                      112.999   Durbin-Watson:                   2.256
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2454.038
# Skew:                           3.278   Prob(JB):                         0.00
# Kurtosis:                      25.302   Cond. No.                         6.66
# ==============================================================================

## Get the estimated variance of the error term
res.scale

## Get market realized vol-----

np.sqrt(final_data['crypto_market']**2).plot()
## plt.show()
plt.savefig("market_realized_vol.png")
plt.clf()


### GARCH

# Fit GARCH(1,1) model

model = arch_model(final_data['crypto_market'], vol='Garch', p=1, o=1, q=1, power=1.0, dist="StudentsT")
results = model.fit(disp='off')

# Display results
print(results.summary())

# Extract key outputs
conditional_volatility = results.conditional_volatility
standardized_residuals = results.std_resid
parameters = results.params

conditional_volatility.plot()
#plt.show()
plt.savefig("zgarch_studentT.png")

####
model = arch_model(final_data['crypto_market'], vol='EGARCH', p=1, q=1, mean='Constant')
results = model.fit(disp='on')

# Extract conditional volatility (same as GARCH)
conditional_vol = results.conditional_volatility

conditional_vol.plot()
plt.show()


np.abs(final_data['crypto_market']).plot()
plt.show()


###  stock market vol-----

model = arch_model(final_data['stock_market'], vol='GARCH', p=1, q=1, mean='Constant')
results = model.fit(disp='on')

# Extract conditional volatility (same as GARCH)
conditional_vol = results.conditional_volatility

conditional_vol.plot()
plt.show()
