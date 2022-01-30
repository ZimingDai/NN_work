import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


class Arima_Class:
    def __init__(self, arima_para, seasonal_para):
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        p = arima_para['p']
        d = arima_para['d']
        q = arima_para['q']
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(p, d, q))  # 三个参数的各种组合
        self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_para)
                             for x in list(itertools.product(p, d, q))]

    def fit(self, ts):
        warnings.filterwarnings("ignore")  # 过滤掉warning
        results_list = []
        for param in self.pdq:
            for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(ts,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    results_list.append([param, param_seasonal, results.aic])

                except:
                    continue
        results_list = np.array(results_list)
        lowest_AIC = np.argmin(results_list[:, 2])
        # 寻找到最小的AIC之后，用这个order进行训练。
        mod = sm.tsa.statespace.SARIMAX(ts,
                                        order=results_list[lowest_AIC, 0],
                                        seasonal_order=results_list[lowest_AIC, 1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.final_result = mod.fit()

    def pred(self, pred_start, data):
        pred_dynamic = self.final_result.get_prediction(
            start=pd.to_datetime(pred_start), dynamic=True, full_results=True)
        # ax = data[pred_start:].plot(label="observed", figsize=(15,10))
        # pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
        # ax.set_xlabel('Time')
        # ax.set_ylabel('4G')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        return pred_dynamic.predicted_mean

    def forecast(self, ts, n_steps, ts_label):
        # Get forecast n_steps ahead in future
        pred_uc = self.final_result.get_forecast(steps=n_steps)
        # ax = ts.plot(label='observed', figsize=(15, 10))
        # pred_uc.predicted_mean.plot(ax=ax, label='Forecast in Future')
        # ax.set_xlabel('Time')
        # ax.set_ylabel(ts_label)
        # plt.tight_layout()
        # plt.legend()
        # plt.show()
        return pred_uc.predicted_mean

