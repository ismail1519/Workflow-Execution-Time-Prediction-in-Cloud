import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import pickle

def main():
    train_model()
    forecast_using_model()

def forecast_using_model():
    try:
        print("Workflow Utilization")

        matplotlib.rcParams['axes.labelsize'] = 14
        matplotlib.rcParams['xtick.labelsize'] = 12
        matplotlib.rcParams['ytick.labelsize'] = 12
        matplotlib.rcParams['text.color'] = 'k'

        df = pd.read_excel("WorkflowUtilization.xlsx", parse_dates=['Date'])
        df.set_index('Date',inplace=True)
        y = df['WorkflowUtilization'].resample('24h').mean()

        results = pickle.load(open("workflow_model.sav", 'rb'))

        # Forecast
        pred_uc = results.get_forecast(steps=100)
        pred_ci = pred_uc.conf_int()

        # Plotting
        ax = y.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Avg % WorkFlow Utilization')
        plt.legend()
        plt.show()
    except Exception as ex:
        print("Error!",str(ex),"occured.")

def train_model():
    try:
        df = pd.read_excel("WorkflowUtilization.xlsx", parse_dates=['Date'])
        df.set_index('Date',inplace=True)
        y = df['WorkflowUtilization'].resample('24h').mean()

        # Forecasting using SARIMAX
        mod = sm.tsa.statespace.SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)

        # Fit the model
        print("Fitting the model")
        results = mod.fit()

        # Save the model
        filename = 'workflow_model.sav'
        print("Saving the model to disk: Filename -",filename)
        pickle.dump(results, open(filename, 'wb'))

        print("Model trained.")
    except Exception as ex:
        print("Error!",str(ex),"occured.")


if __name__ == "__main__":
    main()