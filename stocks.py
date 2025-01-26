import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from keras.callbacks import EarlyStopping
from sklearn import metrics


st.title("Real-Time Stock Data Analysis and Prediction")
st.write("This app retrieves stock data and predicts future prices using SVM, Random Forest, or LSTM.")

algorithm = st.selectbox("Choose Prediction Algorithm", ("SVM", "Random Forest", "LSTM"))

ticker = st.text_input("Enter Stock Ticker", value="^NSEI")

start_date = st.date_input("Select Start Date", value=dt.date(2024, 1, 1))
end_date = st.date_input("Select End Date", value=dt.date.today())

# Fetching data on button click
if st.button("Get Data"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.write("No data found for the given date range.")
    else:
        st.write("### Stock Data")
        st.write(df.tail())
        st.write("### Data Description")
        st.write(df.describe())
        st.write("### Missing Values")
        st.write(df.isna().any())

        # Calculate moving averages
        df['10 MA'] = df['Close'].rolling(window=10).mean()
        df['20 MA'] = df['Close'].rolling(window=20).mean()
        df_cleaned = df.dropna()

        # Plotting the Close price and moving averages
        st.write("### Closing Price and Moving Averages")
        plt.figure(figsize=(12, 6))
        plt.plot(df_cleaned.index, df_cleaned['Close'], label='Close Price', color='blue', linewidth=2)
        plt.plot(df_cleaned.index, df_cleaned['10 MA'], label='10-Day MA', color='orange', linestyle='--')
        plt.plot(df_cleaned.index, df_cleaned['20 MA'], label='20-Day MA', color='green', linestyle='--')
        plt.title(f'{ticker} Closing Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # Prepare data for SVM and Random Forest
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prediction'] = df['Prev_Close'].shift(-15)
        df = df.dropna()

        X = np.array(df[['Prev_Close']])
        y = np.array(df['Prediction'])
        X_train = X[:-15]
        y_train = y[:-15]
        x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        if algorithm == "SVM":
            svr_poly = SVR(kernel='poly', degree=2, C=1.0)
            svr_poly.fit(x_train, y_train)

            svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
            svr_rbf.fit(x_train, y_train)

            forecast = np.array(df[['Prev_Close']])[-15:]
            svm_prediction_poly = svr_poly.predict(forecast)
            svm_prediction_rbf = svr_rbf.predict(forecast)

            st.write("### Predicted Prices for the Next 15 Days (SVM Polynomial Kernel)")
            st.write(svm_prediction_poly)
            st.write("### Predicted Prices for the Next 15 Days (SVM RBF Kernel)")
            st.write(svm_prediction_rbf)

            # Model performance for SVM
            for model_name, model in zip(['Polynomial Kernel', 'RBF Kernel'], [svr_poly, svr_rbf]):
                y_pred_train = model.predict(x_train)
                st.write(f"### Model Performance for SVM - {model_name}")
                st.write(f"MAE: {(metrics.mean_absolute_error(y_train, y_pred_train) / y_train.mean()) * 100:.2f}%")
                st.write(f"MSE: {metrics.mean_squared_error(y_train, y_pred_train):.2f}")
                st.write(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)):.2f}")
                st.write(f"R² Score: {metrics.r2_score(y_train, y_pred_train):.2f}")

        elif algorithm == "Random Forest":
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, oob_score=True)
            rf_model.fit(x_train, y_train)

            forecast = np.array(df[['Prev_Close']])[-15:]
            rf_prediction = rf_model.predict(forecast)

            st.write("### Predicted Prices for the Next 15 Days (Random Forest)")
            st.write(rf_prediction)

            # Model performance for Random Forest
            y_pred_train = rf_model.predict(x_train)
            st.write(f"### Model Performance for Random Forest")
            st.write(f"MAE: {(metrics.mean_absolute_error(y_train, y_pred_train) / y_train.mean()) * 100:.2f}%")
            st.write(f"MSE: {metrics.mean_squared_error(y_train, y_pred_train):.2f}")
            st.write(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)):.2f}")
            st.write(f"R² Score: {metrics.r2_score(y_train, y_pred_train):.2f}")

        elif algorithm == "LSTM":
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[['Close']]) 

            # Prepare the data for LSTM
            def prepare_data_lstm(data, time_step=1):
                X, y = [], []
                for i in range(len(data) - time_step):
                    X.append(data[i:(i + time_step), :]) 
                    y.append(data[i + time_step, 0])      
                return np.array(X), np.array(y)

            time_step = 10
            X, y = prepare_data_lstm(scaled_data, time_step)

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1)) 

            model.compile(optimizer='adam', loss='mean_squared_error')

            early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[early_stop])

            predictions = model.predict(X_test)

            predictions = predictions.reshape(-1, 1)

            last_features = X_test[:, -1, :] 


            predictions_full = np.concatenate((last_features, predictions), axis=1)

            predictions_inverse = scaler.inverse_transform(predictions_full)[:, -1]

            y_test_full = np.concatenate((last_features, y_test.reshape(-1, 1)), axis=1)

            # Inverse transform the actual values
            y_test_actual = scaler.inverse_transform(y_test_full)[:, -1]

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_inverse))
            st.write("RMSE:", rmse)

            plt.figure(figsize=(14, 5))
            plt.plot(y_test_actual, label='Actual Prices', color='blue')
            plt.plot(predictions_inverse, label='LSTM Predictions', color='red')
            plt.title('Stock Price Prediction')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)
