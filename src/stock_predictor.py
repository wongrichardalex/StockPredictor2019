#install dependencies
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

#Reference:
#https://medium.com/@randerson112358/predict-stock-prices-using-python-machine-learning-53aa024da20a

def forecast(quandl_code, n_days):
    #get stock data
    stock_data = quandl.get(quandl_code)
    #look at data
    #print(stock_data.head())

    #get adjusted close price
    adjusted_close = stock_data[['Adj. Close']]
    #look at data
    #print(adjusted_close.head())

    new_data = adjusted_close

    #predicting 'n' days out, where forecast_out = n
    forecast_out = n_days
    #create another column (the target) shifted 'n' units up
    new_data['Prediction'] = adjusted_close[['Adj. Close']].shift(-forecast_out)
    #print new data set
    #print(new_data.tail())

    #create independent data set (X)
    #convert data frame to a numpy array
    X = np.array(new_data.drop(['Prediction'],1))

    #remove last '30' rows
    X = X[:-forecast_out]
    #print(X)

    #create dependent data set (Y)
    #convert data frame to numpy array
    Y = np.array(new_data['Prediction'])
    #get all of the y values except the last '30' rows
    Y = Y[:-forecast_out]
    #print(Y)

    #split data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    
    #set x_forecast equal to last 30 rows of the OG data set from adj. close column
    x_forecast = np.array(new_data.drop(['Prediction'], 1))[-forecast_out:]

    svm(x_train, y_train, x_test, y_test, x_forecast)
    lr(x_train, y_train, x_test, y_test, x_forecast)

##Support Vector Machine##
def svm(x_train, y_train, x_test, y_test, x_forecast):

    #create and train Support Vector Machine (Regressor)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)

    #testing model: score returns coefficient of determination R^2 of prediction
    #best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)
    print("svm confidence: ", svm_confidence)

    # print support vector regressor model predictions for the next '30' days
    svm_prediction = svr_rbf.predict(x_forecast)
    print(svm_prediction)
    
##Linear Regression Model##
def lr(x_train, y_train, x_test, y_test, x_forecast):

    #create and train Linear Regression Model
    lr = LinearRegression()
    #train model
    lr.fit(x_train, y_train)

    #testing model: score returns coefficient of determination R^2 of the prediction
    #best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)
    print("lr  confidence: ", lr_confidence)

    # print linear regression model predictions for the next '30' days
    lr_prediction = lr.predict(x_forecast)
    print(lr_prediction)

def main():
    print("Welcome to Top Tech Stock Predictor")

    companies = ["Amazon", "Google", "Apple", "Microsoft", "Facebook"]
    codes = ["WIKI/AMZN", "WIKI/GOOGL", "WIKI/AAPL", "WIKI/MSFT", "WIKI/FB"]

    cont = True

    while(cont):
        print("")
        
        ind = 0
        for company in companies:
            print(str(ind) + ": " + company)
            ind += 1
            
        print("")
        index = int(input("Please choose a company via number from the above list:" ))
        print("")
        print(companies[index] + ": ")
        forecast(codes[index], 30)
        cont = input("Continue? (y/n)") == "y"

main()




















