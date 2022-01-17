#.CAPenv\scripts\activate
#for requirements file
### LIBRARIES & Setup
import math
import praw
import requests
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from psaw import PushshiftAPI
from datetime import timedelta
import pandas_datareader as web
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews
from keras.models import Sequential
from keras.layers import Dense, LSTM
from flask import Flask, render_template
from flask.templating import render_template
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

api = PushshiftAPI()
weight = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
i = 0

############################################## ROBINHOOD ##############################################
stock_names = []
stock_symbol = []

header= {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36'}
page= requests.get("https://stonks.news/top-100/robinhood", headers=header)
soup = BeautifulSoup(page.content, 'html.parser')

stocks = soup.find(class_='ant-table-tbody')
stock_picks = stocks.find_all(class_='ant-table-row ant-table-row-level-0')

# Get stock names
for stock in stock_picks:
    stock_names.append(stock.td.next_sibling.next_sibling.get_text())
    i+=1
    if i == 20:
        i = 0
        break

for stock in stock_picks:
    stock_symbol.append(stock.a.get_text())
    i+=1
    if i == 20:
        i = 0
        break

data = {'Symbol':stock_symbol, 'Company':stock_names, "Weight":weight}
df1 = pd.DataFrame(data)
df1

### REDDIT scrape
stock_names = []
stock_symbol = []

header= {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36'}
page = requests.get("https://apewisdom.io", headers=header)
soup = BeautifulSoup(page.content, 'html.parser')

stocks = soup.find(class_='table marketcap-table dataTable default-table')
stock_picks = stocks.find_all(class_='company-name')
symbol = stocks.find_all(class_=['badge badge-company', 'badge badge-etf'])

for stock in stock_picks:
    stock_names.append(stock.get_text())
    i+=1
    if i == 20:
        i = 0
        break

for stock in symbol:
    stock_symbol.append(stock.get_text())
    i+=1
    if i == 20:
        i = 0
        break
    
data = {'Symbol':stock_symbol, 'Company':stock_names, 'Weight':weight}
df2 = pd.DataFrame(data)
df2['Company'] = df2['Company'].str.replace(r'\r\n', '')
df2

### YAHOO FINANCE scrape
stock_names = []
stock_symbol = []

header= {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36'}
#page= requests.get("https://finance.yahoo.com/trending-tickers", headers=header)
page= requests.get("https://finance.yahoo.com/most-active", headers=header)
#lol yahoo finance is just crypto on weekends..... new link used
soup = BeautifulSoup(page.content, 'html.parser')

stocks = soup.find(class_='W(100%)')
stock_picks = stocks.find_all(class_='simpTblRow')
#get rid of crypto tickers
crypto = "USD"

# Get stock names
for stock in stock_picks:
    if crypto in stock.td.next_sibling.get_text():
        continue
    else:
        stock_names.append(stock.td.next_sibling.get_text())
        i+=1
        if i == 20:
            i = 0
            break

for stock in stock_picks:
    if crypto in stock.a.get_text():
        continue
    else:
        stock_symbol.append(stock.a.get_text())
        i+=1
        if i == 20:
            i = 0
            break

data = {'Symbol':stock_symbol, 'Company':stock_names, "Weight":weight}
df3 = pd.DataFrame(data)
df3

### FINVIZ scrape
company_name = []
company_ticker = []  

header= {'User-Agent':'PostmanRuntime/7.28.4'}  
#URL = 'https://finviz.com/screener.ashx?v=111&s=n_majornews&f=geo_usa'
URL = 'https://finviz.com/screener.ashx?v=111&s=ta_unusualvolume&f=geo_usa'
#When testing this link only had 17 stocks.... so im using another link instead

payload={}
headers = {
'User-Agent': 'PostmanRuntime/7.28.4'
}

page = requests.request("GET", URL, headers=headers, data=payload)
soup = BeautifulSoup(page.text,'html.parser')

dark_rows = soup.find_all('tr',attrs={'class':'table-dark-row-cp'})
light_rows = soup.find_all('tr', attrs={'class':'table-light-row-cp'})

for i in dark_rows: 
        row = i.find_all('td')
        company_name.append(row[2].text.strip()) #company name 
        company_ticker.append(row[1].text.strip()) #ticker name
        
for i in light_rows: 
        row = i.find_all('td')
        company_name.append(row[2].text.strip()) #company name 
        company_ticker.append(row[1].text.strip()) #ticker name
        
data = {'Symbol':company_ticker, 'Company':company_name, "Weight":weight}
df4 = pd.DataFrame(data)
df4

### APPENDING df's 
dfm = df1.append([df2, df3, df4], ignore_index=True)
dfc= dfm.groupby(["Symbol"]).agg({'Weight': 'sum', 'Company': 'first'})
dfc = dfc.sort_values(by=['Weight'], ascending=False)
dfc = dfc.rename_axis('Symbol').reset_index()
#############################################################################################################chnaged this to 3 :) my laptop cant take the heat
dfc = dfc[:3]
dfc
print('### STOCKS TO BE FORECASTED ###')
print(dfc)
print('###############################')

############################################## SENTIMENT ANALYSIS IS STARTING ##############################################

#### REDDIT SENTIMENT 
reddit = praw.Reddit(client_id="AjwZL1WQ39xebSOVf0vg_Q", #my client id
                    client_secret="XRbzVJP7X9TW09dH7jmHAYTNRqfqQw",  #your client secret
                    user_agent="candycloud", #user agent name
                    username = "IRMGang",     # your reddit username
                    password = "GiveUsAnA+")     # your reddit password

path = r'C:\Users\youss\Desktop\spam\\'

sub = ['WallStreetBets'
    , 'Stocks', 'Wallstreetbetsnew', 'WallStreetbetsELITE', 'StockMarket', 
    'investing', 'SPACs', 'options', 'Daytrading', 'Shortsqueeze'
        ]   #make a list of subreddits you want to scrape the data from

posts = pd.DataFrame()
comments = pd.DataFrame()

for index, i in dfc.iterrows():
    dfc.update('"' + i[['Symbol']].astype(str) + '"')
    query = i["Symbol"]
    for s in sub:
        subreddit = reddit.subreddit(s)
        post_dict = {
        "subreddit" : [],
        "stock" : [],
        "title" : [],
        "score" : [],
        "id" : [],
        "url" : [],
        "comms_num": [],
        "created" : [],
        "body" : []
        }
        comments_dict = {
        "subreddit" : [],
        "comment_id" : [],
        "comment_parent_id" : [],
        "stock" : [],
        "comment_body" : [],
        "comment_link_id" : []
        }
        for submission in tqdm(api.search_submissions(q=query, after="24h", subreddit=subreddit)):
            post_dict["subreddit"].append(submission.subreddit)
            post_dict["stock"].append(query)
            post_dict["title"].append(submission.title)
            post_dict["score"].append(submission.score)
            post_dict["id"].append(submission.id)
            post_dict["url"].append(submission.url)
            post_dict["comms_num"].append(submission.num_comments)
            post_dict["created"].append(submission.created)
            post_dict["body"].append(submission.selftext)
        for comment in tqdm(api.search_comments(q=query, after="24h", subreddit=subreddit)):
            comments_dict["subreddit"].append(comment.subreddit)
            comments_dict["stock"].append(query)
            comments_dict["comment_id"].append(comment.id)
            comments_dict["comment_parent_id"].append(comment.parent_id)
            comments_dict["comment_body"].append(comment.body)
            comments_dict["comment_link_id"].append(comment.link_id)
        post_data = pd.DataFrame(post_dict)
        posts = posts.append(post_data, ignore_index=True)
        post_comments = pd.DataFrame(comments_dict)
        comments = comments.append(post_comments, ignore_index=True)
        
### GOOGLE NEWS SENTIMENT 

news_data = pd.DataFrame()
for index, i in tqdm(dfc.iterrows()):
    news_dict = {
        "name" : [],
        "title" : [],
        "stock" : []}
    googlenews = GoogleNews()
    googlenews.set_period('1d')
    dfc.update('"' + i[['Company']].astype(str) + '"')
    query = i['Company']
    company = i["Symbol"]
    googlenews.get_news(query)
    news = googlenews.get_texts()
    for i in range(len(news)):
        news_dict["name"].append(query)
        news_dict["title"].append(news[i])
        news_dict["stock"].append(company)
    news_frame = pd.DataFrame(news_dict)
    news_data = news_data.append(news_frame)
    
### VADER ANALYSIS

analyzer = SentimentIntensityAnalyzer()
sentiment_values = []
sentiment_values_body = []
sentiment_values_comments = []
sentiment_values_news = []
for index, i in posts.iterrows():
    sentiment = analyzer.polarity_scores(i['title'])
    sentiment_values.append(sentiment["compound"])
posts['sentiment'] = sentiment_values
for index, i in comments.iterrows():
    sentiment = analyzer.polarity_scores(i['comment_body'])
    sentiment_values_comments.append(sentiment["compound"])
comments['sentiment'] = sentiment_values_comments
for index, i in news_data.iterrows():
    sentiment = analyzer.polarity_scores(i['title'])
    sentiment_values_news.append(sentiment["compound"])
news_data['sentiment'] = sentiment_values_news
for index, i in posts.iterrows():
    sentiment = analyzer.polarity_scores(i['body'])
    sentiment_values_body.append(sentiment["compound"])
posts['sentiment body'] = sentiment_values_body
posts["sentiment"] = posts[["sentiment","sentiment body"]].mean(axis=1)
posts = posts.drop(columns="sentiment body")

### APPEND FINAL df 

final = posts.append([comments, news_data], ignore_index=True)
final = final.groupby(['stock']).agg('mean')
final = final.sort_values(by=['sentiment'], ascending=False)
final = final.rename_axis('Symbol').reset_index()
final = final.drop(columns=["score","comms_num","created"])
final
print('###### SENTIMENT SCORES ######')
print(final)
print('###########################################################################################')

############################################## PREDICTION IS STARTING ##############################################
#demo predition https://github.com/thatguuyG/Stock-Prediction
#https://matplotlib.org/stable/gallery/style_sheets/fivethirtyeight.html
plt.style.use('fivethirtyeight')

#final to a list of tickers
FinalMoney = final['Symbol'].to_list()
print(FinalMoney)

Today = datetime.datetime.now()
#variable for going back in time for validation
StartDay = 333
Todayday = datetime.date.today()

print(Today)
#These are to get the start and end dates to pull historical data from
#EndDay is next weekday
#https://docs.python.org/3/library/datetime.html
EndDay = datetime.date.today()
print('Today date:')
print(EndDay)

if Today.time() < datetime.time(9, 30):
    print(Today.time())
    print("market has not been open today")
    if EndDay.weekday() in set((5, 6)):
        EndDay += datetime.timedelta(days=7 - EndDay.weekday())
        X_FUTURE = (EndDay - Todayday).days
        print(X_FUTURE)
    else:
        EndDay = datetime.date.today()
        X_FUTURE = 1
        print(X_FUTURE)
else:
    print(Today.time())
    print("market has been open today")
    if EndDay.weekday() in set((4, 5)):
        EndDay += datetime.timedelta(days=7 - EndDay.weekday())
        X_FUTURE = (EndDay - Todayday).days
        print(X_FUTURE)
    else:
        EndDay = datetime.date.today() + datetime.timedelta(days=1)
        X_FUTURE = 1
        print(X_FUTURE)


print(EndDay)

#startday is startday days ago
StartDay = EndDay - timedelta(days=StartDay)
print('######### Using data from: ', StartDay, '& predicting to:', EndDay, ' ########')

#### https://docs.python.org/3/library/datetime.html


appended_names = []
appended_predictons = []
appended_senti = []


#The loooop to go thru the list of tickers and sentiment scores
for x in FinalMoney:
    #this pulls the historical data
    df = web.DataReader(x, data_source='yahoo', start=StartDay, end=EndDay)

    print('THIS TICKERs TURN')
    #print(x)

    #to append the names of stocks
    appended_names.append(x)

    #Dataframe with 'close' prices
    #this will be using the close price for the date
    # Using Open price to be exploered in analysis
    data = df.filter(['Close'])
    print(data)

    #Dataframe into a numpy array
    #numpy facilitates use of python packs
    #https://numpy.org/doc/stable/user/whatisnumpy.html
    dataset = data.values

    # compute number of rows to train the modelon
    # Pulling 333 days of data based on StartDay variables
    # thats why the days for the prediction are today and today - 333 
    # https://docs.python.org/3/library/math.html   
    training_data_len = math.ceil(len(dataset) * .8)

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    # normalization of data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    #Trainer dataset
    train_data = scaled_data[0:training_data_len, :]

    # split into x_train and y_train data sets
    #30 for 30 day prediction
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    #guidance on training datasets 
    x_train = []
    y_train = []
    for i in range(30, len(train_data)):
        x_train.append(train_data[i-30:i, 0])
        y_train.append(train_data[i, 0])

    # convert x and y train into numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshape the data into the shape accepted by the LTSM
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # Gives a new shape to an array without changing its data.
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

    # Build the LTSM network model
    # Stacked LSTM Model
    # https://faroit.com/keras-docs/1.0.1/getting-started/sequential-model-guide/
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    #compile the model
    #https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    model.compile(optimizer='adam', loss='mean_squared_error')

    #train the model
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model 
    ############################################################################################################# epochs to 3 :) my laptop cant take the heat
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    #test dataset
    test_data = scaled_data[training_data_len - 30: , : ]

    #create the x_test and y_test datasets
    x_test=[]
    y_test =  dataset[training_data_len : , : ] 
    #Get all of the rows from index
    for i in range(30,len(test_data)):
        x_test.append(test_data[i-30:i,0])

    #Convert x_test to a numpy array 
    x_test = np.array(x_test)

    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)#Undo scaling

    #calculate how accurate the model is by getting the RMSE
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    print('THE R.M.S.E.')
    print(rmse)

    #Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    print('THE VALIDATION FORECAST')
    print(valid)
    #valid is the the 60 day historical perdiction validation 
    
    ##
    #https://stackoverflow.com/questions/65237843/predicting-stock-price-x-days-into-the-future-using-python-machine-learning
    predictions = np.array([])
    last = x_test[-1]
    for i in range(X_FUTURE):
        curr_prediction = model.predict(np.array([last]))
        last = np.concatenate([last[1:], curr_prediction])
        predictions = np.concatenate([predictions, curr_prediction[0]])
    
    predictions = scaler.inverse_transform([predictions])[0]
    #print(predictions)

    #next weekday
    dicts = []
    for i in range(X_FUTURE):
        dicts.append({'Predictions':predictions[i], "Date": EndDay})

    new_data = pd.DataFrame(dicts).set_index("Date")  
    print(new_data)

    #new data is the future prediction.
    #on friday its printing out 3 results all for the following monday

    #https://stackoverflow.com/questions/65237843/predicting-stock-price-x-days-into-the-future-using-python-machine-learning
    ##
    ### Sentiment score
    y = final.loc [ final [ 'Symbol' ] == x]
    q = y['sentiment']
    h = q.to_list()
    i = [float(n) for n in h]
    print('THIS IS THE SENTIMENT SCORE')
    print(i)

    #to append the names of stocks
    appended_senti.append(i)

    #for prediction sentiment
    sentiF = new_data['Predictions'] + (new_data['Predictions'] * i / 100)   
    #print(sentiF)
    ################################ chnage #######################################

    #get only next day prediction
    SentimentPredictions = sentiF.head(1)
    #print(SentimentPredictions)

    #Next day prediction with sentiment to a df
    df44 = pd.DataFrame(SentimentPredictions)
    print(df44)

    #historical data to a df
    df55 = pd.DataFrame(data)
    #print(data)
    print(df55)

    #concat df's together
    toconcat = [df55, df44]
    finalll = pd.concat(toconcat)
    print(finalll)

    pandascells = finalll.to_records(index=True)
    print(pandascells)

    appended_predictons.append(pandascells)
    print(appended_predictons)

    print('End of Prediction')

ticker_1_data = appended_predictons[0]
ticker_2_data = appended_predictons[1]
ticker_3_data = appended_predictons[2]
print(ticker_1_data)
print(ticker_2_data)
print(ticker_3_data)

ticker_1_name = appended_names[0]
ticker_2_name = appended_names[1]
ticker_3_name = appended_names[2]
print(ticker_1_name)
print(ticker_2_name)
print(ticker_3_name)

ticker_1_senti = appended_senti[0]
ticker_2_senti = appended_senti[1]
ticker_3_senti = appended_senti[2]
print(ticker_1_senti)
print(ticker_2_senti)
print(ticker_3_senti)

@app.route("/")
def homepage():
    return render_template("homepage.html", ticker_1_data=ticker_1_data, ticker_2_data=ticker_2_data, ticker_3_data=ticker_3_data, ticker_1_name=ticker_1_name, ticker_2_name=ticker_2_name, ticker_3_name=ticker_3_name, ticker_1_senti=ticker_1_senti, ticker_2_senti=ticker_2_senti, ticker_3_senti=ticker_3_senti)

@app.route("/team/")
def teaminfo():
    return render_template("teampage.html")

if __name__ == "__main__":
    app.run()

