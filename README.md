# Stock-Trading-Ensemble-Model
In this project, we use Ensemble Stock Trading Algorithm with RL Algorithm DDPG, PPO, and A2C.

# Result Presentation
![download-9](https://github.com/haodongzhang0118/Stock-Trading-Ensemble-Model/assets/128533950/e0c5a890-81ae-40b4-9339-728053c0acde)
From the result, our ensemble can get 16% return while the benchmark (market) can just get -2% return. This indicates that our result is useful and benefical

# Demo
The project.ipynb is the simple demo that using our ensemble stragety with several RL trained models to trade from 2022-04-01 to 2023-09-01.

We recommend download the github repo, and run our ipynb file in the colab. (Since the local kernal will die if when we try to run the plot session)

Here is the result of our demo:

# Set up the project

All files have been written well. Begin the project by just following the instructions below:

1) Download the dataset from Yahoo Finance with chosen Start Date (2012-01-01) to the End Date (2023-09-01)
2) Pre-Process the data with indicators that we need for training and trading.
Above two step has been finished by DataRetrieve.py. Using RetrieveTrainingData function to do all of these.
Attention: Since Yahoo API has access limitations, it is also a good choice to use the Pre_ProcessedFile.csv in our repo to get the data
