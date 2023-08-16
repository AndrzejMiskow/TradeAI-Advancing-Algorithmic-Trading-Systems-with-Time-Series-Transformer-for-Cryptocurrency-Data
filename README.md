# TradeAI


Integrating deep learning methods into algorithmic trading systems is advancing the financial industry,
enabling sophisticated analysis and decision-making capabilities previously limited to institutional investors, 
making them accessible to retail investors. However, the application of deep learning in medium-
frequency trading, particularly in the cryptocurrency market, remains largely unexplored. This research
aims to bridge this gap by investigating the feasibility and effectiveness of integrating deep learning
methods into an algorithmic trading system specifically tailored for cryptocurrency trading. This study
presents a comprehensive full-stack algorithmic trading system that integrates deep learning methods for
trading strategies. Specifically, we delve into utilising novel transformer architectures, including Informer,
Pyraformer, and an enhanced original transformer, into predicting cryptocurrency prices. Our research
findings showcase these transformer models’ superior performance compared to traditional ARIMA models, 
particularly when operating on larger datasets. Notably, the Pyraformer model exhibits exceptional
predictive accuracy while maintaining efficient training and inference times. Moreover, we seamlessly 
integrate these predictive signals into the environment definition of a Deep Reinforcement Learning (DRL)
model, enabling effective order generation and decision-making. The findings contribute to understanding 
transformer models’ effectiveness in medium-frequency cryptocurrency price prediction and provide
a promising architecture for future research and development in this evolving field.

## SETUP

- Setup Commands

  ```
  conda create -n tradeai python=3.8 -y

  conda activate tradeai

  pip install --upgrade pip setuptools wheel

  pip install -r requirements.txt

  sudo apt-get install git-lfs (linux)

  brew install git-lfs (homebrew)

  choco install git-lfs (microsoft)
  ```

- Running TradeAI (locally and in container)

  ```
  ./run_locally.sh
  ```

- Run Client (connecting to local TradeAI)

  ```
  npm install

  npm start
  ```

- Running TradeAI (in azure; extra requirements: docker, docker-compose, az)

  ```
  ./run_in_azure.sh
  ```

- How to use DVC to manage data

  1. Ask to be added to DVC remote repository

  2. dvc pull - should be prompted to login into google account.

  3. Modify data

  4. dvc add [data_file]

  5. Commit changes to [data_file].csv.dvc to git

  6. dvc push
  
