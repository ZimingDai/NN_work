# NN_work

横向对比GCN、GRU、TGCN、HA和ARIMA

## Python 依赖

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning
* torchmetrics
* python-dotenv



## 训练集

训练集在NN_work 的data中。



## 训练方法

```bash
# GCN
python main.py --model_name GCN --data city --max_epochs 500 --learning_rate 0.001 --weight_decay 0 --batch_size 64 --hidden_dim 100 --settings
# GRU
python main.py --model_name GRU --data city --max_epochs 100 --learning_rate 0.001 --weight_decay 1.5e-3 --batch_size 64 --hidden_dim 100 --settings
# T-GCN
python main.py --model_name TGCN --data city --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings

# GCN
python main.py --model_name GCN --data los --max_epochs 500 --learning_rate 0.001 --weight_decay 0 --batch_size 64 --hidden_dim 100 --settings
# GRU
python main.py --model_name GRU --data los --max_epochs 100 --learning_rate 0.001 --weight_decay 1.5e-3 --batch_size 64 --hidden_dim 100 --settings
# T-GCN
python main.py --model_name TGCN --data los --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings


# ARIMA
python ARIMA_main.py

# HA
python HA_main.py
```

