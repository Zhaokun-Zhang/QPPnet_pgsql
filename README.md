## 运行环境

见 `env.yml`



## 运行方法

#### 模型训练

```
$ python train.py
```

会根据parse.py中的默认参数进行训练

- latency预测

```
$ python train.py --type latency
```

- memory预测

```
$ python train.py --type memory
```

#### 模型评估

```
$ python test.py --type latency
```

评估模型在test dataset上的性能，同时会分别统计每一条query的性能

