0. 存储逻辑
1. 进行autoencoder的pretrain
```
bash pretrain.sh
```
使用datasets/sevir_pretrain.py中的数据集进行预训练，它是将原先的数据拆分成单帧得到的。
2. 进行确定式模型的训练，以训练EarthFormer为例:
```
bash train_deterministic.sh
```
3. 利用autoencoder预处理数据并存储
```
bash preprocess.sh
```
将原数据以及EarthFormer预测的结果以latent vector的形式存储，加速训练。
建议直接读我处理好的。
4. 进行diffusion模型的训练
```
bash train.sh
```

