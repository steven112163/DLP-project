# DLP-project
💻 Deep Learning and Practice Final Project  
🏹 Stock Prediction using Transformer



## Dataset
Download data from [dataset](https://www.kaggle.com/jacksoncrow/stock-market-dataset) and unzip it into directory "archive".  
```shell
$ kaggle datasets download -d jacksoncrow/stock-market-dataset
```



## Reference
Reference is implemented in Tensorflow.  
[Stock predictions with state-of-the-art Transformer and Time Embeddings](https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6)



## Arguments
|Argument|Description|Option|Default|
|---|---|---|---|
|`'-e', '--epochs'`|Number of epochs|int|10|
|`'-w', '--warmup'`|Number of epochs for warmup|int|2|
|`'-l', '--learning_rate'`|Learning rate|float|0.001|
|`'-b', '--batch_size'`|Batch size|int|64|
|`'-s', '--seq_len'`|Sequence length (consecutive days)|int|128|
|`'-ne', '--num_encoder'`|Number of transformer encoder in the network|int|3|
|`'-a', '--attn_dim'`|Dimension of single attention output|int|96|
|`'-nh', '--num_heads'`|Number of heads for multi-attention|int|12|
|`'-d', '--dropout_rate'`|Dropout rate|float|0.3|
|`'-hs', '--hidden_size'`|Hidden size between the linear layers in the encoder|int|256|
|`'-loss', '--loss_function'`|Loss function|'l1' or 'l2'|'l2'|
|`'-i', '--inference_only'`|Inference only or not|'store_true'|False|
|`'-r', '--root_dir'`|Directory containing the downloaded data|str|'archive'|
|`'-v', '--verbosity'`|Verbosity level|0-2|0|
