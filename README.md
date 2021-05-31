# DLP-project
💻 Deep Learning and Practice Final Project  
🏹 Stock Prediction using Transformer



## Dataset
Download data from [dataset](https://www.kaggle.com/jacksoncrow/stock-market-dataset).  
```shell
kaggle datasets download -d jacksoncrow/stock-market-dataset
```



## Arguments
|Argument|Description|Option|Default|
|---|---|---|---|
|`'-e', '--epochs'`|Number of epochs|int|100|
|`'-b', '--batch_size'`|Batch size|int|32|
|`'-s', '--seq_len'`|Sequence length (consecutive days)|int|128|
|`'-ne', '--num_encoder'`|Number of transformer encoder in the network|int|3|
|`'-a', '--attn_dim'`|Dimension of single attention output|int|96|
|`'-nh', '--num_heads'`|Number of heads for multi-attention|int|12|
|`'-d', '--dropout_rate'`|Dropout rate|float|0.1|
|`'-hs', '--hidden_size'`|Hidden size between the linear layers in the network|int|256|
|`'-r', '--root_dir'`|Directory containing the downloaded data|str|'archive'|
|`'-v', '--verbosity'`|Verbosity level|0-2|0|
