# 设计上采用整体到局部的方式

1. 首先定义一个transformer的基本架构
分别由encoder decoder embedding+position embedding构成
小功能上 将mask在这里处理。 由于我们mask的实现方式是基于token的所以在这里实现mask的转换。

2. encoder将包含n层encoder_layer，decoder将包含n层decoder_layer

3. encoder_layer 调用mutilhead_attention、add_norm和 ff_layer实现数据的流动

4. mutilhead_attention将包含 self_attention

## 接下来是最基础的几个构件

1. self_attention将在mutilhead_attention分头后进行注意力打分，并返回 V = Scorces*V
2. add_norm 和 ff_layer会在encoder_layer通过 pytorch的api实现
3. mask将在token_ids上用tensor的广播机制构建
