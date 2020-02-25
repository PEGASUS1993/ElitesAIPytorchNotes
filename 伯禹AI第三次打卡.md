
# 第三次打卡 2020.2.25 Word2vec,优化算法进阶，模型微调

# 1.词嵌入基础

我们使用 one-hot 向量表示单词，虽然它们构造起来很容易，但通常并不是一个好选择。一个主要的原因是，one-hot 词向量无法准确表达不同词之间的相似度，如我们常常使用的余弦相似度。

Word2Vec 词嵌入工具的提出正是为了解决上面这个问题，它将每个词表示成一个定长的向量，并通过在语料库上的预训练使得这些向量能较好地表达不同词之间的相似和类比关系，以引入一定的语义信息。基于两种概率模型的假设，我们可以定义两种 Word2Vec 模型：
1. [Skip-Gram 跳字模型](https://zh.d2l.ai/chapter_natural-language-processing/word2vec.html#%E8%B7%B3%E5%AD%97%E6%A8%A1%E5%9E%8B)：假设背景词由中心词生成，即建模 $P(w_o\mid w_c)$，其中 $w_c$ 为中心词，$w_o$ 为任一背景词；

![Image Name](https://cdn.kesci.com/upload/image/q5mjsq84o9.png?imageView2/0/w/960/h/960)

2. [CBOW (continuous bag-of-words) 连续词袋模型](https://zh.d2l.ai/chapter_natural-language-processing/word2vec.html#%E8%BF%9E%E7%BB%AD%E8%AF%8D%E8%A2%8B%E6%A8%A1%E5%9E%8B)：假设中心词由背景词生成，即建模 $P(w_c\mid \mathcal{W}_o)$，其中 $\mathcal{W}_o$ 为背景词的集合。

![Image Name](https://cdn.kesci.com/upload/image/q5mjt4r02n.png?imageView2/0/w/960/h/960)

在这里我们主要介绍 Skip-Gram 模型的实现，CBOW 实现与其类似，读者可之后自己尝试实现。后续的内容将大致从以下四个部分展开：

1. PTB 数据集
2. Skip-Gram 跳字模型
3. 负采样近似
4. 训练模型


```python
import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
```

## PTB 数据集

简单来说，Word2Vec 能从语料中学到如何将离散的词映射为连续空间中的向量，并保留其语义上的相似关系。那么为了训练 Word2Vec 模型，我们就需要一个自然语言语料库，模型将从中学习各个单词间的关系，这里我们使用经典的 PTB 语料库进行训练。[PTB (Penn Tree Bank)](https://catalog.ldc.upenn.edu/LDC99T42) 是一个常用的小型语料库，它采样自《华尔街日报》的文章，包括训练集、验证集和测试集。我们将在PTB训练集上训练词嵌入模型。

### 载入数据集

数据集训练文件 `ptb.train.txt` 示例：
```
aer banknote berlitz calloway centrust cluett fromstein gitano guterman ...
pierre  N years old will join the board as a nonexecutive director nov. N 
mr.  is chairman of  n.v. the dutch publishing group 
...
```


```python
with open('/home/kesci/input/ptb_train1020/ptb.train.txt', 'r') as f:
    lines = f.readlines() # 该数据集中句子以换行符为分割
    raw_dataset = [st.split() for st in lines] # st是sentence的缩写，单词以空格为分割
print('# sentences: %d' % len(raw_dataset))

# 对于数据集的前3个句子，打印每个句子的词数和前5个词
# 句尾符为 '' ，生僻词全用 '' 表示，数字则被替换成了 'N'
for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])
```

    # sentences: 42068
    # tokens: 24 ['aer', 'banknote', 'berlitz', 'calloway', 'centrust']
    # tokens: 15 ['pierre', '<unk>', 'N', 'years', 'old']
    # tokens: 11 ['mr.', '<unk>', 'is', 'chairman', 'of']
    

### 建立词语索引


```python
counter = collections.Counter([tk for st in raw_dataset for tk in st]) # tk是token的缩写
counter = dict(filter(lambda x: x[1] >= 5, counter.items())) # 只保留在数据集中至少出现5次的词

idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset] # raw_dataset中的单词在这一步被转换为对应的idx
num_tokens = sum([len(st) for st in dataset])
'# tokens: %d' % num_tokens
```




    '# tokens: 887100'



### 二次采样

文本数据中一般会出现一些高频词，如英文中的“the”“a”和“in”。通常来说，在一个背景窗口中，一个词（如“chip”）和较低频词（如“microprocessor”）同时出现比和较高频词（如“the”）同时出现对训练词嵌入模型更有益。因此，训练词嵌入模型时可以对词进行二次采样。 具体来说，数据集中每个被索引词 $w_i$ 将有一定概率被丢弃，该丢弃概率为


$$

P(w_i)=\max(1-\sqrt{\frac{t}{f(w_i)}},0)

$$


其中  $f(w_i)$  是数据集中词 $w_i$ 的个数与总词数之比，常数 $t$ 是一个超参数（实验中设为 $10^{−4}$）。可见，只有当 $f(w_i)>t$ 时，我们才有可能在二次采样中丢弃词 $w_i$，并且越高频的词被丢弃的概率越大。具体的代码如下：


```python
def discard(idx):
    '''
    @params:
        idx: 单词的下标
    @return: True/False 表示是否丢弃该单词
    '''
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print('# tokens: %d' % sum([len(st) for st in subsampled_dataset]))

def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

print(compare_counts('the'))
print(compare_counts('join'))
```

    # tokens: 375995
    # the: before=50770, after=2161
    # join: before=45, after=45
    

### 提取中心词和背景词


```python
def get_centers_and_contexts(dataset, max_window_size):
    '''
    @params:
        dataset: 数据集为句子的集合，每个句子则为单词的集合，此时单词已经被转换为相应数字下标
        max_window_size: 背景词的词窗大小的最大值
    @return:
        centers: 中心词的集合
        contexts: 背景词窗的集合，与中心词对应，每个背景词窗则为背景词的集合
    '''
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size) # 随机选取背景词窗大小
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

    dataset [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9]]
    center 0 has contexts [1, 2]
    center 1 has contexts [0, 2, 3]
    center 2 has contexts [0, 1, 3, 4]
    center 3 has contexts [2, 4]
    center 4 has contexts [3, 5]
    center 5 has contexts [4, 6]
    center 6 has contexts [5]
    center 7 has contexts [8]
    center 8 has contexts [7, 9]
    center 9 has contexts [7, 8]
    

*注：数据批量读取的实现需要依赖负采样近似的实现，故放于负采样近似部分进行讲解。*

## Skip-Gram 跳字模型

在跳字模型中，每个词被表示成两个 $d$ 维向量，用来计算条件概率。假设这个词在词典中索引为 $i$ ，当它为中心词时向量表示为 $\boldsymbol{v}_i\in\mathbb{R}^d$，而为背景词时向量表示为 $\boldsymbol{u}_i\in\mathbb{R}^d$ 。设中心词 $w_c$ 在词典中索引为 $c$，背景词 $w_o$ 在词典中索引为 $o$，我们假设给定中心词生成背景词的条件概率满足下式：


$$

P(w_o\mid w_c)=\frac{\exp(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{\sum_{i\in\mathcal{V}}\exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}

$$


### PyTorch 预置的 Embedding 层


```python
embed = nn.Embedding(num_embeddings=10, embedding_dim=4)
print(embed.weight)

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
print(embed(x))
```

    Parameter containing:
    tensor([[-0.7417, -1.9469, -0.5745,  1.4267],
            [ 1.1483,  1.4781,  0.3064, -0.2893],
            [ 0.6840,  2.4566, -0.1872, -2.2061],
            [ 0.3386,  1.3820, -0.3142,  0.2427],
            [ 0.4802, -0.6375, -0.4730,  1.2114],
            [ 0.7130, -0.9774,  0.5321,  1.4228],
            [-0.6726, -0.5829, -0.4888, -0.3290],
            [ 0.3152, -0.6827,  0.9950, -0.3326],
            [-1.4651,  1.2344,  1.9976, -1.5962],
            [ 0.0872,  0.0130, -2.1396, -0.6361]], requires_grad=True)
    tensor([[[ 1.1483,  1.4781,  0.3064, -0.2893],
             [ 0.6840,  2.4566, -0.1872, -2.2061],
             [ 0.3386,  1.3820, -0.3142,  0.2427]],
    
            [[ 0.4802, -0.6375, -0.4730,  1.2114],
             [ 0.7130, -0.9774,  0.5321,  1.4228],
             [-0.6726, -0.5829, -0.4888, -0.3290]]], grad_fn=<EmbeddingBackward>)
    

### PyTorch 预置的批量乘法


```python
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)
```

    torch.Size([2, 1, 6])
    

### Skip-Gram 模型的前向计算


```python
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    '''
    @params:
        center: 中心词下标，形状为 (n, 1) 的整数张量
        contexts_and_negatives: 背景词和噪音词下标，形状为 (n, m) 的整数张量
        embed_v: 中心词的 embedding 层
        embed_u: 背景词的 embedding 层
    @return:
        pred: 中心词与背景词（或噪音词）的内积，之后可用于计算概率 p(w_o|w_c)
    '''
    v = embed_v(center) # shape of (n, 1, d)
    u = embed_u(contexts_and_negatives) # shape of (n, m, d)
    pred = torch.bmm(v, u.permute(0, 2, 1)) # bmm((n, 1, d), (n, d, m)) => shape of (n, 1, m)
    return pred
```

## 负采样近似

由于 softmax 运算考虑了背景词可能是词典 $\mathcal{V}$ 中的任一词，对于含几十万或上百万词的较大词典，就可能导致计算的开销过大。我们将以 skip-gram 模型为例，介绍负采样 (negative sampling) 的实现来尝试解决这个问题。

负采样方法用以下公式来近似条件概率 $P(w_o\mid w_c)=\frac{\exp(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{\sum_{i\in\mathcal{V}}\exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}$：


$$

P(w_o\mid w_c)=P(D=1\mid w_c,w_o)\prod_{k=1,w_k\sim P(w)}^K P(D=0\mid w_c,w_k)

$$


其中 $P(D=1\mid w_c,w_o)=\sigma(\boldsymbol{u}_o^\top\boldsymbol{v}_c)$，$\sigma(\cdot)$ 为 sigmoid 函数。对于一对中心词和背景词，我们从词典中随机采样 $K$ 个噪声词（实验中设 $K=5$）。根据 Word2Vec 论文的建议，噪声词采样概率 $P(w)$ 设为 $w$ 词频与总词频之比的 $0.75$ 次方。


```python
def get_negatives(all_contexts, sampling_weights, K):
    '''
    @params:
        all_contexts: [[w_o1, w_o2, ...], [...], ... ]
        sampling_weights: 每个单词的噪声词采样概率
        K: 随机采样个数
    @return:
        all_negatives: [[w_n1, w_n2, ...], [...], ...]
    '''
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)
```


### 批量读取数据


```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives
        
    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)
    
def batchify(data):
    '''
    用作DataLoader的参数collate_fn
    @params:
        data: 长为batch_size的列表，列表中的每个元素都是__getitem__得到的结果
    @outputs:
        batch: 批量化后得到 (centers, contexts_negatives, masks, labels) 元组
            centers: 中心词下标，形状为 (n, 1) 的整数张量
            contexts_negatives: 背景词和噪声词的下标，形状为 (n, m) 的整数张量
            masks: 与补齐相对应的掩码，形状为 (n, m) 的0/1整数张量
            labels: 指示中心词的标签，形状为 (n, m) 的0/1整数张量
    '''
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)] # 使用掩码变量mask来避免填充项对损失函数计算的影响
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
        batch = (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))
    return batch

batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4

dataset = MyDataset(all_centers, all_contexts, all_negatives)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                            collate_fn=batchify, 
                            num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break
```

    centers shape: torch.Size([512, 1])
    contexts_negatives shape: torch.Size([512, 60])
    masks shape: torch.Size([512, 60])
    labels shape: torch.Size([512, 60])
    

## 训练模型

### 损失函数

应用负采样方法后，我们可利用最大似然估计的对数等价形式将损失函数定义为如下


$$

\sum_{t=1}^T\sum_{-m\le j\le m,j\ne 0} [-\log P(D=1\mid w^{(t)},w^{(t+j)})-\sum_{k=1,w_k\sim P(w)^K}\log P(D=0\mid w^{(t)},w_k)]

$$


根据这个损失函数的定义，我们可以直接使用二元交叉熵损失函数进行计算：


```python
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        '''
        @params:
            inputs: 经过sigmoid层后为预测D=1的概率
            targets: 0/1向量，1代表背景词，0代表噪音词
        @return:
            res: 平均到每个label的loss
        '''
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        res = res.sum(dim=1) / mask.float().sum(dim=1)
        return res

loss = SigmoidBinaryCrossEntropyLoss()

pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]]) # 标签变量label中的1和0分别代表背景词和噪声词
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
print(loss(pred, label, mask))

def sigmd(x):
    return - math.log(1 / (1 + math.exp(-x)))
print('%.4f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4)) # 注意1-sigmoid(x) = sigmoid(-x)
print('%.4f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))
```

### 模型初始化


```python
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size))
```

### 训练模型


```python
def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]
            
            pred = skip_gram(center, context_negative, net[0], net[1])
            
            l = loss(pred.view(label.shape), label, mask).mean() # 一个batch的平均loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))

train(net, 0.01, 5)
```

```
train on cpu
epoch 1, loss 0.61, time 221.30s
epoch 2, loss 0.42, time 227.70s
epoch 3, loss 0.38, time 240.50s
epoch 4, loss 0.36, time 253.79s
epoch 5, loss 0.34, time 238.51s
```



### 测试模型


```python
def get_similar_tokens(query_token, k, embed):
    '''
    @params:
        query_token: 给定的词语
        k: 近义词的个数
        embed: 预训练词向量
    '''
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))
        
get_similar_tokens('chip', 3, net[0])
```

```
cosine sim=0.446: intel
cosine sim=0.427: computer
cosine sim=0.427: computers
```


# 2.优化算法进阶 
## Momentum

在 [Section 11.4](https://d2l.ai/chapter_optimization/sgd.html#sec-sgd) 中，我们提到，目标函数有关自变量的梯度代表了目标函数在自变量当前位置下降最快的方向。因此，梯度下降也叫作最陡下降（steepest descent）。在每次迭代中，梯度下降根据自变量当前位置，沿着当前位置的梯度更新自变量。然而，如果自变量的迭代方向仅仅取决于自变量当前位置，这可能会带来一些问题。对于noisy gradient,我们需要谨慎的选取学习率和batch size, 来控制梯度方差和收敛的结果。


$$
\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{g}_{i, t-1}.
$$



## An ill-conditioned Problem

Condition Number of Hessian Matrix:
   
$$
 cond_{H} = \frac{\lambda_{max}}{\lambda_{min}} 
$$
 
where $\lambda_{max}, \lambda_{min}$ is the maximum amd minimum eignvalue of Hessian matrix.

让我们考虑一个输入和输出分别为二维向量$\boldsymbol{x} = [x_1, x_2]^\top$和标量的目标函数:


$$
 f(\boldsymbol{x})=0.1x_1^2+2x_2^2
$$



$$
 cond_{H} = \frac{4}{0.2} = 20 \quad \rightarrow \quad \text{ill-conditioned} 
$$
 

## Maximum Learning Rate
+ For $f(x)$, according to convex optimizaiton conclusions, we need step size $\eta$.
+ To guarantee the convergence, we need to have $\eta$ .

## Supp: Preconditioning

在二阶优化中，我们使用Hessian matrix的逆矩阵(或者pseudo inverse)来左乘梯度向量 $i.e. \Delta_{x} = H^{-1}\mathbf{g}$，这样的做法称为precondition，相当于将 $H$ 映射为一个单位矩阵，拥有分布均匀的Spectrum，也即我们去优化的等价标函数的Hessian matrix为良好的identity matrix。


与[Section 11.4](https://d2l.ai/chapter_optimization/sgd.html#sec-sgd)一节中不同，这里将$x_1^2$系数从$1$减小到了$0.1$。下面实现基于这个目标函数的梯度下降，并演示使用学习率为$0.4$时自变量的迭代轨迹。


```python
%matplotlib inline
import sys
sys.path.append("/home/kesci/input") 
import d2lzh1981 as d2l
import torch

eta = 0.4

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

    epoch 20, x1 -0.943467, x2 -0.000073
    


<img src="https://cdn.kesci.com/rt_upload/14F1B2ECD12E4F64B054F7421F312BF7/q5qnwqtu81.png">


可以看到，同一位置上，目标函数在竖直方向（$x_2$轴方向）比在水平方向（$x_1$轴方向）的斜率的绝对值更大。因此，给定学习率，梯度下降迭代自变量时会使自变量在竖直方向比在水平方向移动幅度更大。那么，我们需要一个较小的学习率从而避免自变量在竖直方向上越过目标函数最优解。然而，这会造成自变量在水平方向上朝最优解移动变慢。

下面我们试着将学习率调得稍大一点，此时自变量在竖直方向不断越过最优解并逐渐发散。

### Solution to ill-condition
+ __Preconditioning gradient vector__: applied in Adam, RMSProp, AdaGrad, Adelta, KFC, Natural gradient and other secord-order optimization algorithms.
+ __Averaging history gradient__: like momentum, which allows larger learning rates to accelerate convergence; applied in Adam, RMSProp, SGD momentum.  


```python
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

    epoch 20, x1 -0.387814, x2 -1673.365109
    


<img src="https://cdn.kesci.com/rt_upload/A293DD7BF95245F998F279897A8A8359/q5qnxcgkw7.png">


## Momentum Algorithm

动量法的提出是为了解决梯度下降的上述问题。设时间步 $t$ 的自变量为 $\boldsymbol{x}_t$，学习率为 $\eta_t$。
在时间步 $t=0$，动量法创建速度变量 $\boldsymbol{m}_0$，并将其元素初始化成 0。在时间步 $t>0$，动量法对每次迭代的步骤做如下修改：


$$
\begin{aligned}
\boldsymbol{m}_t &\leftarrow \beta \boldsymbol{m}_{t-1} + \eta_t \boldsymbol{g}_t, \\
\boldsymbol{x}_t &\leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{m}_t,
\end{aligned}
$$

Another version:

$$
\begin{aligned}
\boldsymbol{m}_t &\leftarrow \beta \boldsymbol{m}_{t-1} + (1-\beta) \boldsymbol{g}_t, \\
\boldsymbol{x}_t &\leftarrow \boldsymbol{x}_{t-1} - \alpha_t \boldsymbol{m}_t,
\end{aligned}
$$

$$
\alpha_t = \frac{\eta_t}{1-\beta} 
$$

其中，动量超参数 $\beta$满足 $0 \leq \beta < 1$。当 $\beta=0$ 时，动量法等价于小批量随机梯度下降。

在解释动量法的数学原理前，让我们先从实验中观察梯度下降在使用动量法后的迭代轨迹。


```python
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + eta * 0.2 * x1
    v2 = beta * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

eta, beta = 0.4, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

    epoch 20, x1 -0.062843, x2 0.001202
    


<img src="https://cdn.kesci.com/rt_upload/337EA9AC35D849FA84C00D0811DF3F33/q5qnz4bzn0.png">


可以看到使用较小的学习率 $\eta=0.4$ 和动量超参数 $\beta=0.5$ 时，动量法在竖直方向上的移动更加平滑，且在水平方向上更快逼近最优解。下面使用较大的学习率 $\eta=0.6$，此时自变量也不再发散。


```python
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

    epoch 20, x1 0.007188, x2 0.002553
    


<img src="https://cdn.kesci.com/rt_upload/B3D7EA65B09743499C1FA28F723B3DF6/q5qnzu2jzs.png">


### Exponential Moving Average

为了从数学上理解动量法，让我们先解释一下指数加权移动平均（exponential moving average）。给定超参数 $0 \leq \beta < 1$，当前时间步 $t$ 的变量 $y_t$ 是上一时间步 $t-1$ 的变量 $y_{t-1}$ 和当前时间步另一变量 $x_t$ 的线性组合：

$$
y_t = \beta y_{t-1} + (1-\beta) x_t.
$$

我们可以对 $y_t$ 展开：

$$
\begin{aligned}
y_t  &= (1-\beta) x_t + \beta y_{t-1}\\
         &= (1-\beta)x_t + (1-\beta) \cdot \beta x_{t-1} + \beta^2y_{t-2}\\
         &= (1-\beta)x_t + (1-\beta) \cdot \beta x_{t-1} + (1-\beta) \cdot \beta^2x_{t-2} + \beta^3y_{t-3}\\
         &= (1-\beta) \sum_{i=0}^{t} \beta^{i}x_{t-i}
\end{aligned}
$$

$$
(1-\beta)\sum_{i=0}^{t} \beta^{i} = \frac{1-\beta^{t}}{1-\beta} (1-\beta) = (1-\beta^{t})
$$

### Supp
Approximate Average of $\frac{1}{1-\beta}$ Steps

令 $n = 1/(1-\beta)$，那么 $\left(1-1/n\right)^n = \beta^{1/(1-\beta)}$。因为

$$
 \lim_{n \rightarrow \infty}  \left(1-\frac{1}{n}\right)^n = \exp(-1) \approx 0.3679,
$$

所以当 $\beta \rightarrow 1$时，$\beta^{1/(1-\beta)}=\exp(-1)$，如 $0.95^{20} \approx \exp(-1)$。如果把 $\exp(-1)$ 当作一个比较小的数，我们可以在近似中忽略所有含 $\beta^{1/(1-\beta)}$ 和比 $\beta^{1/(1-\beta)}$ 更高阶的系数的项。例如，当 $\beta=0.95$ 时，

$$
y_t \approx 0.05 \sum_{i=0}^{19} 0.95^i x_{t-i}.
$$

因此，在实际中，我们常常将 $y_t$ 看作是对最近 $1/(1-\beta)$ 个时间步的 $x_t$ 值的加权平均。例如，当 $\gamma = 0.95$ 时，$y_t$ 可以被看作对最近20个时间步的 $x_t$ 值的加权平均；当 $\beta = 0.9$ 时，$y_t$ 可以看作是对最近10个时间步的 $x_t$ 值的加权平均。而且，离当前时间步 $t$ 越近的 $x_t$ 值获得的权重越大（越接近1）。


### 由指数加权移动平均理解动量法

现在，我们对动量法的速度变量做变形：

$$
\boldsymbol{m}_t \leftarrow \beta \boldsymbol{m}_{t-1} + (1 - \beta) \left(\frac{\eta_t}{1 - \beta} \boldsymbol{g}_t\right). 
$$

Another version:

$$
\boldsymbol{m}_t \leftarrow \beta \boldsymbol{m}_{t-1} + (1 - \beta) \boldsymbol{g}_t. 
$$


$$
\begin{aligned}
\boldsymbol{x}_t &\leftarrow \boldsymbol{x}_{t-1} - \alpha_t \boldsymbol{m}_t,
\end{aligned}
$$


$$
\alpha_t = \frac{\eta_t}{1-\beta} 
$$


由指数加权移动平均的形式可得，速度变量 $\boldsymbol{v}_t$ 实际上对序列 $\{\eta_{t-i}\boldsymbol{g}_{t-i} /(1-\beta):i=0,\ldots,1/(1-\beta)-1\}$ 做了指数加权移动平均。换句话说，相比于小批量随机梯度下降，动量法在每个时间步的自变量更新量近似于将前者对应的最近 $1/(1-\beta)$ 个时间步的更新量做了指数加权移动平均后再除以 $1-\beta$。所以，在动量法中，自变量在各个方向上的移动幅度不仅取决当前梯度，还取决于过去的各个梯度在各个方向上是否一致。在本节之前示例的优化问题中，所有梯度在水平方向上为正（向右），而在竖直方向上时正（向上）时负（向下）。这样，我们就可以使用较大的学习率，从而使自变量向最优解更快移动。


## Implement

相对于小批量随机梯度下降，动量法需要对每一个自变量维护一个同它一样形状的速度变量，且超参数里多了动量超参数。实现中，我们将速度变量用更广义的状态变量`states`表示。


```python
def get_data_ch7():  
    data = np.genfromtxt('/home/kesci/input/airfoil4755/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32)

features, labels = get_data_ch7()

def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data
```

我们先将动量超参数`momentum`设0.5


```python
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.02, 'momentum': 0.5}, features, labels)
```

    loss: 0.243297, 0.057950 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/8B8F9D4CBF2F48B88D446C367E97D587/q5qod8io3b.svg">


将动量超参数`momentum`增大到0.9


```python
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.02, 'momentum': 0.9}, features, labels)
```

    loss: 0.260418, 0.059441 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/2683F607420E443C951B824B2D8BED83/q5qodf7e0m.svg">


可见目标函数值在后期迭代过程中的变化不够平滑。直觉上，10倍小批量梯度比2倍小批量梯度大了5倍，我们可以试着将学习率减小到原来的1/5。此时目标函数值在下降了一段时间后变化更加平滑。


```python
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.004, 'momentum': 0.9}, features, labels)
```

    loss: 0.243650, 0.063532 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/771286646D7A40EEA3D1BDF283C4726A/q5qodjrkb3.svg">


## Pytorch Class

在Pytorch中，```torch.optim.SGD```已实现了Momentum。


```python
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                    features, labels)
```

    loss: 0.243692, 0.048604 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/061E57E1FC1240988C134FC43E749BEE/q5qodoy8py.svg">


# 11.7 AdaGrad

在之前介绍过的优化算法中，目标函数自变量的每一个元素在相同时间步都使用同一个学习率来自我迭代。举个例子，假设目标函数为$f$，自变量为一个二维向量$[x_1, x_2]^\top$，该向量中每一个元素在迭代时都使用相同的学习率。例如，在学习率为$\eta$的梯度下降中，元素$x_1$和$x_2$都使用相同的学习率$\eta$来自我迭代：


$$

x_1 \leftarrow x_1 - \eta \frac{\partial{f}}{\partial{x_1}}, \quad
x_2 \leftarrow x_2 - \eta \frac{\partial{f}}{\partial{x_2}}.

$$


在[“动量法”](./momentum.ipynb)一节里我们看到当$x_1$和$x_2$的梯度值有较大差别时，需要选择足够小的学习率使得自变量在梯度值较大的维度上不发散。但这样会导致自变量在梯度值较小的维度上迭代过慢。动量法依赖指数加权移动平均使得自变量的更新方向更加一致，从而降低发散的可能。本节我们介绍AdaGrad算法，它根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题 [1]。


## Algorithm

AdaGrad算法会使用一个小批量随机梯度$\boldsymbol{g}_t$按元素平方的累加变量$\boldsymbol{s}_t$。在时间步0，AdaGrad将$\boldsymbol{s}_0$中每个元素初始化为0。在时间步$t$，首先将小批量随机梯度$\boldsymbol{g}_t$按元素平方后累加到变量$\boldsymbol{s}_t$：


$$
\boldsymbol{s}_t \leftarrow \boldsymbol{s}_{t-1} + \boldsymbol{g}_t \odot \boldsymbol{g}_t,
$$


其中$\odot$是按元素相乘。接着，我们将目标函数自变量中每个元素的学习率通过按元素运算重新调整一下：


$$
\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \frac{\eta}{\sqrt{\boldsymbol{s}_t + \epsilon}} \odot \boldsymbol{g}_t,
$$


其中$\eta$是学习率，$\epsilon$是为了维持数值稳定性而添加的常数，如$10^{-6}$。这里开方、除法和乘法的运算都是按元素运算的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

## Feature

需要强调的是，小批量随机梯度按元素平方的累加变量$\boldsymbol{s}_t$出现在学习率的分母项中。因此，如果目标函数有关自变量中某个元素的偏导数一直都较大，那么该元素的学习率将下降较快；反之，如果目标函数有关自变量中某个元素的偏导数一直都较小，那么该元素的学习率将下降较慢。然而，由于$\boldsymbol{s}_t$一直在累加按元素平方的梯度，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。所以，当学习率在迭代早期降得较快且当前解依然不佳时，AdaGrad算法在迭代后期由于学习率过小，可能较难找到一个有用的解。

下面我们仍然以目标函数$f(\boldsymbol{x})=0.1x_1^2+2x_2^2$为例观察AdaGrad算法对自变量的迭代轨迹。我们实现AdaGrad算法并使用和上一节实验中相同的学习率0.4。可以看到，自变量的迭代轨迹较平滑。但由于$\boldsymbol{s}_t$的累加效果使学习率不断衰减，自变量在迭代后期的移动幅度较小。


```python
%matplotlib inline
import math
import torch
import sys
sys.path.append("/home/kesci/input") 
import d2lzh1981 as d2l

def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

    epoch 20, x1 -2.382563, x2 -0.158591
    


<img src="https://cdn.kesci.com/rt_upload/65D88109B129448EB6DAC9C0A04110BF/q5qoefd6ox.svg">


下面将学习率增大到2。可以看到自变量更为迅速地逼近了最优解。


```python
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

    epoch 20, x1 -0.002295, x2 -0.000000
    


<img src="https://cdn.kesci.com/rt_upload/90B791EDF32649498EB29AFD2D77302A/q5qoekdeom.svg">


## Implement

同动量法一样，AdaGrad算法需要对每个自变量维护同它一样形状的状态变量。我们根据AdaGrad算法中的公式实现该算法。


```python
def get_data_ch7():  
    data = np.genfromtxt('/home/kesci/input/airfoil4755/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32)
        
features, labels = get_data_ch7()

def init_adagrad_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += (p.grad.data**2)
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)
```

使用更大的学习率来训练模型。


```python
d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
```

    loss: 0.242258, 0.061548 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/FB3ACF978EAE4A158BFCC322169D396C/q5qofl6l7n.svg">


## Pytorch Class

通过名称为“adagrad”的`Trainer`实例，我们便可使用Pytorch提供的AdaGrad算法来训练模型。


```python
d2l.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
```

    loss: 0.243800, 0.060953 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/9ADC04FA976240DE8656E060A4B98F49/q5qofropkx.svg">


# 11.8 RMSProp

我们在[“AdaGrad算法”](adagrad.ipynb)一节中提到，因为调整学习率时分母上的变量$\boldsymbol{s}_t$一直在累加按元素平方的小批量随机梯度，所以目标函数自变量每个元素的学习率在迭代过程中一直在降低（或不变）。因此，当学习率在迭代早期降得较快且当前解依然不佳时，AdaGrad算法在迭代后期由于学习率过小，可能较难找到一个有用的解。为了解决这一问题，RMSProp算法对AdaGrad算法做了修改。该算法源自Coursera上的一门课程，即“机器学习的神经网络”。

## Algorithm

我们在[“动量法”](momentum.ipynb)一节里介绍过指数加权移动平均。不同于AdaGrad算法里状态变量$\boldsymbol{s}_t$是截至时间步$t$所有小批量随机梯度$\boldsymbol{g}_t$按元素平方和，RMSProp算法将这些梯度按元素平方做指数加权移动平均。具体来说，给定超参数$0 \leq \gamma 0$计算


$$
\boldsymbol{v}_t \leftarrow \beta \boldsymbol{v}_{t-1} + (1 - \beta) \boldsymbol{g}_t \odot \boldsymbol{g}_t. 
$$


和AdaGrad算法一样，RMSProp算法将目标函数自变量中每个元素的学习率通过按元素运算重新调整，然后更新自变量


$$
\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \frac{\alpha}{\sqrt{\boldsymbol{v}_t + \epsilon}} \odot \boldsymbol{g}_t, 
$$


其中$\eta$是学习率，$\epsilon$是为了维持数值稳定性而添加的常数，如$10^{-6}$。因为RMSProp算法的状态变量$\boldsymbol{s}_t$是对平方项$\boldsymbol{g}_t \odot \boldsymbol{g}_t$的指数加权移动平均，所以可以看作是最近$1/(1-\beta)$个时间步的小批量随机梯度平方项的加权平均。如此一来，自变量每个元素的学习率在迭代过程中就不再一直降低（或不变）。

照例，让我们先观察RMSProp算法对目标函数$f(\boldsymbol{x})=0.1x_1^2+2x_2^2$中自变量的迭代轨迹。回忆在[“AdaGrad算法”](adagrad.ipynb)一节使用的学习率为0.4的AdaGrad算法，自变量在迭代后期的移动幅度较小。但在同样的学习率下，RMSProp算法可以更快逼近最优解。


```python
%matplotlib inline
import math
import torch
import sys
sys.path.append("/home/kesci/input") 
import d2lzh1981 as d2l

def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = beta * s1 + (1 - beta) * g1 ** 2
    s2 = beta * s2 + (1 - beta) * g2 ** 2
    x1 -= alpha / math.sqrt(s1 + eps) * g1
    x2 -= alpha / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

alpha, beta = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

    epoch 20, x1 -0.010599, x2 0.000000
    


<img src="https://cdn.kesci.com/rt_upload/488520FCB5BC4DFF811770D00333B399/q5qog9m8u0.svg">


## Implement

接下来按照RMSProp算法中的公式实现该算法。


```python
def get_data_ch7():  
    data = np.genfromtxt('/home/kesci/input/airfoil4755/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32)
        
features, labels = get_data_ch7()

def init_rmsprop_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['beta'], 1e-6
    for p, s in zip(params, states):
        s.data = gamma * s.data + (1 - gamma) * (p.grad.data)**2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)
```

我们将初始学习率设为0.01，并将超参数$\gamma$设为0.9。此时，变量$\boldsymbol{s}_t$可看作是最近$1/(1-0.9) = 10$个时间步的平方项$\boldsymbol{g}_t \odot \boldsymbol{g}_t$的加权平均。


```python
d2l.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'beta': 0.9},
              features, labels)
```

    loss: 0.243334, 0.063004 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/5C9361F719B844808D67652F774041F3/q5qogvxs90.svg">


## Pytorch Class

通过名称为“rmsprop”的`Trainer`实例，我们便可使用Gluon提供的RMSProp算法来训练模型。注意，超参数$\gamma$通过`gamma1`指定。


```python
d2l.train_pytorch_ch7(torch.optim.RMSprop, {'lr': 0.01, 'alpha': 0.9},
                    features, labels)
```

    loss: 0.244934, 0.062977 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/B18281B434DC4DAD833ADF5A911D81C2/q5qoh04h4o.svg">


# 11.9 AdaDelta

除了RMSProp算法以外，另一个常用优化算法AdaDelta算法也针对AdaGrad算法在迭代后期可能较难找到有用解的问题做了改进 [1]。有意思的是，AdaDelta算法没有学习率这一超参数。

## Algorithm

AdaDelta算法也像RMSProp算法一样，使用了小批量随机梯度$\boldsymbol{g}_t$按元素平方的指数加权移动平均变量$\boldsymbol{s}_t$。在时间步0，它的所有元素被初始化为0。给定超参数$0 \leq \rho 0$，同RMSProp算法一样计算


$$
\boldsymbol{s}_t \leftarrow \rho \boldsymbol{s}_{t-1} + (1 - \rho) \boldsymbol{g}_t \odot \boldsymbol{g}_t. 
$$


与RMSProp算法不同的是，AdaDelta算法还维护一个额外的状态变量$\Delta\boldsymbol{x}_t$，其元素同样在时间步0时被初始化为0。我们使用$\Delta\boldsymbol{x}_{t-1}$来计算自变量的变化量：


$$
 \boldsymbol{g}_t' \leftarrow \sqrt{\frac{\Delta\boldsymbol{x}_{t-1} + \epsilon}{\boldsymbol{s}_t + \epsilon}}   \odot \boldsymbol{g}_t, 
$$


其中$\epsilon$是为了维持数值稳定性而添加的常数，如$10^{-5}$。接着更新自变量：


$$
\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}'_t. 
$$


最后，我们使用$\Delta\boldsymbol{x}_t$来记录自变量变化量$\boldsymbol{g}'_t$按元素平方的指数加权移动平均：


$$
\Delta\boldsymbol{x}_t \leftarrow \rho \Delta\boldsymbol{x}_{t-1} + (1 - \rho) \boldsymbol{g}'_t \odot \boldsymbol{g}'_t. 
$$


可以看到，如不考虑$\epsilon$的影响，AdaDelta算法与RMSProp算法的不同之处在于使用$\sqrt{\Delta\boldsymbol{x}_{t-1}}$来替代超参数$\eta$。


## Implement

AdaDelta算法需要对每个自变量维护两个状态变量，即$\boldsymbol{s}_t$和$\Delta\boldsymbol{x}_t$。我们按AdaDelta算法中的公式实现该算法。


```python
def init_adadelta_states():
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    delta_w, delta_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * (p.grad.data**2)
        g =  p.grad.data * torch.sqrt((delta + eps) / (s + eps))
        p.data -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```


```python
d2l.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)
```

    loss: 0.243485, 0.084914 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/48D75FB92AAB4D568DD1A3FBA62408EF/q5qohc7hny.svg">


## Pytorch Class

通过名称为“adadelta”的`Trainer`实例，我们便可使用pytorch提供的AdaDelta算法。它的超参数可以通过`rho`来指定。


```python
d2l.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)
```

    loss: 0.267756, 0.061329 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/8E66D8902B3045AAABC2D59F2CFA73A0/q5qohjtwx7.svg">


# 11.10 Adam

Adam算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均 [1]。下面我们来介绍这个算法。

## Algorithm

Adam算法使用了动量变量$\boldsymbol{m}_t$和RMSProp算法中小批量随机梯度按元素平方的指数加权移动平均变量$\boldsymbol{v}_t$，并在时间步0将它们中每个元素初始化为0。给定超参数$0 \leq \beta_1 < 1$（算法作者建议设为0.9），时间步$t$的动量变量$\boldsymbol{m}_t$即小批量随机梯度$\boldsymbol{g}_t$的指数加权移动平均：


$$
\boldsymbol{m}_t \leftarrow \beta_1 \boldsymbol{m}_{t-1} + (1 - \beta_1) \boldsymbol{g}_t. 
$$


和RMSProp算法中一样，给定超参数$0 \leq \beta_2 < 1$（算法作者建议设为0.999），
将小批量随机梯度按元素平方后的项$\boldsymbol{g}_t \odot \boldsymbol{g}_t$做指数加权移动平均得到$\boldsymbol{v}_t$：


$$
\boldsymbol{v}_t \leftarrow \beta_2 \boldsymbol{v}_{t-1} + (1 - \beta_2) \boldsymbol{g}_t \odot \boldsymbol{g}_t. 
$$


由于我们将$\boldsymbol{m}_0$和$\boldsymbol{s}_0$中的元素都初始化为0，
在时间步$t$我们得到$\boldsymbol{m}_t =  (1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} \boldsymbol{g}_i$。将过去各时间步小批量随机梯度的权值相加，得到 $(1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} = 1 - \beta_1^t$。需要注意的是，当$t$较小时，过去各时间步小批量随机梯度权值之和会较小。例如，当$\beta_1 = 0.9$时，$\boldsymbol{m}_1 = 0.1\boldsymbol{g}_1$。为了消除这样的影响，对于任意时间步$t$，我们可以将$\boldsymbol{m}_t$再除以$1 - \beta_1^t$，从而使过去各时间步小批量随机梯度权值之和为1。这也叫作偏差修正。在Adam算法中，我们对变量$\boldsymbol{m}_t$和$\boldsymbol{v}_t$均作偏差修正：


$$
\hat{\boldsymbol{m}}_t \leftarrow \frac{\boldsymbol{m}_t}{1 - \beta_1^t}, 
$$



$$
\hat{\boldsymbol{v}}_t \leftarrow \frac{\boldsymbol{v}_t}{1 - \beta_2^t}. 
$$



接下来，Adam算法使用以上偏差修正后的变量$\hat{\boldsymbol{m}}_t$和$\hat{\boldsymbol{m}}_t$，将模型参数中每个元素的学习率通过按元素运算重新调整：


$$
\boldsymbol{g}_t' \leftarrow \frac{\eta \hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon},
$$


其中$\eta$是学习率，$\epsilon$是为了维持数值稳定性而添加的常数，如$10^{-8}$。和AdaGrad算法、RMSProp算法以及AdaDelta算法一样，目标函数自变量中每个元素都分别拥有自己的学习率。最后，使用$\boldsymbol{g}_t'$迭代自变量：


$$
\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}_t'. 
$$


## Implement

我们按照Adam算法中的公式实现该算法。其中时间步$t$通过`hyperparams`参数传入`adam`函数。


```python
%matplotlib inline
import torch
import sys
sys.path.append("/home/kesci/input") 
import d2lzh1981 as d2l

def get_data_ch7():  
    data = np.genfromtxt('/home/kesci/input/airfoil4755/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32)
        
features, labels = get_data_ch7()

def init_adam_states():
    v_w, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad.data
        s[:] = beta2 * s + (1 - beta2) * p.grad.data**2
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```


```python
d2l.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)
```

    loss: 0.242722, 0.089254 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/46DA5110F99A4BB58180C0D38497C943/q5qoij5h08.svg">


## Pytorch Class


```python
d2l.train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)
```

    loss: 0.242389, 0.073228 sec per epoch
    


<img src="https://cdn.kesci.com/rt_upload/8E491F60A4FB4C2990C60CF86E00BAF1/q5qoio531k.svg">


# 3.模型微调
在前面的一些章节中，我们介绍了如何在只有6万张图像的Fashion-MNIST训练数据集上训练模型。我们还描述了学术界当下使用最广泛的大规模图像数据集ImageNet，它有超过1,000万的图像和1,000类的物体。然而，我们平常接触到数据集的规模通常在这两者之间。

假设我们想从图像中识别出不同种类的椅子，然后将购买链接推荐给用户。一种可能的方法是先找出100种常见的椅子，为每种椅子拍摄1,000张不同角度的图像，然后在收集到的图像数据集上训练一个分类模型。这个椅子数据集虽然可能比Fashion-MNIST数据集要庞大，但样本数仍然不及ImageNet数据集中样本数的十分之一。这可能会导致适用于ImageNet数据集的复杂模型在这个椅子数据集上过拟合。同时，因为数据量有限，最终训练得到的模型的精度也可能达不到实用的要求。

为了应对上述问题，一个显而易见的解决办法是收集更多的数据。然而，收集和标注数据会花费大量的时间和资金。例如，为了收集ImageNet数据集，研究人员花费了数百万美元的研究经费。虽然目前的数据采集成本已降低了不少，但其成本仍然不可忽略。

另外一种解决办法是应用迁移学习（transfer learning），将从源数据集学到的知识迁移到目标数据集上。例如，虽然ImageNet数据集的图像大多跟椅子无关，但在该数据集上训练的模型可以抽取较通用的图像特征，从而能够帮助识别边缘、纹理、形状和物体组成等。这些类似的特征对于识别椅子也可能同样有效。

本节我们介绍迁移学习中的一种常用技术：微调（fine tuning）。如图9.1所示，微调由以下4步构成。

1. 在源数据集（如ImageNet数据集）上预训练一个神经网络模型，即源模型。
2. 创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
3. 为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。
4. 在目标数据集（如椅子数据集）上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。


![Image Name](https://cdn.kesci.com/upload/image/q5u3kayjap.png?imageView2/0/w/640/h/640)


当目标数据集远小于源数据集时，微调有助于提升模型的泛化能力。
## 9.2.1 热狗识别
接下来我们来实践一个具体的例子：热狗识别。我们将基于一个小数据集对在ImageNet数据集上训练好的ResNet模型进行微调。该小数据集含有数千张包含热狗和不包含热狗的图像。我们将使用微调得到的模型来识别一张图像中是否包含热狗。

首先，导入实验所需的包或模块。torchvision的[`models`](https://pytorch.org/docs/stable/torchvision/models.html)包提供了常用的预训练模型。如果希望获取更多的预训练模型，可以使用使用[`pretrained-models.pytorch`](https://github.com/Cadene/pretrained-models.pytorch)仓库。


```python
%matplotlib inline
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys

sys.path.append("/home/kesci/input/")
import d2lzh1981 as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 9.2.1.1 获取数据集

我们使用的热狗数据集（[点击下载](https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip)）是从网上抓取的，它含有1400张包含热狗的正类图像，和同样多包含其他食品的负类图像。各类的1000张图像被用于训练，其余则用于测试。

我们首先将压缩后的数据集下载到路径`data_dir`之下，然后在该路径将下载好的数据集解压，得到两个文件夹`hotdog/train`和`hotdog/test`。这两个文件夹下面均有`hotdog`和`not-hotdog`两个类别文件夹，每个类别文件夹里面是图像文件。


```python
import os
os.listdir('/home/kesci/input/resnet185352')
```




    ['resnet18-5c106cde.pth']




```python
data_dir = '/home/kesci/input/hotdog4014'
os.listdir(os.path.join(data_dir, "hotdog"))
```




    ['test', 'train']



我们创建两个`ImageFolder`实例来分别读取训练数据集和测试数据集中的所有图像文件。


```python
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))
```

下面画出前8张正类图像和最后8张负类图像。可以看到，它们的大小和高宽比各不相同。


```python
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```


<img src="https://cdn.kesci.com/rt_upload/12977C25072E48678693B46BE55C66E6/q5ubute3bm.png">


在训练时，我们先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为高和宽均为224像素的输入。测试时，我们将图像的高和宽均缩放为256像素，然后从中裁剪出高和宽均为224像素的中心区域作为输入。此外，我们对RGB（红、绿、蓝）三个颜色通道的数值做标准化：每个数值减去该通道所有数值的平均值，再除以该通道所有数值的标准差作为输出。
> 注: 在使用预训练模型时，一定要和预训练时作同样的预处理。
如果你使用的是`torchvision`的`models`，那就要求:
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].


```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])
```

### 定义和初始化模型
我们使用在ImageNet数据集上预训练的ResNet-18作为源模型。这里指定`pretrained=True`来自动下载并加载预训练的模型参数。在第一次使用时需要联网下载模型参数。


```python
pretrained_net = models.resnet18(pretrained=False)
pretrained_net.load_state_dict(torch.load('/home/kesci/input/resnet185352/resnet18-5c106cde.pth'))
```




    <All keys matched successfully>



下面打印源模型的成员变量`fc`。作为一个全连接层，它将ResNet最终的全局平均池化层输出变换成ImageNet数据集上1000类的输出。


```python
print(pretrained_net.fc)
```

    Linear(in_features=512, out_features=1000, bias=True)
    

> 注: 如果你使用的是其他模型，那可能没有成员变量`fc`（比如models中的VGG预训练模型），所以正确做法是查看对应模型源码中其定义部分，这样既不会出错也能加深我们对模型的理解。[`pretrained-models.pytorch`](https://github.com/Cadene/pretrained-models.pytorch)仓库貌似统一了接口，但是我还是建议使用时查看一下对应模型的源码。

可见此时`pretrained_net`最后的输出个数等于目标数据集的类别数1000。所以我们应该将最后的`fc`成修改我们需要的输出类别数:


```python
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)
```

    Linear(in_features=512, out_features=2, bias=True)
    

此时，`pretrained_net`的`fc`层就被随机初始化了，但是其他层依然保存着预训练得到的参数。由于是在很大的ImageNet数据集上预训练的，所以参数已经足够好，因此一般只需使用较小的学习率来微调这些参数，而`fc`中的随机初始化参数一般需要更大的学习率从头训练。PyTorch可以方便的对模型的不同部分设置不同的学习参数，我们在下面代码中将`fc`的学习率设为已经预训练过的部分的10倍。


```python
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)
```

###  微调模型


```python
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)
```


```python
train_fine_tuning(pretrained_net, optimizer)
```

    training on  cpu
    epoch 1, loss 3.4516, train acc 0.687, test acc 0.884, time 298.2 sec
    epoch 2, loss 0.1550, train acc 0.924, test acc 0.895, time 296.2 sec
    epoch 3, loss 0.1028, train acc 0.903, test acc 0.950, time 295.0 sec
    epoch 4, loss 0.0495, train acc 0.931, test acc 0.897, time 294.0 sec
    epoch 5, loss 0.1454, train acc 0.878, test acc 0.939, time 291.0 sec
    

作为对比，我们定义一个相同的模型，但将它的所有模型参数都初始化为随机值。由于整个模型都需要从头训练，我们可以使用较大的学习率。


```python
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)
```

    training on  cpu
    epoch 1, loss 2.6391, train acc 0.598, test acc 0.734, time 292.4 sec
    epoch 2, loss 0.2703, train acc 0.790, test acc 0.632, time 289.7 sec
    epoch 3, loss 0.1584, train acc 0.810, test acc 0.825, time 290.2 sec
    epoch 4, loss 0.1177, train acc 0.805, test acc 0.787, time 288.6 sec
    epoch 5, loss 0.0782, train acc 0.829, test acc 0.828, time 289.8 sec
    

输出：
```
training on  cuda
epoch 1, loss 2.6686, train acc 0.582, test acc 0.556, time 25.3 sec
epoch 2, loss 0.2434, train acc 0.797, test acc 0.776, time 25.3 sec
epoch 3, loss 0.1251, train acc 0.845, test acc 0.802, time 24.9 sec
epoch 4, loss 0.0958, train acc 0.833, test acc 0.810, time 25.0 sec
epoch 5, loss 0.0757, train acc 0.836, test acc 0.780, time 24.9 sec
```

