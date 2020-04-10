# 4.2

数据初步分析，初版模型框架为K Means + RNN



# 4.3

deaths.csv数据分析，结果见data_analysis.md

简单可视化，结果见img文件夹。

全数据K Means实现（两类）

Non-zero K Means实现（三类）

检查death数目和指数分布的拟合情况。

RNN实现。



#### **疑问**：

sample_submission里面的county数量是3223个。deaths.csv中的county个数是3195(包括了Statewide Unallocated)。



## TODO:

add confirmed case

比较函数拟合的结果和rnn的结果

# 4.4

不同county在4.3当天的死亡率方差较大。

同一个county在时间线上的死亡率方差也较大。

```
# of death / # of confirmed:
std of county of > 50 death is 0.014409632565737372
mean of county of > 50 death is 0.029978383880098266

std of all county is 0.1377636853574628
mean of all county is 0.08411215325002097
```



## TODO

加入医疗，医疗数目，county贫困程度等数据，输入RNN。

SIR

# 4.5

SIR模型跑通了！



## TODO

查资料，[parameters](https://github.com/ryansmcgee/seirsplus)估计

之后可以考虑用机器学习预测参数

graph的方法之后或许可以考虑尝试



# 4.6

### DONE

generate different models for different counties



### TODO

加入confirmed_case作为initN。

对于预测的death count，加入分布，生成可提交的文件。



