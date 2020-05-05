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



# 4.16

### 今日done：

用DTW找到相似的county，用geometry的数据补充相似性。

找到多个类的county之后，对每个/每类county做平稳性检验。

如果不平稳，做差分使得平稳；然后套模型（LR，Lasso，Ridge）



### TODO：

正常输出结果





# 4.20

跑通了linearregression和lasso，输出了结果并上传了。

### TODO

解决很多预测值为0的问题（过拟合？

调参



# 4.21

✅DTW里的threshold，分类的个数（4-5类？）

✅了解一下平稳性检验的置信度看是不是要调

✅regression中的惩罚项

✅ridge alpha=20， 有负数

✅window的长度



# 5.3

## Done: 

✅run dnn

当前label长度为14：

没有删除全零的时候，共10万条数据

删除全零的label之后，还剩3万条数据👈当前采用。



## Related File: 

preprocessForNN.py + DNN.py



## TODO

✅pinball loss

❌add more feature to train DNN

❌特征工程（归一化ect）

❌smooth the input by rolling mean

❌modify special counties (New York etc.) by hand

# 5.4

用于evaluation的数据来源是nyt_us_counties_daily.csv，和当前用的deahts.csv的不同在于

> nyt_us_counties_daily是每日新增的数据，deahts是累计到当日的数据
>
> 数据来源不同导致每日新增的death数目也不完全相同。

于是在preprocessForNN里面加了一个transform_format函数，用于生成./processed_data/daily_death_from_nyu.csv和daily_confirmed_from_nyu.csv数据。

数据格式和之前的deaths.csv/confirmed_cases.csv相同，数据来源是nyt_us_counties_daily，保存的是每日的新增数据。

之后每次更新完数据生成两个文件就可。

---

修改了DNN模型结构，复杂++。

把每个county的policy加到DNN的feature中。

