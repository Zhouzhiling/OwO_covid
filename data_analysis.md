# data_analysis

## Data Update

In order to get the latest updated data at any time, run:

```
git pull upstream master
git add .
git commit -m 
git push origin master
```



## Data Analysis

### ./data/us/air_quality

空气质量数据，关联不大。



### ./data/us/covid

第一次repo用到的数据：

[confirmed_cases.csv](https://github.com/Zhouzhiling/OwO_covid/blob/master/data/us/covid/confirmed_cases.csv)

[daily_state_tests.csv](https://github.com/Zhouzhiling/OwO_covid/blob/master/data/us/covid/daily_state_tests.csv)



### deaths.csv_4.2

是累加数据还是当日数据？

> max = 499
>
> 截止4.2，无死亡人数的county数目是2563，county总数是3195
>
> 已出现死亡的county大多符合指数分布

<img src="./img/cumulative death.png" alt="image-20200403141242425" style="zoom:50%;" />





### ./data/us/demographics

一些州的面积，人口，贫穷程度，失业率



### ./data/us/flu

过去5年流感的分布，感染人数，死亡数



### ./data/us/frozen

重复数据



### ./data/us/geolocation

county之间的地理位置信息，county的中心点等。



### ./data/us/hospital

医院数目，床位数量，icu床位数



### ./data/us/mobility

社交距离的有效程度衡量。



### ./data/us/other

航班限制政策

政策开始执行的时间

county之间的连接频率



### ./data/us/processing_data

FIPS和county名称的对应关系



### ./data/us/respiratory_disease





## Questions

sample_submission里面的county数量是3223个。deaths.csv中的county个数是3195(包括了Statewide Unallocated)。