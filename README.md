# twStock notebook

------


## 目的
1. 擷取台股即時資料分析
2. 分析個股定期不定額
3. **注意台股TWSE 有request limit** _(每5秒鐘3個request,過的話會被封鎖ip至少20分鐘以上)_
------

#  瑞耘
* 四大買賣點分析


```python
import twstock
from twstock import Stock
from twstock import BestFourPoint
import pandas as pd
import numpy as np

print(twstock.codes['6532'].name)
print(twstock.codes['6532'].start)
print(twstock.codes['6532'])   

stock = Stock('6532')               # 擷取股價
bfp = BestFourPoint(stock)
print('\n判斷是否為四大 買點')
print(bfp.best_four_point_to_buy())    # 判斷是否為四大買點
print('\n判斷是否為四大 賣點')
print(bfp.best_four_point_to_sell())   # 判斷是否為四大賣點
print('\n綜合判斷')
print(bfp.best_four_point())           # 綜合判斷

ma_p = stock.moving_average(stock.price, 5)       # 計算五日均價
ma_c = stock.moving_average(stock.capacity, 5)    # 計算五日均量
ma_p_cont = stock.continuous(ma_p)                # 計算五日均價持續天數
ma_br = stock.ma_bias_ratio(5, 10)                # 計算五日、十日乖離值
d = {'ma_p': ma_p, 'ma_c': ma_c}
df = pd.DataFrame(data=d)
print('\n計算五日均價持續天數:')
print(ma_p_cont)
print('\n計算 ma_p(五日均價) ma_c(五日均量):')
print(df)

twstock.realtime.get('6532')    # 擷取當前股票資訊
twstock.realtime.get(['6532'])  # 擷取當前三檔資訊
```

    瑞耘
    2016/09/26
    StockCodeInfo(type='股票', code='6532', name='瑞耘', ISIN='TW0006532005', start='2016/09/26', market='上櫃', group='半導體業', CFI='ESVUFR')
    
    判斷是否為四大 買點
    量大收紅, 三日均價大於六日均價
    
    判斷是否為四大 賣點
    False
    
    綜合判斷
    (True, '量大收紅, 三日均價大於六日均價')
    
    計算五日均價持續天數:
    26
    
    計算 ma_p(五日均價) ma_c(五日均量):
         ma_p       ma_c
    0   30.00   887800.0
    1   30.62   905000.0
    2   31.55   823400.0
    3   31.93   763600.0
    4   32.03   501000.0
    5   32.14   427400.0
    6   32.99   423200.0
    7   33.94   781400.0
    8   34.83   858000.0
    9   35.88   927400.0
    10  36.76   967600.0
    11  37.09  1033200.0
    12  37.27   715400.0
    13  37.68   879400.0
    14  38.03  1172600.0
    15  38.44  1193200.0
    16  38.54  1200200.0
    17  38.86  1421200.0
    18  39.24  1312200.0
    19  39.47  1018800.0
    20  40.33  1263600.0
    21  41.91  1732800.0
    22  43.27  1805400.0
    23  44.42  1852400.0
    24  45.83  2184800.0
    25  46.94  2297200.0
    26  48.52  2758000.0
    




    {'6532': {'timestamp': 1588833000.0,
      'info': {'code': '6532',
       'channel': '6532.tw',
       'name': '瑞耘',
       'fullname': '瑞耘科技股份有限公司',
       'time': '2020-05-07 14:30:00'},
      'realtime': {'latest_trade_price': '-',
       'trade_volume': '-',
       'accumulate_trade_volume': '3422',
       'best_bid_price': ['59.2000', '59.1000', '59.0000', '58.9000', '58.8000'],
       'best_bid_volume': ['637', '2', '3', '7', '4'],
       'best_ask_price': ['-'],
       'best_ask_volume': ['-'],
       'open': '56.8000',
       'high': '59.2000',
       'low': '55.8000'},
      'success': True},
     'success': True}



## 基本操作
* 匯入twstock library:


```python
from twstock import Stock

stock = Stock('2892')                             # 擷取第一金股價
ma_p = stock.moving_average(stock.price, 5)       # 計算五日均價
ma_c = stock.moving_average(stock.capacity, 5)    # 計算五日均量
ma_p_cont = stock.continuous(ma_p)                # 計算五日均價持續天數
ma_br = stock.ma_bias_ratio(5, 10)                # 計算五日、十日乖離值
```


```python
ma_p_cont
```


```python
import pandas as pd
import numpy as np
d = {'ma_p': ma_p, 'ma_c': ma_c}
df = pd.DataFrame(data=d)
df
```


```python
import pandas as pd
import numpy as np

d = {'ma_br': ma_br}
df = pd.DataFrame(data=d)
df
```

* 擷取自 2015 年 1 月至今之資料


```python
stock.fetch_from(2015, 1)
```

* 基本資料之使用:


```python
stock.price
```


```python
stock.capacity
```


```python
stock.data[0]
```

------

## 附件：Juypter Notebook 基本操作 

### Jupyter Notebook 基本操作介紹影片:


```python
%%HTML
<iframe width="560" height="315" src="https://www.youtube.com/embed/HW29067qVWk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

------
### Juypter % 及 %% 外掛程式運用:


```python
%lsmagic
```

### 圖表基本操作 (matplot library):


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
```

### 計算時間:


```python
%%timeit
square_evens = [n*n for n in range(1000)]
```

### 資料呈現(panda):

* 解決ImportError: cannot import name 'nosetester'問題:
  * numpy 1.11.1 version
    * pip3 uninstall numpy
    * pip3 install numpy==1.11.1
  * pandas 0.19.2 version 
    * pip3 uninstall pandas
    * pip3 install pandas==0.19.2


```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,5))
df.head()

```
