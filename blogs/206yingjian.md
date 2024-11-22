---
layout: page
permalink: /blogs/206yingjian/index.html
title: J.P.M Quantitative Project
---

## J.P.M Quantitative 

---

### 🔥Task1 

>选取五只股票进行有效前沿的绘制以及给定预期收益的最佳持仓配比求解


### 🍴 具体任务实现的算法

1.有效前沿的绘制：*sco*

2.最佳持仓配比求解：*cvxopt/sco*

<br>有效前沿作为代表了一组风险和预期回报之间的最佳组合，这些组合在给定的风险水平下提供了最高的预期回报，或者在给定的预期回报下具有最低的风险。

<center>
<img src = "/blogs/206yingjian.assets/有效前沿2.png">
</center>

   
---
### 基础数据及其特征展示

````python    
```
    pip install cvxopt
    pip install yfinance #下载股票数据
````


````python
```
    import numpy as np
    import pandas as pd  
    import matplotlib.pyplot as plt  
    from cvxopt import matrix, solvers
    import yfinance as yf 

    # 获取股票数据  
    tickers = ['GOOG', 'MSFT', 'AAPL', 'NVDA', 'AMZN']  
    data = yf.download(tickers, start='2024-01-01', end='2024-10-01')['Adj Close']

    data.shift(1) #创造一行空行用于计算收益率

    #将股票按照初始交易日进行归一化处理并可视化
    (data/data.iloc[0]).plot(figsize=(10,6),grid=True)

````
<center>
<img src = "/blogs/206yingjian.assets/Return Rate.png">
</center>

---
### 算法实现

#### 投资组合可行解
````python
```
    #随机进行2000次模拟  
    #Rp_list,Vp_list分别存储每个模拟投资的预期收益率和波动率
    n=5   #五只股票五个投资组合
    I=2000
    Rp_list=np.ones(I)  #预期收益率
    Vp_list=np.ones(I)  #波动率
    SR_list=np.ones(I)  #夏普比率

    #模拟过程
    for i in np.arange(I):
       x=np.random.rand(n)  #生成n个随机权重
       weights=x/sum(x)     #权重归一化，使其和为1
       Rp_list[i]=np.sum(weights*Manual_LR)  
       Vp_list[i]=np.sqrt(np.dot(weights,np.dot
(Cov_LR,weights.T)))  
       SR_list[i]=Rp_list[i]/Vp_list[i]

    #展示结果    
    plt.figure(figsize=(8, 4), dpi=100, facecolor='white')  
    plt.scatter(Vp_list, Rp_list)  
    plt.title('The relationship between expected portfolio return and volatility', pad=20)  
    plt.xlabel('Volatility', labelpad=20)  
    plt.ylabel('Expected Return', labelpad=20)  
    plt.grid()

````

<center>
<img src = "/blogs/206yingjian.assets/The relationship between expected portfolio return and volatility.png">
</center>

---
#### 投资组合的有效前沿
````python
```
    import scipy.optimize as sco

   #必要参数构建
    def f(w):
       w=np.array(w)
       Rp_opt=np.sum(w*Manual_LR)
       Vp_opt=np.sqrt(np.dot(w,np.dot(Cov_LR,w.T)))
       return np.array([Rp_opt,Vp_opt])

    def Vmin_f(w):
       return f(w)[1]

    cons=({'type':'eq','fun':lambda x:np.sum(x)-1},{'type':'eq','fun':lambda x:f(x)[0]-0.15})
    bnds=((0,1),(0,1),(0,1),(0,1),(0,1))

    #权重决定重要性
    w0=np.array([0.2, 0.2, 0.2, 0.2, 0.2])  
    result=sco.minimize(fun=Vmin_f,x0=w0,method='SLSQP',bounds=bnds,
    constraints=cons)
    cons_vmin=({'type':'eq','fun':lambda x:np.sum(x)-1})
    result_vmin=sco.minimize(fun=Vmin_f,x0=w0,
    method='SLSQP',bounds=bnds,constraints=cons_vmin)

    Vp_vmin=result_vmin['fun']
    Rp_vmin=np.sum(Manual_LR*result_vmin['x'])
 
    Rp_target=np.linspace(Rp_vmin,0.95,300)
    Vp_target=[]

    for r in Rp_target:
      cons_new=({'type':'eq','fun':lambda x:np.sum(x)-1},{'type':'eq','fun':lambda x:f(x)[0]-r})
      result_new=sco.minimize(fun=Vmin_f,x0=w0,
      method='SLSQP',bounds=bnds,constraints=cons_new)
       Vp_target.append(result_new['fun'])

    #展示结果    
     plt.figure(figsize=(8, 4))  
     plt.scatter(Vp_list,Rp_list,c=SR_list, cmap='YlGnBu', marker='o')  
     plt.colorbar(label='Sharpe Ratio')  
     plt.plot(Vp_target,Rp_target,'r-',label="efficient frontier")
     plt.scatter(sdp, rp, marker='*', color='r', s=300, label='Max Sharpe Ratio')  
     plt.title('Efficient Frontier')  
     plt.xlabel('Annualized Volatility')  
     plt.ylabel('Annualized Return')  
     plt.legend(labelspacing=0.8)  
     plt.show()
````
<center>
<img src = "/blogs/206yingjian.assets/Efficient Frontier.png">
</center>

---
#### 必要参数设置
> 股价收益率不能处理对称处理上涨和下跌，增加50%和减少50%的影响不会相互抵消,多期收益计算容易产生累积误差，适合分析不连续性的收益事件，如分红和其他一次性收益。<br>对数收益率：假设市场是连续复利的，对数收益率更反映真实收益，上下波动对称，且多期收益可以简单相加而不产生累积误差，更适合正态分布假设,长期投资和复杂的金融模型适合使用对数收益率。

````python
```
    #计算股票的对数收益率并且展示描述性统计指标
    Log_return=np.log(data/data.shift(1))
    #计算股票的年平均收益率  通过计算该序列的算术平均值的到平均对数收益率
    Manual_LR=Log_return.mean()*252
    #计算股票收益率的年化波动率  计算平均波动率后年化
    Vol_LR=Log_return.std()*np.sqrt(252)
    #计算股票的协方差矩阵并进行年化处理
    Cov_LR=Log_return.cov()*252
    #计算股票的相关系数矩阵
    Corr_LR=Log_return.corr()

````
---
#### 资本市场线
````python
``` 
    Rf=0.03  #无风险利率
    def F(w):
       w=np.array(w)
       Rp_opt=np.sum(w*Manual_LR)
       Vp_opt=np.sqrt(np.dot(w,np.dot(Cov_LR,w.T)))
       Slope=(Rp_opt-Rf)/Vp_opt
       return np.array([Rp_opt,Vp_opt,Slope])

    def Slope_F(w):
          return -F(w)[-1]

    w1=np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cons_Slope=({'type':'eq','fun':lambda x:np.sum(x)-1})
    result_slope=sco.minimize(fun=Slope_F,x0=w1,method='SLSQP',bounds=bnds,
constraints=cons_Slope)

    Rm=np.sum(Manual_LR*wm)
    Vm=(Rm-Rf)/Slope
    Rp_CML=np.linspace(Rf,0.95,200)
    Vp_CML=(Rp_CML-Rf)/Slope

    #展示结果    
    plt.figure(figsize=(8, 4))  
    plt.scatter(Vp_list,Rp_list,c=SR_list, cmap='YlGnBu', marker='o')  
    plt.colorbar(label='Sharpe Ratio')  
    plt.plot(Vp_target,Rp_target,'r-',label="efficient frontier")
    plt.plot(Vp_vmin,Rp_vmin,'g*',label='Global minimum volatility',markersize=13)
    plt.plot(Vp_CML,Rp_CML,'b--',label='market portfolio',markersize=13)
    plt.scatter(sdp, rp, marker='*', color='r', s=300, label='Max Sharpe Ratio')  
    plt.title('Efficient Frontier')  
    plt.xlabel('Annualized Volatility')  
    plt.ylabel('Annualized Return')  
    plt.legend(labelspacing=0.8)  
    plt.show()
````
<center>
<img src = "/blogs/206yingjian.assets/Total_Efficient Frontier.png">
</center>

---
#### 基于cvxopt的资产组合配置

````python
```
   #计算每个股票日收益率的百分比变化并且移除掉有缺失值的行  
   returns = data.pct_change().dropna()  
 
   cov_matrix = returns.cov() * 252  # 年化协方差矩阵  
   print(cov_matrix)  

   #选用cvxopt，作为凸优化问题的工具的默认参数已经比较高效，如果想进一步优化性能，可以
   #通过优化矩阵表示（如利用系数矩阵）和调节求解器参数（如迭代次数、容忍度）来改善性能

   #投资组合优化  
   def optimize_portfolio(cov_matrix, mean_returns, target_return):  
       n = len(mean_returns)  
       P = matrix(cov_matrix.values)  
       q = matrix(np.zeros((n, 1)))  
       G = matrix(np.vstack((-np.array(mean_returns), -np.identity(n))))  
       h = matrix(np.hstack((-target_return, np.zeros(n))))  
       A = matrix(1.0, (1, n))  
       b = matrix(1.0)  

       solvers.options['show_progress'] = False  
       sol = solvers.qp(P, q, G, h, A, b)  

       return sol['x']  

   # 目标返回  
   target_return = 0.1  # 10% 的目标年化收益率  
   mean_returns = returns.mean() * 252  

   optimal_weights = optimize_portfolio(cov_matrix, mean_returns, target_return)  
   print("\nOptimal Portfolio Weights:")  
   print([round(w, 4) for w in optimal_weights])  



---

### 🔥Task2

>基于Black-Litterman的最佳持仓配比求解

#### 背景
> Markowitz模型输入参数包括历史数据法和情景分析法两种方法,情景分析法的缺点是主观因素,随意性太强,因此使用历史数据法, 将资产的均值和协方差输入模型是比较常见的作法. 不过, 不足之处很明显: 未来的资产收益率分布不一定与过去相同. 此外, Markowitz 模型结果对输入参数过于敏感。<br>Black-Litterman模型就是基于此的改进. 其核心思想是将投资者对大类资产的观点 (主观观点) 与市场均衡收益率 (先验预期收益率)相结合,从而形成新的预期收益率(后验预期收益率). 这里的先验预期收益率的分布可以是贝叶斯推断中的先验概率密度函数的多元正态分布形式,投资者的主观观点就是贝叶斯推断中的似然函数(可以看作新的信息, 因为做出主观判断必然是从外界获取得到了这些资产的收益率变化信息), 而相应的, 后验预期收益率也可以从后验概率密度函数中得到<br>核心思想是将投资者对大类资产的观点与市场均衡收益率相结合,从而形成新的预期收益率，是贝叶斯推断中的先验概率密度函数的多元正态分布形式,投资者的主观观点是似然函数, 后验预期收益率从后验概率密度函数中得到。


### 🍴 具体任务实现的算法


````



---



>持续记录中



