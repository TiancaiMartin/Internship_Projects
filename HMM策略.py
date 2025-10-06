import pandas as pd
import numpy as np
import empyrical as ep
from hmmlearn.hmm import GMMHMM,GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

#行情数据
sec_post_1d_df = pd.read_pickle(r'/Users/tiancaixiaohuoban/Desktop/实习/买方实习/中信建投期货（金融工程）/策略设计/dom_cont_post_1d')

# test
df = sec_post_1d_df.loc['A'].T
close = df['close']
high = df['high']
low = df['low']
volume = df['volume']

datelist = pd.to_datetime(df.index)
logreturn = (np.log(np.array(close[1:])) - np.log(np.array(close[:-1])))[4:]
logreturn5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
differentreturn = (np.log(np.array(high)) - np.log(np.array(low)))
closeidx = close[1:]

X = np.column_stack([logreturn, differentreturn, logreturn5])
# 使用tqdm显示进度
hmm = GaussianHMM(n_components=6, covariance_type='diag', n_iter=5000).fit(X)
latent_states_sequence = hmm.predict(X)

sns.set_style('white')
plt.figure(figsize = (15,8))
for i in range(hmm.n_components):
    state=(latent_states_sequence == i)
    plt.plot(datelist[state],closeidx[state],'.',label ='latent state %d'%i,lw = 1)
    plt.legend()
    plt.grid(1)

# 根据隐马尔科夫模型的状态，计算每个状态的收益率
