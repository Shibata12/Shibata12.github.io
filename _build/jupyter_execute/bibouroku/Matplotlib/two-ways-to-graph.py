#!/usr/bin/env python
# coding: utf-8

# # 二通りのグラフ作成方法
# **投稿日：2022年6月22日<br>最終更新日：2022年6月22日**
# 
# matplotlibでグラフを作成するには，MATLAB形式のAPIとオブジェクト指向APIの二通りの方法がある．

# **実行環境**<br>
# このノートブックはjupyter lab上で実行している．pythonのバージョンは以下の通り．

# In[1]:


get_ipython().system('python --version')


# **ライブラリ**<br>
# 使用する各ライブラリのバージョンは以下の通り．

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

print(f"Matplotlib version {mpl.__version__}")
print(f"Numpy version      {np.__version__}")


# ## MATLAB形式のAPI
# MATLABでグラフを作成する時と同じようにmatplotlibを使用できる．以下のコードはグラフを縦に二つ並べるときの書き方である．これを実現するには`plt.subplot()`を使用する．

# In[3]:


x = np.linspace(0, 10, 100)

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))

plt.show()


# ## オブジェクト指向API
# 上記のグラフと同じものをオブジェクト指向APIで作成するには，`plt.subplots()`で生成される`Figure`オブジェクトと`Axes`オブジェクトを使用する．
# ```{note}
# Matplotlibの公式ドキュメントによると（[A note on the Object-Oriented API vs. Pyplot](https://matplotlib.org/stable/tutorials/introductory/lifecycle.html#a-note-on-the-object-oriented-api-vs-pyplot)），オブジェクト指向APIを使用することが推奨されている．
# ```

# In[4]:


fig, ax = plt.subplots(2)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

plt.show()


# **参考文献**<br>
# * [Pythonデータサイエンスハンドブック Jake VanderPlas 著, 菊池 彰 訳, オライリー・ジャパン, ISBN978-4-87311-841-3](https://www.oreilly.co.jp/books/9784873118413/)
