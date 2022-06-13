#!/usr/bin/env python
# coding: utf-8

# # 不均衡データセットのベースラインモデル評価
# 
# **投稿日：2022年6月13日<br>最終更新日：2022年6月13日**
# 
# [Imbalanced-learn](https://imbalanced-learn.org/stable/)に実装されている[27の不均衡なデータセット](https://imbalanced-learn.org/stable/datasets/index.html)を使用する。
# ここでは、以下のことを行う。
# * 27のデータセットの読み込み
# * LightGBMによるベースラインモデルの評価
# 
# **実行環境**<br>
# このnotebookはGoogle Colaboratoryで実行されている。

# In[1]:


get_ipython().system('python --version')


# **ライブラリ**<br>
# 使用しているライブラリは以下の4つ。

# In[2]:


import imblearn
import sklearn
import pandas as pd
import lightgbm as lgb

print(f"Imbalanced-learn version {imblearn.__version__}")
print(f"Scikit-learn version     {sklearn.__version__}")
print(f"Pandas version           {pd.__version__}")
print(f"LightGBM version         {lgb.__version__}")


# ## データセットの読み込み
# Imbalanced-learnのベンチマークデータセットを使用する。
# 
# Imbalanced-learnのインストール<br>
# **conda**
# ```bash
# conda install -c conda-forge imbalanced-learn
# ```
# 
# 27のデータセットは[`imblearn.datasets.fetch_datasets()`](https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html#imblearn.datasets.fetch_datasets)で読み込む。
# データセットは[Imbalanced dataset for benchmarking: zenodo](https://zenodo.org/record/61452#.YqLgplXP1Gg)からダウンロードされる。

# In[3]:


from imblearn.datasets import fetch_datasets

datasets = fetch_datasets()
datasets


# データセットの情報は下記の表にまとめてある。
# Nameはデータセットの名前であり、個々のデータセットにアクセスするときに使用する。
# Repositoryはデータセットの配布場所である。
# 中には多クラス分類問題のデータセットが含まれているが、二クラス分類問題となるように加工されている。
# Targetは少数クラスのカテゴリを指している。
# Ratioは多数クラスと少数クラス間のサンプル数の比である。
# これは、不均衡の度合いを示す。
# \#Sはサンプル数であり、\#Fは特徴数である。
# カテゴリ変数がone-hot表現に変換されていることに注意する。
# 
# |ID|Name          | Repository & Target           | Ratio | #S      | #F  |
# |--|--------------|-------------------------------|------:|--------:|----:|
# |1 |ecoli         | UCI, target: imU              | 8.6:1 | 336     | 7   |
# |2 |optical_digits| UCI, target: 8                | 9.1:1 | 5,620   | 64  |
# |3 |satimage      | UCI, target: 4                | 9.3:1 | 6,435   | 36  |
# |4 |pen_digits    | UCI, target: 5                | 9.4:1 | 10,992  | 16  |
# |5 |abalone       | UCI, target: 7                | 9.7:1 | 4,177   | 10  |
# |6 |sick_euthyroid| UCI, target: sick euthyroid   | 9.8:1 | 3,163   | 42  |
# |7 |spectrometer  | UCI, target: >=44             | 11:1  | 531     | 93  |
# |8 |car_eval_34   | UCI, target: good, v good     | 12:1  | 1,728   | 21  |
# |9 |isolet        | UCI, target: A, B             | 12:1  | 7,797   | 617 |
# |10|us_crime      | UCI, target: >0.65            | 12:1  | 1,994   | 100 |
# |11|yeast_ml8     | LIBSVM, target: 8             | 13:1  | 2,417   | 103 |
# |12|scene         | LIBSVM, target: >one label    | 13:1  | 2,407   | 294 |
# |13|libras_move   | UCI, target: 1                | 14:1  | 360     | 90  |
# |14|thyroid_sick  | UCI, target: sick             | 15:1  | 3,772   | 52  |
# |15|coil_2000     | KDD, CoIL, target: minority   | 16:1  | 9,822   | 85  |
# |16|arrhythmia    | UCI, target: 06               | 17:1  | 452     | 278 |
# |17|solar_flare_m0| UCI, target: M->0             | 19:1  | 1,389   | 32  |
# |18|oil           | UCI, target: minority         | 22:1  | 937     | 49  |
# |19|car_eval_4    | UCI, target: vgood            | 26:1  | 1,728   | 21  |
# |20|wine_quality  | UCI, wine, target: <=4        | 26:1  | 4,898   | 11  |
# |21|letter_img    | UCI, target: Z                | 26:1  | 20,000  | 16  |
# |22|yeast_me2     | UCI, target: ME2              | 28:1  | 1,484   | 8   |
# |23|webpage       | LIBSVM, w7a, target: minority | 33:1  | 34,780  | 300 |
# |24|ozone_level   | UCI, ozone, data              | 34:1  | 2,536   | 72  |
# |25|mammography   | UCI, target: minority         | 42:1  | 11,183  | 6   |
# |26|protein_homo  | KDD CUP 2004, minority        | 111:1 | 145,751 | 74  |
# |27|abalone_19    | UCI, target: 19               | 130:1 | 4,177   | 10  |

# ## ベースラインモデル
# ベースラインモデルには[LightGBM](https://lightgbm.readthedocs.io/en/latest/)を使用する。
# 
# LightGBMのインストール<br>
# **conda**
# ```bash
# conda install -c conda-forge lightgbm
# ```
# 
# データセットを読み込んだ時点で最低限の前処理が施されているため、前処理を行わずともLightGBMを訓練することが可能である。
# 評価指標には正解率、F1値、適合率、再現率を用いて、層化10分割交差検証を行う。
# これを`sklearn.model_selection.cross_validata()`で実装する。
# 実行時に`UndefinedMetricWarning`が出る場合は、おそらく陽性を検出できていないことを示している。
# つまり、すべての予測が陰性となっていることを示している。
# そのため、適合率、再現率、F1値は0になっている。

# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_validate

dataset_scores = {}
scoring = ["accuracy", "f1", "precision", "recall"]
for dataset_name in datasets:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_validate(
        lgb.LGBMClassifier(),
        datasets[dataset_name].data,
        datasets[dataset_name].target,
        scoring=scoring,
        cv=kfold
    )
    dataset_scores[dataset_name] = scores


# 他の評価指標を検討したい場合は、まずは`sklearn.metrics.SCORERS`を参照するとよい。

# In[5]:


sklearn.metrics.SCORERS


# 今回は、この中の`accuracy`, `f1`, `precision`, `recall`を選択した。
# 
# `dataset_scores`には評価結果だけでなくモデルの訓練時間も含まれる。例えば、ecoliデータセットの交差検証の結果を確認する。

# In[6]:


dataset_scores["ecoli"]


# 各フォールドの訓練時間は`"fit_time"`でアクセスできる。

# In[7]:


print(f'各フォールドの訓練時間\n{dataset_scores["ecoli"]["fit_time"]}')
print(f'モデル訓練時間の平均\n{dataset_scores["ecoli"]["fit_time"].mean():.4f}秒')


# `dataset_scores`をpandasのデータフレームに変換して見やすくする。

# In[8]:


scores_dict = {}
for dataset_name in datasets:
    scores_dict[dataset_name] =         ["{:.4f} ± {:.4f}".format(
            dataset_scores[dataset_name]["test_"+score_name].mean(),
            dataset_scores[dataset_name]["test_"+score_name].std()
        ) for score_name in scoring]

pd.DataFrame(scores_dict, index=scoring).T


# データセットの中には、陽性を全く検出できていないものがある。
# 今回はyeast_ml8, abalone_19の2つのデータセットが、陽性を全く検出できていない。
# yeast_ml8, abalone_19はF1値、適合率、再現率が0である。
# 
# 一方、陽性と陰性の分類が完璧なデータセットがある。
# car_eval_4は陽性と陰性を全て正しく分類している。
# 正解率、F1値、適合率、再現率のすべてが1.0である。
