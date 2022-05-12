# Anacondaで仮想環境を作成
**投稿日：2021年12月01日<br>最終更新日：2021年12月01日**

仮想環境を作成する機会は、さほどありません。そのため、私は仮想環境を作成するたびに調べています（すぐ忘れる）。繰り返して調べるのが面倒なので、簡単にまとめておきます。

Anacondaで仮想環境を作成するには、以下のコマンドを実行する。pythonのバージョンは3.8としている。同時にnumpy, pandas, matplotlibのパッケージをインストールしている。
```bash
conda create --name myenv python=3.8 numpy pandas matplotlib
```

作成した仮想環境を以下のコマンドで確認する。
```bash
conda info --envs
# 以下のコマンドは上記の省略形式
conda info -e
```
仮想環境を有効にするのは、以下のコマンド。
```bash
conda activate myenv
```
仮想環境を無効にするのは、以下のコマンド。
```bash
conda deactivate
```
仮想環境の削除は以下のコマンド。
```bash
conda remove --name myenv --all
```

