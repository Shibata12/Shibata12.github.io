# Jupyter Book の更新方法
**投稿日：2021年11月22日<br>最終更新日：2021年11月22日**

2021年11月5日にこのサイトを公開して2週間ほど経過しました。驚くことにサイトの更新方法を忘れていました。今回はJupyter Bookの更新方法を自分用にまとめておきます。

**ローカル環境でJupyter Bookを更新する**<br>
gitでクローンしたフォルダに移動してJupyter Bookを更新する。以下のコマンドで実行できる。
```bash
cd Shibata12.github.io
jupyter-book build --all .
```

**gitでadd, commit, push**<br>
Jupyter Bookの更新をしたら、以下のコマンドを実行した後にpushする。
```bash
git checkout master
ghp-import -n -p -f _build/html
git add .
git commit -m "comment"
git push
```
これでサイトが更新される。

**所感**<br>
よく分かんないけど更新できた。

**参考**<br>
https://jupyterbook.org/start/publish.html
https://qiita.com/magolors/items/620860558661b527f267