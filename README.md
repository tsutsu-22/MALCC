# RGB-LIME
LIMEという予測結果の解釈を画像分類のRGBチャンネルに拡張したデモになります。

cat.jpg(元画像)

![ori](test/2000/ori/test_output.png)

↑をtiger catと推定するときの重要領域と色を可視化することができます。

![important area](test/2000/merge/0.005.png)

右目の緑(G)を分類根拠として使っているようです。

# 使い方

以下を実行するだけです。
```
python main.py
```

パッケージは以下でインストールできます。
tensorflow-gpuはmain.pyを実行するだけならただのtensorflowでも大丈夫かもしれません。
```
pip install -r requirements.txt
```

## param
以下の内容を変更することが可能です。
```python
  ####param####
  num_pattern=2000  #重回帰分析のパターン数
  img='cat.jpg' #入力画像名
  svdir='test'
  savename='test_output'
  th=0.005 #上位何％を表示するか
  #############
```
num_pattern...マスク画像を何パターン作るか、多いほど正確になるが、計算時間が増える

img...入力画像のパス、ImageNetで分類したいものを同ディレクトリに置いてここに記述

svdir...結果をどのディレクトリに保存するか

savename...結果の画像名

th...重要領域と色を上位何％まで表示するか


### 参考
大まかな流れは↓を参考にして、画像も拝借しました。

https://ascii.jp/elem/000/004/007/4007762/

LIMEのgit↓

https://github.com/marcotcr/lime/tree/73f03130b1fa8dbb3378457e78c82d4889942f83


