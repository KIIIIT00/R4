<div id="top"></div>
目次

1. [プロジェクトについて](#1プロジェクトについて)
2. [ディレクトリ構造](#2ディレクトリ構造)
3. [環境構築](#3環境構築)

## 1.プロジェクトについて
This repository is a weed detection program used in the R4 project, a project under development at AISLAB, Ritsumeikan University.
It is not intended for use by third parties.

## 2.ディレクトリ構造
プロジェクトのディレクトリ構造は以下の通りです
:
```
.
├─datasets # CNNの訓練用データが格納されるフォルダ（クラスごと）
│  ├─train
│  │  ├─little_weed
│  │  ├─many_weed
│  │  └─no_weed
│  └─val
│      ├─little_weed
│      ├─many_weed
│      └─no_weed
├─flash_imgs # 元の画像
│  ├─little_weed
│  ├─many_weed
│  └─no_weed
├─figures # CNNの損失，正解率のグラフなどを格納
├─models # 学習済みモデルや結果を格納
├─utils # 使いまわすロジック
├─.gitignore
└─README.md
```

## 3.環境構築
#### 環境構築

```
pip install -r requirements.txt
```