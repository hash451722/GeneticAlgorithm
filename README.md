# 遺伝的アルゴリズム (Genetic Algorithm)

## 世代交代モデル
JGG (Just Generation Gap)
 - 個体集団から親個体を評価値とは無関係にランダムに非復元抽出する
 - 親個体郡に交叉を繰り返し適用し子個体を生成する
 - 子個体郡から評価値が上位の個体を親個体の数だけ選択する
 - 選択した個体を個体集団に追加し、次世代の個体集団とする

 - 個体集団の個体数は、15n～50nを推奨(nは次元数)
 - 生成する子個体数は、10nを推奨(nは次元数)


## 交叉　
SPX (Simplex crossover)
 - 親個体数は (入力変数の次元 + 1個)
 - 重心 親個体ベクトルを ε 倍した点により張られる単体(simplex)の内部に子個体をランダム生成
 - 変数間依存性や悪スケール性に強い
 - nを次元数としたときに、n+1個の親個体を頂点とするn次元単体(simplex)に相似な単体内に一様分布に従って子個体を生成する交叉である。
 - 変数間依存性に対処することができる交叉である。
 - パラメータの拡張率εの推奨値は ε=sqrt(n+2) である。
 - 樋口等，人工知能学会論文誌，Vol.16, No.1, pp.147-155 (2001)


## 用語
 - 遺伝的アルゴリズム genetic algorithm
 - 個体集団 population
 - 個体 individual
 - 適応度 fitness
 - 選択 selection
 - 交叉 crossover
 - 突然変異 mulation
 - 世代 generation
 - 子孫 offspring


## 参考
[Test Problems in Optimization](https://arxiv.org/abs/1008.0549)