
---

# Reproduction of LAM (StQP) Model — Cesarone et al. (2011)

This repository is an **independent reproduction** of the model and algorithms presented in the following paper:

> **Cesarone, Francesco, Andrea Scozzari, and Fabio Tardella.
> *"Portfolio selection problems in practice: a comparison between linear and quadratic optimization models."*
> arXiv preprint arXiv:1105.3594 (2011).**

The purpose of this repository is solely **academic and educational**: to understand and reimplement the LAM (Linearized Active-Set Method) / StQP-based formulation described in the paper and to provide reference code for research study.

⚠️ **Disclaimer**

* 本リポジトリは、上記論文の著者・出版社とは**一切関係ありません**。
* 本リポジトリの内容・コードは論文の非公式な再現実装であり、
  **正確性・完全性・再現性を保証するものではありません**。
* 本実装に起因するいかなる損害・損失に対しても、作成者は**責任を負いません**。
* 研究・学習目的でのみご利用ください。

---

# I. データおよびモデルパラメータの要件

実装には、以下の金融データおよびユーザー定義のモデルパラメータが必要です。

| 要件                    | 定義 / 説明                                               |
| --------------------- | ----------------------------------------------------- |
| **資産数 ($n$)**         | ポートフォリオに含まれうる総資産数                                   |
| **期待収益率ベクトル ($\mu$)** | 各資産 $i$ の期待収益率 $\mu_i$ のベクトル                         |
| **共分散行列 ($\Sigma$)**  | 各資産 $i$ と $j$ の共分散 $\sigma_{ij}$ を含む $n \times n$ 行列 |
| **目標収益率 ($\rho$)**    | ポートフォリオが達成すべき目標期待収益率                                 |
| **カーディナリティ制約 ($K$)**  | ポートフォリオに含めることができる最大資産数                               |
| **数量制約（下限 $\ell_i$）** | 投資比率 $x_i > 0$ である場合の最小投資比率 $\ell_i$                 |
| **数量制約（上限 $u_i$）**    | 各資産の最大投資比率 $u_i$。                                     |
| **ペナルティ係数 ($M$)**     | 目標収益率制約を目的関数に組み込むための大きな正の係数               |

---

# II. 定式化の要件（目的関数と制約）

LAM モデルを StQP として実装するためには、元の MIQP の等式制約をペナルティ項として目的関数に組み込み、残りの制約をカーディナリティ制約付き StQP の形式で記述する必要があります。

## 1. 目的関数（二次ペナルティの導入）

```math
\text{Min } f_M(x) =
\sum_{i=1}^n \sum_{j=1}^n \sigma_{ij} x_i x_j
+
M \left( \sum_{i=1}^n \mu_i x_i - \rho \right)^2
```

この目的関数は
```math
f_M(x) = x^T Q_M x
```
の形に変換でき、$Q_M$ は $\Sigma$ にペナルティ項が加わった修正行列です。

---

## 2. 制約条件

* **標準単体制約**：
  $\sum x_i = 1$, $x_i \ge 0$

* **カーディナリティ制約**：
  $|\mathrm{supp}(x)| \le K$

* **数量制約**：
  $\ell_i \le x_i \le u_i$

---

# III. 解法アルゴリズムの要件（本実装での扱い）

本リポジトリでは、Cesarone et al. (2011) の LAM（StQP）モデルを完全に厳密に再現するのではなく、**Increasing Set Algorithm (IS)** の構造を参考にした **Forward Selection 型の近似アルゴリズム**によって、カーディナリティ制約付き StQP を解く実装を提供する。

---

## 1. 本実装でのアルゴリズム（Forward Selection / Greedy IS）

論文で提案されている IS（Increasing Set Algorithm）は以下の処理を含む複雑な手法だが、本リポジトリのコードではその構造を簡略化し、以下の Greedy Forward Selection ベースの手法を用いている。

### **Step 1 — k = 1 から K まで資産数を増やしながら探索**

* 現在の資産集合 ( S_{k-1} ) に対し、
  追加候補となる資産を 1 つずつ試し、
  各候補について次の QP を解いて目的関数値を比較する：

```math
\min_x \; x^\top \Sigma x + M (x^\top \mu - \rho)^2
```

* その中で最も目的関数が小さくなる資産を追加し、
  ( S_k = S_{k-1} \cup {i^*} ) とする。

---

### **Step 2 — 部分集合に対する QP（StQP）の解法**

各候補資産を追加した部分集合 ( I ) に対して：

* 部分ベクトル ( \mu_I )
* 部分共分散行列 ( \Sigma_I )

を抽出し、
SciPy の SLSQP により **数量制約付き QP** を解く：

制約：

```math
\sum_{i \in I} x_i = 1, \quad
\ell \le x_i \le u
```

目的関数：

```math
x^\top \Sigma_I x + M (x^\top \mu_I - \rho)^2
```

---

### **Step 3 — 最良解の更新**

* k 個の資産で得た最良の QP 解が過去の最良値より良ければ更新
* これを k=1〜K の間で繰り返し、**最良の部分集合**とその **最適ウェイトベクトル**を返す

---

## 2. Increasing Set Algorithm（論文版）との関係

論文で提案されている Increasing Set Algorithm（IS）は、
以下の手順を含む厳密な StQP 解法：

* 部分行列の正定値性判定
* relative interior 最適性の検証
* feasible / infeasible サブセットの分類
* ( C''_K ) に対する追加の凸 QP 解法

本実装は **IS の完全な再現ではなく**、
IS の「資産集合を増やしながら探索する」構造のみを踏襲した
**Forward Selection（貪欲法）による近似的な解法**である。

---

## 3. 使用ライブラリ（本実装）

本リポジトリのコードに必要なのは以下のライブラリ：

### **線形代数・数値計算**

* **NumPy**
* **Pandas**

### **部分集合ごとの QP（SLSQP）**

* **SciPy Optimize**

※ 論文の IS の完全版実装に必要となる
「正定値性判定」「半正定値計画」「外部 QP ソルバ（Gurobi/MOSEK/CVXPY）」
などは **本コードでは使用していない**。
