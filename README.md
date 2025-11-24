
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

# III. 解法アルゴリズムの要件

LAM（StQP）モデルの解法には、本論文で提案された **Increasing Set Algorithm (IS)** を実装する必要があります。

## 1. Increasing Set Algorithm (IS)

* $j = 1 \dots K$ に対して資産組合せを探索
* 各部分集合 $I$ について

  * 部分行列 $Q_I$ の正定値性チェック
  * 相対内部での最適性チェック
* 数量制約を満たす集合 $C'_j$ と満たさない集合 $C''_j$ を区別
* $C''_K$ に対しては追加の凸二次計画（QP）を解く

---

## 2. 必要なライブラリ

* **NumPy / SciPy**（線形代数）
* **CVXPY / Gurobi / MOSEK / SciPy Optimize**
  （最終ステージの QP のため）

