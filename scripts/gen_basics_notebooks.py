#!/usr/bin/env python3
"""Batch-generate all 9 basics notebooks."""

import nbformat as nbf

MODELS = [
    {
        "name": "logistic_regression",
        "title": "Logistic Regression",
        "desc": "Single Linear layer + Softmax for MNIST digit classification (92.3%).",
        "formula": r"""$$P(y=c \mid x) = \frac{e^{w_c \cdot x + b_c}}{\sum_{j=1}^{10} e^{w_j \cdot x + b_j}}$$

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log P(y_i \mid x_i)$$""",
        "import_code": "from basics.logistic_regression import train",
        "run_code": "train()",
        "questions": [
            "Logistic Regression 和 Linear Regression 的区别是什么？（输出 vs 损失函数）",
            "为什么用 Softmax + CrossEntropy 而不是 MSE 做分类？",
            "把学习率从 0.1 改到 0.01，准确率会怎么变？试试。",
        ],
    },
    {
        "name": "linear_regression",
        "title": "Linear Regression",
        "desc": "Normal Equation + Gradient Descent on California Housing (R²=0.583).",
        "formula": r"""$$\theta = (X^T X)^{-1} X^T y \quad \text{(Normal Equation)}$$

$$\theta \leftarrow \theta - \alpha \cdot \frac{2}{m} X^T (X\theta - y) \quad \text{(Gradient Descent)}$$""",
        "import_code": "from basics.linear_regression import train",
        "run_code": "train()",
        "questions": [
            "Normal Equation 和 Gradient Descent 的优缺点？",
            "特征标准化为什么对 GD 重要而对 Normal Equation 不重要？",
            "把学习率从 0.1 改到 1.0，GD 还会收敛吗？",
        ],
    },
    {
        "name": "svm",
        "title": "Support Vector Machine",
        "desc": "Primal GD + Dual SMO with Linear and RBF kernels (MNIST 93.3%).",
        "formula": r"""**Primal (hinge loss + L2):**

$$\min \frac{1}{n} \sum \max(0, 1 - y_i (w \cdot x_i + b)) + \lambda \|w\|^2$$

**Dual (kernel trick):**

$$\max \sum \alpha_i - \frac{1}{2} \sum \sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$""",
        "import_code": "from basics.svm import main",
        "run_code": "main()",
        "questions": [
            "RBF kernel 为什么能处理非线性可分数据？",
            "SMO 为什么选择两个 α 同时优化而不是一个？",
            "参数 C 越大，模型更偏向于什么（大间隔还是少错误）？",
        ],
    },
    {
        "name": "perceptron",
        "title": "Perceptron",
        "desc": "Single neuron with step activation (Rosenblatt 1958).",
        "formula": r"""$$y = \text{sign}(w \cdot x + b)$$

**更新规则（误分类时）：**

$$w \leftarrow w + \eta \cdot y \cdot x$$

$$b \leftarrow b + \eta \cdot y$$""",
        "import_code": "from basics.perceptron import demo",
        "run_code": "demo()",
        "questions": [
            "Perceptron 为什么只能解决线性可分问题？",
            "Perceptron 和 Logistic Regression 的核心区别是什么？",
            "Perceptron Convergence Theorem 保证什么？",
        ],
    },
    {
        "name": "k_means",
        "title": "K-Means",
        "desc": "Unsupervised clustering on MNIST, 57.8% cluster purity.",
        "formula": r"""**E-step（分配）：**

$$c_i = \arg\min_k \|x_i - \mu_k\|^2$$

**M-step（更新）：**

$$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$$""",
        "import_code": "from basics.k_means import train",
        "run_code": "train()",
        "questions": [
            "K-Means 一定能收敛吗？收敛到全局最优吗？",
            "k 值怎么选？（提示：肘部法则）",
            "为什么 K-Means 对初始中心点敏感？K-Means++ 怎么改进？",
        ],
    },
    {
        "name": "decision_tree",
        "title": "Decision Tree",
        "desc": "ID3/CART on Iris dataset (93.3%, ASCII tree).",
        "formula": r"""**熵（impurity 度量）：**

$$H(S) = -\sum p_i \log_2 p_i$$

**信息增益：**

$$IG = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$$""",
        "import_code": "from basics.decision_tree import demo",
        "run_code": "demo()",
        "questions": [
            "决策树在 Iris 上只用了哪两个特征？为什么？（提示：print_tree 观察）",
            "max_depth 太小会欠拟合，太大会过拟合，怎么选？",
            "信息增益和基尼系数（Gini impurity）有什么区别？",
        ],
    },
    {
        "name": "naive_bayes",
        "title": "Naive Bayes",
        "desc": "Gaussian Naive Bayes on MNIST (53.0%) — shows independence assumption gap.",
        "formula": r"""**贝叶斯定理 + 特征独立假设：**

$$P(y \mid x) \propto P(y) \prod_{i=1}^d P(x_i \mid y)$$

**高斯似然：**

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{iy}^2}} \exp\left(-\frac{(x_i - \mu_{iy})^2}{2\sigma_{iy}^2}\right)$$""",
        "import_code": "from basics.naive_bayes import demo",
        "run_code": "demo()",
        "questions": [
            "为什么 Logistic Regression（92.3%）远好于 Naive Bayes（53.0%）？",
            "像素之间真的独立吗？相邻像素的关系是怎样的？",
            "如果特征满足独立假设，Naive Bayes 是最优分类器吗？",
        ],
    },
    {
        "name": "pca",
        "title": "PCA",
        "desc": "SVD-based dimensionality reduction, MNIST 2D visualisation.",
        "formula": r"""**中心化：** $\tilde{X} = X - \bar{x}$

**SVD：** $\tilde{X} = U \Sigma V^T$

**投影到前 k 个主成分：** $X_{\text{proj}} = \tilde{X} \cdot V_{:,:k}$""",
        "import_code": "from basics.pca import demo",
        "run_code": "demo()",
        "questions": [
            "PCA 的第一主成分捕获了什么？（在 MNIST 上观察 ASCII 图）",
            "为什么用 SVD 而不是直接做协方差矩阵的特征分解？",
            "保留多少主成分能保留 90% 的方差？",
        ],
    },
    {
        "name": "knn",
        "title": "k-NN",
        "desc": "k-Nearest Neighbors on MNIST — instance-based, no training.",
        "formula": r"""**欧氏距离：**

$$d(x, y) = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$$

**预测（多数投票）：**

$$\hat{y} = \text{majority vote of } k \text{ nearest neighbors}$$""",
        "import_code": "from basics.knn import demo",
        "run_code": "demo()",
        "questions": [
            "k=1 和 k=10 的区别是什么？（偏差-方差权衡）",
            "为什么 k-NN 在 784 维空间表现有限？（维度灾难）",
            'k-NN 为什么是"懒惰学习"？训练阶段做了什么？',
        ],
    },
]


def md(cells, s):
    cells.append(nbf.v4.new_markdown_cell(s))

def code(cells, s):
    cells.append(nbf.v4.new_code_cell(s))


for m in MODELS:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    }
    cells = []

    # Title
    md(cells, f"# {m['title']}\n\n{m['desc']}")

    # Background
    md(cells, f"""## 背景

{m['desc']} 本 notebook 演示 {m['title']} 的完整实现，模型代码见 `basics/{m['name']}.py`。
""")

    # Math
    md(cells, f"""## 数学原理

{m['formula']}
""")

    # Import + run
    code(cells, m['import_code'])
    code(cells, m['run_code'])

    # Questions
    qs = "\n".join(f"{i+1}. {q}" for i, q in enumerate(m['questions']))
    md(cells, f"""## 思考题

{qs}
""")

    nb.cells = cells
    path = f"basics/{m['name']}.ipynb"
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"Generated {path}")
