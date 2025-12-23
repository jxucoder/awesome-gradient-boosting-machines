# Awesome Gradient Boosting Machines

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/jxucoder/awesome-gradient-boosting-machines/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A curated list of gradient boosting frameworks, research papers, tutorials, and resources for machine learning practitioners and researchers.

<p align="center">
  <img width="600" src="https://raw.githubusercontent.com/benedekrozemberczki/awesome-gradient-boosting-papers/master/boosting.gif" alt="Gradient Boosting">
</p>

Gradient boosting is one of the most powerful techniques for building predictive models. It builds an ensemble of weak prediction models (typically decision trees) in a stage-wise fashion and generalizes them by allowing optimization of an arbitrary differentiable loss function.

---

## Contents

- [Implementations](#implementations)
  - [XGBoost](#xgboost)
  - [LightGBM](#lightgbm)
  - [CatBoost](#catboost)
  - [Other Frameworks](#other-frameworks)
- [Research Papers](#research-papers)
  - [2025](#2025)
  - [2024](#2024)
  - [2023](#2023)
  - [2022](#2022)
  - [2021](#2021)
  - [2020](#2020)
  - [2019](#2019)
  - [Foundational Papers](#foundational-papers)
- [Tutorials & Guides](#tutorials--guides)
- [Interpretability & Explainability](#interpretability--explainability)
- [Blog Posts](#blog-posts)
- [Videos & Talks](#videos--talks)
- [Books](#books)
- [Benchmarks & Comparisons](#benchmarks--comparisons)
- [Real-World Applications](#real-world-applications)
- [Related Awesome Lists](#related-awesome-lists)
- [Contributing](#contributing)

---

## Implementations

### XGBoost

- [XGBoost](https://github.com/dmlc/xgboost) - Scalable, portable, and distributed gradient boosting library. ⭐ 26k+
  - **v2.0+ (2023)**: Improved memory usage, vector-leaf tree methods, and PySpark support.
- [XGBoost Documentation](https://xgboost.readthedocs.io/) - Official documentation with tutorials and API reference.
- [XGBoost Python Package](https://pypi.org/project/xgboost/) - Python wrapper for XGBoost.
- [XGBoost4J-Spark](https://xgboost.readthedocs.io/en/stable/jvm/xgboost4j_spark_tutorial.html) - XGBoost integration with Apache Spark.

### LightGBM

- [LightGBM](https://github.com/microsoft/LightGBM) - A fast, distributed, high-performance gradient boosting framework by Microsoft. ⭐ 17k+
  - **v4.0+ (2023)**: Quantized training support, improved GPU performance, and new objectives.
- [LightGBM Documentation](https://lightgbm.readthedocs.io/) - Official documentation and tutorials.
- [LightGBM Python Package](https://pypi.org/project/lightgbm/) - Python wrapper for LightGBM.

### CatBoost

- [CatBoost](https://github.com/catboost/catboost) - A fast, scalable, high-performance gradient boosting on decision trees library by Yandex. ⭐ 8k+
- [CatBoost Documentation](https://catboost.ai/docs/) - Official documentation with tutorials.
- [CatBoost Python Package](https://pypi.org/project/catboost/) - Python wrapper for CatBoost.
- [CatBoost Tutorials](https://github.com/catboost/tutorials) - Official tutorials and examples.

### Other Frameworks

- [NGBoost](https://github.com/stanfordmlgroup/ngboost) - Natural gradient boosting for probabilistic prediction by Stanford ML Group.
- [PGBM](https://github.com/elephaint/pgbm) - Probabilistic Gradient Boosting Machines with native GPU acceleration, auto-differentiation, and uncertainty estimates. Built on PyTorch/Numba.
- [GBNet](https://github.com/mthorrell/gbnet) - Integrates XGBoost/LightGBM with PyTorch for auto-differentiation of custom loss functions and hybrid neural network + GBM models. [[JOSS Paper](https://joss.theoj.org/papers/10.21105/joss.08047)]
- [Perpetual](https://github.com/perpetual-ml/perpetual) - Hyperparameter-free gradient boosting that self-generalizes. Just set a `budget` parameter instead of tuning hyperparameters. Written in Rust with Python bindings.
- [GBDT-PL](https://github.com/GBDT-PL/GBDT-PL) - Gradient Boosting with Piece-Wise Linear Regression Trees. Accelerates convergence and optimized for SIMD parallelism. (Now available in LightGBM via `linear_tree=true`)
- [SGTB](https://github.com/bloomberg/sgtb) - Structured Gradient Tree Boosting for collective entity disambiguation by Bloomberg.
- [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) - GPU-accelerated gradient boosting decision tree library.
- [InterpretML / EBM](https://github.com/interpretml/interpret) - Microsoft's Explainable Boosting Machine - a glass-box model as accurate as black-box GBMs but fully interpretable. ⭐ 6k+
- [FLAML](https://github.com/microsoft/FLAML) - Microsoft's Fast and Lightweight AutoML library with efficient GBM hyperparameter tuning. ⭐ 4k+
- [AutoGluon-Tabular](https://github.com/autogluon/autogluon) - Amazon's AutoML that ensembles multiple GBMs (XGBoost, LightGBM, CatBoost) for state-of-the-art tabular performance. ⭐ 8k+
- [Scikit-learn HistGradientBoosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) - Fast histogram-based GBM inspired by LightGBM, native to scikit-learn.
- [Scikit-learn GradientBoosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) - Classic gradient boosting implementation in scikit-learn.
- [H2O GBM](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html) - H2O's gradient boosting machine implementation.
- [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) - Decision forest algorithms including gradient boosted trees in TensorFlow.
- [RAPIDS cuML](https://github.com/rapidsai/cuml) - GPU machine learning library including XGBoost/gradient boosting support.
- [SnapML](https://www.zurich.ibm.com/snapml/) - IBM's library for training generalized linear models and gradient boosting machines.
- [PyGBM](https://github.com/ogrisel/pygbm) - Experimental gradient boosting machines in Python (archived).

---

## Research Papers

### 2025

- **Gradient Boosting Reinforcement Learning** (ICML 2025)
  - Fuhrer, B., Tessler, C., & Dalal, G.
  - [[Paper](https://icml.cc/virtual/2025/poster/45118)]
  - *Adapts GBT to RL tasks, outperforming neural networks on structured/categorical data with better OOD robustness.*

- **Robust-Multi-Task Gradient Boosting** (ICML 2025)
  - Emami, S., Martínez-Muñoz, G., & Hernández-Lobato, D.
  - [[Paper](https://arxiv.org/abs/2507.11411)]
  - *R-MTGB framework that models task heterogeneity, detects outlier tasks, and promotes knowledge transfer.*

- **Gradient Boosted Mixed Models** (ICML 2025)
  - Prevett, M. L., Hui, F. K. C., Tho, Z. Y., Welsh, A. H., & Westveld, A. H.
  - [[Paper](https://arxiv.org/abs/2511.00217)]
  - *GBMixed extends boosting to jointly estimate mean and variance components for clustered data.*

- **Inductive Inference of Gradient-Boosted Decision Trees on Graphs** (ICML 2025)
  - Vandervorst, F., Deprez, B., Verbeke, W., & Verdonck, T.
  - [[Paper](https://arxiv.org/abs/2510.05676)]
  - *Graph gradient boosting machine (G-GBM) for supervised learning on heterogeneous/dynamic graphs.*

- **Quadratic Upper Bound for Boosting Robustness** (ICML 2025)
  - You, E., & Lee, H. W.
  - [[Paper](https://icml.cc/media/icml-2025/Slides/44505.pdf)]
  - *Theoretical insights into stability and performance of gradient boosting algorithms.*

- **Statistical Inference for Gradient Boosting Regression** (NeurIPS 2025)
  - [[Paper](https://neurips.cc/virtual/2025/poster/116752)]
  - *Unified framework for statistical inference with dropout and parallel training; establishes central limit theorem for boosting.*

- **MorphBoost: Self-Organizing Universal Gradient Boosting with Adaptive Tree Morphing** (NeurIPS 2025)
  - [[Paper](https://arxiv.org/abs/2511.13234)]
  - *Self-organizing tree structures that dynamically adjust splitting behavior during training.*

- **GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models** (NeurIPS 2025)
  - [[Paper](https://arxiv.org/abs/2505.20685)]
  - *Uses pre-trained tabular models as surrogate with gradient info for high-dimensional optimization.*

- **NRGBOOST: Energy-Based Generative Boosted Trees** (ICLR 2025)
  - Bravo, J.
  - [[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/beca256c79944fa6f31ef4383b9a731a-Paper-Conference.pdf)]
  - *Energy-based generative boosting analogous to XGBoost's second-order boosting; enables generative modeling on tabular data.*

- **Decision Trees That Remember: Recurrent Decision Trees with Memory** (ICLR 2025)
  - Marton, S., Schneider, M., Brinkmann, J., Lüdtke, S., Bartelt, C., & Stuckenschmidt, H.
  - [[Paper](https://openreview.net/forum?id=ReMeDe)]
  - *ReMeDe Trees: recurrent decision trees with memory for sequential data and long-term dependencies.*

- **Boosting Methods for Interval-Censored Data** (ICLR 2025)
  - Bian, Y., Yi, G. Y., & He, W.
  - [[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/82a0696bea2c4ebf726fc796eaca7a55-Paper-Conference.pdf)] [[Code](https://github.com/krisyuanbian/L2BOOST-IC)]
  - *Nonparametric boosting for regression/classification with interval-censored data; L2Boost-CUT and L2Boost-IMP algorithms.*

- **Online Gradient Boosting Decision Tree: In-Place Updates for Adding/Deleting Data** (ICLR 2025)
  - [[Paper](https://openreview.net/pdf/5a0dbbf081d7064bbfbee481ab3e580f09793f08.pdf)]
  - *Online GBDT with incremental and decremental learning support.*

- **GAdaBoost: Efficient and Robust AdaBoost Based on Granular-Ball Structure** (AAAI 2025)
  - Xie, Q., et al.
  - [[Paper](https://arxiv.org/abs/2506.02390)]
  - *Two-stage framework using granular-ball data compression; robust under noisy conditions for multiclass classification.*

### 2024

- **TabR: Tabular Deep Learning Meets Nearest Neighbors** (ICLR 2024)
  - Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A., & Babenko, A.
  - [[Paper](https://arxiv.org/abs/2307.14338)] [[Code](https://github.com/yandex-research/tabular-dl-tabr)]

- **Language Models are Realistic Tabular Data Generators** (ICLR 2024)
  - Borisov, V., Seßler, K., Leemann, T., Heyer, M., & Kasneci, G.
  - [[Paper](https://arxiv.org/abs/2310.01859)] [[Code](https://github.com/vadim-borisov/GReaT)]
  - *GReaT: Generating Realistic Tabular data with Large Language Models.*

- **T2G-Former: Organizing Tabular Features via Relation-Weighted Graph for Tree-Based Models** (AAAI 2024)
  - Yan, J., et al.
  - [[Paper](https://arxiv.org/abs/2312.10372)]
  - *Graph-based feature organization to enhance tree-based models.*

### 2023

- **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second** (ICLR 2023)
  - Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F.
  - [[Paper](https://arxiv.org/abs/2207.01848)] [[Code](https://github.com/automl/TabPFN)]

- **TabDDPM: Modelling Tabular Data with Diffusion Models** (ICML 2023)
  - Kotelnikov, A., Barber, D., & Babenko, A.
  - [[Paper](https://arxiv.org/abs/2209.15421)] [[Code](https://github.com/rotot0/tab-ddpm)]

- **Revisiting Deep Learning Models for Tabular Data** (NeurIPS 2023)
  - Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A.
  - [[Paper](https://arxiv.org/abs/2106.11959)] [[Code](https://github.com/yandex-research/rtdl)]

- **XTab: Cross-table Pretraining for Tabular Transformers** (ICML 2023)
  - Zhu, B., Shi, X., Erickson, N., Li, M., Karypis, G., & Shoaran, M.
  - [[Paper](https://arxiv.org/abs/2305.06090)] [[Code](https://github.com/BingzhaoZhu/XTab)]

### 2022

- **Why do tree-based models still outperform deep learning on tabular data?** (NeurIPS 2022)
  - Grinsztajn, L., Oyallon, E., & Varoquaux, G.
  - [[Paper](https://arxiv.org/abs/2207.08815)] [[Code](https://github.com/LeoGrin/tabular-benchmark)]

- **Gradient Boosted Decision Tree Neural Network** (ICLR 2022)
  - Chen, S., Guestrin, C.
  - [[Paper](https://arxiv.org/abs/2107.05882)]

- **NODE: Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data** (ICLR 2022)
  - Popov, S., Morozov, S., & Babenko, A.
  - [[Paper](https://arxiv.org/abs/1909.06312)] [[Code](https://github.com/Qwicen/node)]

- **SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training** (AAAI 2022)
  - Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T.
  - [[Paper](https://arxiv.org/abs/2106.01342)] [[Code](https://github.com/somepago/saint)]

### 2021

- **Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression** (KDD 2021)
  - Sprangers, O., Schelter, S., & de Rijke, M.
  - [[Paper](https://arxiv.org/abs/2106.01682)] [[Code](https://github.com/elephaint/pgbm)]

- **XGBoost: A Scalable Tree Boosting System Updates** (SIGKDD 2021)
  - Chen, T., & Guestrin, C.
  - [[Paper](https://arxiv.org/abs/1603.02754)] [[Code](https://github.com/dmlc/xgboost)]

- **TabNet: Attentive Interpretable Tabular Learning** (AAAI 2021)
  - Arik, S. Ö., & Pfister, T.
  - [[Paper](https://arxiv.org/abs/1908.07442)] [[Code](https://github.com/dreamquark-ai/tabnet)]

- **Regularization is All You Need: Simple Neural Nets Can Excel on Tabular Data** (NeurIPS 2021)
  - Kadra, A., Lindauer, M., Hutter, F., & Grabocka, J.
  - [[Paper](https://arxiv.org/abs/2106.11189)]

### 2020

- **Gradient Boosted Decision Trees for High Dimensional Sparse Output** (ICML 2020)
  - Si, S., Zhang, H., Keerthi, S. S., Mahajan, D., Dhillon, I. S., & Hsieh, C. J.
  - [[Paper](https://arxiv.org/abs/2001.07248)]

- **AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data** (ICML 2020 AutoML Workshop)
  - Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., & Smola, A.
  - [[Paper](https://arxiv.org/abs/2003.06505)] [[Code](https://github.com/awslabs/autogluon)]

### 2019

- **Gradient Boosting with Piece-Wise Linear Regression Trees** (IJCAI 2019)
  - Shi, Y., Li, J., & Li, Z.
  - [[Paper](https://www.ijcai.org/Proceedings/2019/0476.pdf)] [[Code](https://github.com/GBDT-PL/GBDT-PL)]

- **CatBoost: Unbiased Boosting with Categorical Features** (NeurIPS 2019)
  - Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A.
  - [[Paper](https://arxiv.org/abs/1706.09516)] [[Code](https://github.com/catboost/catboost)]

- **NGBoost: Natural Gradient Boosting for Probabilistic Prediction** (NeurIPS 2019)
  - Duan, T., Avati, A., Ding, D. Y., Basu, S., Ng, A. Y., & Schuler, A.
  - [[Paper](https://arxiv.org/abs/1910.03225)] [[Code](https://github.com/stanfordmlgroup/ngboost)]

### 2018

- **Collective Entity Disambiguation with Structured Gradient Tree Boosting** (NAACL 2018)
  - Yang, Y., Irsoy, O., & Rahman, K. S.
  - [[Paper](https://arxiv.org/pdf/1802.10229.pdf)] [[Code](https://github.com/bloomberg/sgtb)]

- **CatBoost: Unbiased Boosting with Categorical Features** (NeurIPS 2018)
  - Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A.
  - [[Paper](https://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf)] [[Code](https://github.com/catboost/catboost)]

- **Multi-Layered Gradient Boosting Decision Trees** (NeurIPS 2018)
  - Feng, J., Yu, Y., & Zhou, Z. H.
  - [[Paper](https://papers.nips.cc/paper/7614-multi-layered-gradient-boosting-decision-trees.pdf)] [[Code](https://github.com/kingfengji/mGBDT)]

- **Learning Deep ResNet Blocks Sequentially using Boosting Theory** (ICML 2018)
  - Huang, F., Ash, J. T., Langford, J., & Schapire, R. E.
  - [[Paper](https://arxiv.org/abs/1706.04964)] [[Code](https://github.com/JordanAsh/boostresnet)]

- **Functional Gradient Boosting based on Residual Network Perception** (ICML 2018)
  - Nitanda, A., & Suzuki, T.
  - [[Paper](https://arxiv.org/abs/1802.09031)] [[Code](https://github.com/anitan0925/ResFGB)]

- **Finding Influential Training Samples for Gradient Boosted Decision Trees** (ICML 2018)
  - Sharchilev, B., Ustinovskiy, Y., Serdyukov, P., & de Rijke, M.
  - [[Paper](https://arxiv.org/abs/1802.06640)]

- **Boosting Variational Inference: an Optimization Perspective** (AISTATS 2018)
  - Locatello, F., Khanna, R., Ghosh, J., & Rätsch, G.
  - [[Paper](https://arxiv.org/abs/1708.01733)] [[Code](https://github.com/ratschlab/boosting-bbvi)]

### 2017

- **LightGBM: A Highly Efficient Gradient Boosting Decision Tree** (NeurIPS 2017)
  - Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.
  - [[Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)] [[Code](https://github.com/microsoft/LightGBM)]

- **AdaGAN: Boosting Generative Models** (NeurIPS 2017)
  - Tolstikhin, I. O., Gelly, S., Bousquet, O., Simon-Gabriel, C. J., & Schölkopf, B.
  - [[Paper](https://arxiv.org/abs/1701.02386)] [[Code](https://github.com/tolstikhin/adagan)]

- **Gradient Boosted Decision Trees for High Dimensional Sparse Output** (ICML 2017)
  - Si, S., Zhang, H., Keerthi, S. S., Mahajan, D., Dhillon, I. S., & Hsieh, C. J.
  - [[Paper](http://proceedings.mlr.press/v70/si17a.html)] [[Code](https://github.com/springdaisy/GBDT)]

- **Variational Boosting: Iteratively Refining Posterior Approximations** (ICML 2017)
  - Miller, A. C., Foti, N. J., & Adams, R. P.
  - [[Paper](https://arxiv.org/abs/1611.06585)] [[Code](https://github.com/andymiller/vboost)]

- **BDT: Gradient Boosted Decision Tables for High Accuracy and Scoring Efficiency** (KDD 2017)
  - Lou, Y., & Obukhov, M.
  - [[Paper](https://yinlou.github.io/papers/lou-kdd17.pdf)]

- **Gradient Boosting on Stochastic Data Streams** (AISTATS 2017)
  - Hu, H., Sun, W., Venkatraman, A., Hebert, M., & Bagnell, J. A.
  - [[Paper](https://arxiv.org/abs/1703.00377)]

### 2016

- **XGBoost: A Scalable Tree Boosting System** (KDD 2016)
  - Chen, T., & Guestrin, C.
  - [[Paper](https://arxiv.org/abs/1603.02754)] [[Code](https://github.com/dmlc/xgboost)]

- **Boosting with Abstention** (NeurIPS 2016)
  - Cortes, C., DeSalvo, G., & Mohri, M.
  - [[Paper](https://papers.nips.cc/paper/6336-boosting-with-abstention)]

- **Incremental Boosting Convolutional Neural Network for Facial Action Unit Recognition** (NeurIPS 2016)
  - Han, S., Meng, Z., Khan, A. S., & Tong, Y.
  - [[Paper](https://arxiv.org/abs/1707.05395)] [[Code](https://github.com/sjsingh91/IB-CNN)]

- **Boosted Decision Tree Regression Adjustment for Variance Reduction** (KDD 2016)
  - Poyarkov, A., Drutsa, A., Khalyavin, A., Gusev, G., & Serdyukov, P.
  - [[Paper](https://www.kdd.org/kdd2016/papers/files/adf0653-poyarkovA.pdf)]

- **L-EnsNMF: Boosted Local Topic Discovery via Ensemble of NMF** (ICDM 2016)
  - Suh, S., Choo, J., Lee, J., & Reddy, C. K.
  - [[Paper](https://ieeexplore.ieee.org/document/7837872)] [[Code](https://github.com/benedekrozemberczki/BoostedFactorization)]

### 2015

- **Online Gradient Boosting** (NeurIPS 2015)
  - Beygelzimer, A., Hazan, E., Kale, S., & Luo, H.
  - [[Paper](https://arxiv.org/abs/1506.04820)] [[Code](https://github.com/crm416/online_boosting)]

- **Efficient Second-Order Gradient Boosting for Conditional Random Fields** (AISTATS 2015)
  - Chen, T., Singh, S., Taskar, B., & Guestrin, C.
  - [[Paper](http://proceedings.mlr.press/v38/chen15b.html)]

- **Optimal Action Extraction for Random Forests and Boosted Trees** (KDD 2015)
  - Cui, Z., Chen, W., He, Y., & Chen, Y.
  - [[Paper](https://www.cse.wustl.edu/~ychen/public/OAE.pdf)]

- **A Boosting Algorithm for Item Recommendation with Implicit Feedback** (IJCAI 2015)
  - Liu, Y., Zhao, P., Sun, A., & Miao, C.
  - [[Paper](https://www.ijcai.org/Proceedings/15/Papers/255.pdf)]

### Foundational Papers

- **Greedy Function Approximation: A Gradient Boosting Machine** (Annals of Statistics 2001)
  - Friedman, J. H.
  - [[Paper](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)]

- **Stochastic Gradient Boosting** (Computational Statistics & Data Analysis 2002)
  - Friedman, J. H.
  - [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0167947301000652)]

- **A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting** (JCSS 1997)
  - Freund, Y., & Schapire, R. E.
  - [[Paper](https://www.sciencedirect.com/science/article/pii/S002200009791504X)]

- **Experiments with a New Boosting Algorithm** (ICML 1996)
  - Freund, Y., & Schapire, R. E.
  - [[Paper](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf)]

- **Boosting the Margin: A New Explanation for the Effectiveness of Voting Methods** (Annals of Statistics 1998)
  - Schapire, R. E., Freund, Y., Bartlett, P., & Lee, W. S.
  - [[Paper](https://projecteuclid.org/journals/annals-of-statistics/volume-26/issue-5/Boosting-the-margin--A-new-explanation-for-the-effectiveness/10.1214/aos/1024691352.full)]

---

## Tutorials & Guides

### Getting Started

- [Introduction to Gradient Boosting](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/) - Machine Learning Mastery
- [A Gentle Introduction to XGBoost](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/) - Machine Learning Mastery
- [Complete Guide to LightGBM](https://lightgbm.readthedocs.io/en/latest/Quick-Start.html) - Official Quick Start
- [CatBoost Tutorial](https://catboost.ai/docs/concepts/tutorials) - Official Tutorial

### Parameter Tuning

- [XGBoost Parameters Guide](https://xgboost.readthedocs.io/en/latest/parameter.html) - Official Parameters Documentation
- [LightGBM Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) - Official Tuning Guide
- [CatBoost Training Parameters](https://catboost.ai/docs/references/training-parameters/) - Official Parameters Reference
- [Hyperparameter Tuning with Optuna](https://optuna.org/) - Automatic hyperparameter optimization framework

### Advanced Topics

- [Custom Loss Functions in XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html) - Custom Objective and Evaluation
- [Distributed Training with XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/dask.html) - XGBoost with Dask
- [GPU Training with LightGBM](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html) - GPU Acceleration Guide
- [Handling Missing Values](https://catboost.ai/docs/concepts/algorithm-missing-values-processing.html) - CatBoost Missing Values

### Interpretability & Explainability

- [SHAP](https://github.com/shap/shap) - SHapley Additive exPlanations for interpreting GBM predictions. The gold standard for feature importance. ⭐ 23k+
- [SHAP TreeExplainer Tutorial](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/) - Efficient SHAP values for tree models
- [InterpretML EBM Tutorial](https://interpret.ml/docs/ebm.html) - Glass-box interpretable boosting
- [Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html) - Model-agnostic feature importance

---

## Blog Posts

### Comparisons & Benchmarks

- [XGBoost vs LightGBM vs CatBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db) - Comprehensive Comparison
- [When to Use Different Gradient Boosting Libraries](https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm) - Neptune.ai Blog
- [Gradient Boosting vs Random Forest](https://towardsdatascience.com/gradient-boosting-vs-random-forest-cfa3fa8f0d80) - Comparative Analysis

### Deep Dives

- [How Gradient Boosting Works](https://explained.ai/gradient-boosting/) - Visual and Mathematical Explanation
- [Understanding LightGBM's Histogram-based Algorithm](https://datascience.stackexchange.com/questions/26699/decision-tree-learning-with-histogram-based-algorithm) - Technical Deep Dive
- [CatBoost: Handling Categorical Features](https://towardsdatascience.com/catboost-handling-categorical-features-e9b7b4c0a0e9) - Feature Engineering with CatBoost
- [XGBoost Mathematics Explained](https://towardsdatascience.com/xgboost-mathematics-explained-58262530904a) - Mathematical Foundation

### Practical Applications

- [Winning Kaggle Competitions with Gradient Boosting](https://www.kaggle.com/code/arthurtok/introduction-to-ensembling-stacking-in-python) - Kaggle Tutorial
- [Feature Engineering for Gradient Boosting](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308) - Real Competition Insights
- [Time Series Forecasting with Gradient Boosting](https://towardsdatascience.com/time-series-forecasting-with-gradient-boosting-c66f3d9e3e76) - Time Series Applications

---

## Videos & Talks

### Conference Talks

- [XGBoost: A Scalable Tree Boosting System](https://www.youtube.com/watch?v=Vly8xGnNiWs) - Tianqi Chen, KDD 2016
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://www.youtube.com/watch?v=wPr4PkM-ks8) - Microsoft Research
- [CatBoost - the new generation of gradient boosting](https://www.youtube.com/watch?v=8o0e-r0B5xQ) - Yandex

### Educational Videos

- [Gradient Boosting Explained](https://www.youtube.com/watch?v=3CC4N4z3GJc) - StatQuest with Josh Starmer
- [XGBoost Part 1: Regression](https://www.youtube.com/watch?v=OtD8wVaFm6E) - StatQuest
- [XGBoost Part 2: Classification](https://www.youtube.com/watch?v=8b1JEDvenQU) - StatQuest
- [AdaBoost, Clearly Explained](https://www.youtube.com/watch?v=LsK-xG1cLYA) - StatQuest

### Practical Tutorials

- [XGBoost Python Tutorial](https://www.youtube.com/watch?v=GrJP9FLV3FE) - Hands-on Implementation
- [LightGBM Complete Guide](https://www.youtube.com/watch?v=n_ZMQj09S6w) - Complete Tutorial
- [CatBoost Tutorial for Beginners](https://www.youtube.com/watch?v=MfKJlCyKdak) - Getting Started

---

## Books

- **Hands-On Gradient Boosting with XGBoost and scikit-learn** - Corey Wade (Packt Publishing, 2020)
  - [[Book](https://www.packtpub.com/product/hands-on-gradient-boosting-with-xgboost-and-scikit-learn/9781839218354)]

- **The Elements of Statistical Learning** - Hastie, Tibshirani, & Friedman (Springer, 2009)
  - Chapter 10: Boosting and Additive Trees
  - [[Free PDF](https://hastie.su.domains/ElemStatLearn/)]

- **Pattern Recognition and Machine Learning** - Christopher Bishop (Springer, 2006)
  - [[Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)]

- **Machine Learning: A Probabilistic Perspective** - Kevin Murphy (MIT Press, 2012)
  - [[Book](https://probml.github.io/pml-book/)]

---

## Benchmarks & Comparisons

### Benchmark Repositories

- [Tabular Benchmark](https://github.com/LeoGrin/tabular-benchmark) - Comprehensive benchmark comparing deep learning and gradient boosting
- [OpenML Benchmarking Suites](https://www.openml.org/s/218) - Standardized machine learning benchmarks
- [MLPerf Training](https://mlcommons.org/en/training-normal-11/) - Industry-standard ML benchmarks

### Comparison Studies

- [Deep Neural Networks vs Gradient Boosted Trees](https://arxiv.org/abs/2207.08815) - Empirical comparison study
- [AutoML Benchmark](https://arxiv.org/abs/2207.12560) - Comparing automated ML systems

---

## Real-World Applications

### Finance

- Credit Risk Modeling
- Fraud Detection
- Algorithmic Trading
- Customer Churn Prediction

### Healthcare

- Disease Prediction
- Drug Discovery
- Medical Image Analysis
- Patient Risk Stratification

### E-commerce

- Recommendation Systems
- Demand Forecasting
- Price Optimization
- Customer Segmentation

### Industry

- Predictive Maintenance
- Quality Control
- Supply Chain Optimization
- Energy Consumption Forecasting

---

## Related Awesome Lists

- [Awesome Gradient Boosting Papers](https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers) - Academic papers on gradient boosting
- [Awesome Gradient Boosting](https://github.com/talperetz/awesome-gradient-boosting) - Resources for data scientists
- [Awesome Decision Tree Research](https://github.com/benedekrozemberczki/awesome-decision-tree-papers) - Decision tree papers
- [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning) - General ML resources
- [Awesome AutoML](https://github.com/hibayesian/awesome-automl-papers) - AutoML papers and resources

---

## Contributing

Your contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

### How to Contribute

1. **Add a paper**: Use the [paper template](.github/ISSUE_TEMPLATE/add_paper.yml) or submit a PR
2. **Add a resource**: Use the [resource template](.github/ISSUE_TEMPLATE/add_resource.yml)  
3. **Report broken links**: Use the [broken link template](.github/ISSUE_TEMPLATE/broken_link.yml)

If you have any questions about this list, don't hesitate to [open an issue](https://github.com/jxucoder/awesome-gradient-boosting-machines/issues/new).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>If you find this list useful, please consider giving it a ⭐️ star!</sub>
</p>
