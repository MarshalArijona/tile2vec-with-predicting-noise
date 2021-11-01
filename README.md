Link to the paper: [https://www.researchgate.net/publication/354527366_Tile2Vec_with_Predicting_Noise_for_Land_Cover_Classification]{https://www.researchgate.net/publication/354527366_Tile2Vec_with_Predicting_Noise_for_Land_Cover_Classification}

# Abstract
Tile2vec has proven to be a good representation learning model in the remote sensing field. The success of the model depends on $l2$-norm regularization. However, $l2$-norm regularization has the main drawback that affects the regularization. We propose to replace the $l2$-norm with regularization with predicting noise framework. We then develop an algorithm to integrate the framework. We evaluate the model by using it as a feature extractor on the land cover classification task. The result shows that our proposed model outperforms all the baseline models.

# Tile2vec with predicting Noise (without sampling)

# Tile2vec with predicting Noise (with sampling)

# Requirements
Python == 3.7
PyTorch == 1.9
NumPy == 1.21.3

# Disclaimer
Some of our codes reuse the GitHub project [Tile2vec](https://github.com/ermongroup/tile2vec)

# Reference
1. Jean, N., Wang, S., Samar, A., Azzari, G., Lobell, D., & Ermon, S. (2019, July). Tile2vec: Unsupervised representation learning for spatially distributed data. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 3967-3974).
2. Bojanowski, P., & Joulin, A. (2017, July). Unsupervised learning by predicting noise. In International Conference on Machine Learning (pp. 517-526). PMLR.
