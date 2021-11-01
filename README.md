Link to the paper: https://www.researchgate.net/publication/354527366_Tile2Vec_with_Predicting_Noise_for_Land_Cover_Classification

# Abstract
Tile2vec has proven to be a good representation learning model in the remote sensing field. The success of the model depends on <img src="https://render.githubusercontent.com/render/math?math=l^2">-norm regularization. However, <img src="https://render.githubusercontent.com/render/math?math=l^2">-norm regularization has the main drawback that affects the regularization. We propose to replace the <img src="https://render.githubusercontent.com/render/math?math=l^2">-norm with regularization with predicting noise framework. We then develop an algorithm to integrate the framework. We evaluate the model by using it as a feature extractor on the land cover classification task. The result shows that our proposed model outperforms all the baseline models.

# Tile2vec with predicting Noise (without sampling)
![alt text](https://github.com/MarshalArijona/tile2vec-with-predicting-noise/blob/master/fig/tile2vec-predict-noise-case-1%20(REV).png)

# Tile2vec with predicting Noise (with sampling)
![alt text](https://github.com/MarshalArijona/tile2vec-with-predicting-noise/blob/master/fig/tile2vec-predict-noise-case-2%20(REV)%20(1).png)

# Requirements
1. Python == 3.7
2. PyTorch == 1.9
3. NumPy == 1.21.3

# Disclaimer
Some of our codes reuse the GitHub project [Tile2vec](https://github.com/ermongroup/tile2vec)

# Reference
1. Jean, N., Wang, S., Samar, A., Azzari, G., Lobell, D., & Ermon, S. (2019, July). Tile2vec: Unsupervised representation learning for spatially distributed data. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 3967-3974).
2. Bojanowski, P., & Joulin, A. (2017, July). Unsupervised learning by predicting noise. In International Conference on Machine Learning (pp. 517-526). PMLR.
