import numpy as np
from xgbtree import XGBRegressionTree, softmax
from tqdm import tqdm


class XGBClassifier:
    '''
    Lớp đối tượng cài đặt mô hình XGBoost

    Tham khảo: - https://medium.com/analytics-vidhya/what-makes-xgboost-so-extreme-e1544a4433bb
    '''
    def __init__(self, n_estimators, learning_rate=0.3,
                        min_samples_split=2, max_depth=6, lambda_=1.) -> None:
        '''
        Khởi tạo mô hình XGBoost

        Đầu vào:
        - n_estimator (int): số cây thành phần cho mô hình XGBoost.
        - learning_rate (Optional[float], mặc định=0.3): trọng số dự đoán cho các cây thành phần.
        - min_samples_split (int, mặc định=2): số đối tượng ít nhất để có thể phân nhánh.
        - max_depth (Optional[int]): độ sâu tối đa của cây.
        - lambda_ (float, mặc định=1): hệ số chính quy hóa L2
        '''
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.lambda_ = lambda_

        self.estimators = [XGBRegressionTree(self.lambda_, self.min_samples_split, self.max_depth)
                                    for _ in range(self.n_estimators)]

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.n_classes, self.init_pred = np.unique(y, return_counts=True)
        self.n_classes = len(self.n_classes)

        self.init_pred = self.init_pred.astype(np.float32) / self.n_samples
        y_pred = np.full((self.n_samples, self.n_classes), self.init_pred)
        
        for estimator in tqdm(self.estimators, 'fitting estimators'):
            estimator.fit(X, y, y_pred)

            update_pred = estimator.predict(X)
            if np.isnan(update_pred).any():
                np.savetxt('current_raw.csv', y_pred, fmt='%f', delimiter=',')
                np.savetxt('update_raw.csv', update_pred, fmt='%f', delimiter=',')
                raise ValueError('Leaf node(s) contain NaN(s)')
                
            y_pred += self.learning_rate * update_pred
            y_pred -= y_pred.max(1, keepdims=True) # https://cs231n.github.io/linear-classify/#softmax

        return self

    def predict(self, X, prob=False, raw=False):
        y_pred = np.full((self.n_samples, self.n_classes), self.init_pred)
        for estimator in self.estimators:
            update_pred = estimator.predict(X)
            y_pred += self.learning_rate * update_pred

        y_pred -= y_pred.max(1, keepdims=True) # https://cs231n.github.io/linear-classify/#softmax
        if not raw:
            y_pred = softmax(y_pred)
            if not prob:
                y_pred = np.argmax(y_pred, 1)

        return y_pred

    def predict_proba(self, X):
        return self.predict(X, True)

    def predict_raw(self, X):
        return self.predict(X, raw=True)