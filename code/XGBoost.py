import numpy as np
import json
from xgbtree import XGBRegressionTree, softmax
from tqdm import tqdm


class XGBClassifier:
    '''
    Lớp đối tượng cài đặt mô hình XGBoost

    Tham khảo: - https://medium.com/analytics-vidhya/what-makes-xgboost-so-extreme-e1544a4433bb
    '''
    def __init__(self, n_estimators=1, learning_rate=0.3,
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


    def fit(self, X, y):
        self.estimators = [XGBRegressionTree(self.lambda_, self.min_samples_split, self.max_depth)
                                    for _ in range(self.n_estimators)]

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
        y_pred = np.zeros((X.shape[0], self.n_classes))
        for i, estimator in enumerate(self.estimators):
            update_pred = estimator.predict(X)
            # y_pred[:, i % self.n_classes] += self.learning_rate * update_pred
            y_pred[:, i % self.n_classes] += update_pred

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

    @classmethod
    def load_model(cls, learner_path, tree_path):
        '''
        Đọc mô hình XGBoost đã được huấn luyện.

        Đầu vào:
        - learner_path (str): đường dẫn tới tập tin json lưu thông tin chi tiết các siêu tham số của mô hình. 
        Nội dung tập tin tương tự kết quả phương thức booster.save_model của thư viện xgboost.
        - tree_path (str): đường dẫn tới tập tin json thể hiện cấu trúc các cây thành phần.
        Nội dung tập tin tương tự kết quả phương thức booster.dump_model của thư viện xgboost.
        '''

        # phâh tích cú pháp json và tạo từ điển các tham số của mô hình
        with open(learner_path) as f:
            model = json.load(f)
            
        with open(tree_path) as f:
            tree_list = json.load(f)
            
        #  khởi tạo mô hình và đọc các siêu tham số cần thiết
        xgb_model = cls()
        attributes = json.loads(model['learner']['attributes']['scikit_learn'])
        xgb_model.n_estimators = attributes['n_estimators']
        xgb_model.learning_rate = attributes['learning_rate']
        xgb_model.n_classes = attributes['n_classes_']
        
        # nạp cấu trúc cây thành phần
        xgb_model.estimators = [XGBRegressionTree.load_tree(tree) for tree in tree_list]
        return xgb_model