# cài đặt cây quyết định sử dụng cho mô hình eXtreme Gradient Boosting (XGBoost).
import numpy as np
from numba import njit

@njit
def softmax(x):
    eZ = np.exp(x)
    return eZ / np.expand_dims(eZ.sum(1), 1)

@njit
def ce_softmax_grad(y_true, y_pred):
    '''
    Hàm tính đạo hàm bậc 1 của hàm mất mát softmax loss cho mảng y_pred

    Tham khảo:
    - https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
    '''
    gradient = softmax(y_pred)

    for i, y in enumerate(y_true):
        gradient[i, y] -= 1

    return gradient

@njit
def ce_softmax_hess(y_true, y_pred):
    '''
    Hàm tính đạo hàm bậc 2 của hàm mất mát softmax loss cho mảng y_pred
    theo đạo hàm bậc 1 của hàm softmax

    Tham khảo:
    - https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    '''
    hessian = softmax(y_pred)
    for i, y in enumerate(y_true):
        hessian[i, :y] *= -hessian[i, y]
        hessian[i, y+1:] *= -hessian[i, y]
        hessian[i, y] *= 1 - hessian[i, y]

    return hessian


class Node:
    '''
    Lớp đối tượng thể hiện một nút trong cây quyết dịnh.

    Những thuộc tính lưu trữ bao gồm:
    - Nút nhánh:
        - đặc trưng được xét.
        - ngưỡng so sánh.
        - nút con trái và con phải.

    - Nút lá chỉ có giá trị dự đoán.

    Tham khảo:
    https://github.com/marvinlanhenke/MLFromScratch/blob/main/DecisionTree/Node.py
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None) -> None:
        '''
        Phương thức khởi tạo một nút của mô hình cây quyết định.

        Đầu vào gồm các tham số tùy chọn, mặc định là None:
        - feature (int):
            chỉ mục đặc trưng.

        - threshold (float):
            ngưỡng giá trị phân nhánh.

        - left, right (Node):
            nút con trái và phải

        - value (any):
            giá trị của nút khi là nút lá.
        '''
        self.feature = feature
        self.threshold = threshold
        
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

@njit
def _gain(y, y_pred, lambda_):
    '''
    Hàm tính độ lợi cho một cách phân nhánh đề xuất
    sử dụng hàm mất mát softmax.

    Đầu vào:
    - y, y_pred (np.ndarray): mảng nhãn đúng và ma trậb các véctơ dự đoán nhãn
    của các đối tượng.

    - lambda_: hệ số chính quy hóa L2

    Đầu ra:
    - gain (float) độ lợi của cách phân nhánh đề xuất.

    Tham khảo:
    - https://medium.com/analytics-vidhya/what-makes-xgboost-so-extreme-e1544a4433bb
    '''
    nominator = ce_softmax_grad(y, y_pred).sum(0) ** 2
    denominator = ce_softmax_hess(y, y_pred).sum(0) + lambda_
    # denominator[denominator==0] = 1

    gain = nominator / denominator
    # gain[np.isnan(gain)] = 0

    return gain.sum()

@njit
def _newton_boosting(y, y_pred, lambda_):
    '''
    Hàm tính giá trị nút lá trên cây
    theo phương trình (5) trong bài báo "XGBoost: A Scalable Tree Boosting System".

    Đầu vào:
    - y, y_pred (np.ndarray): mảng nhãn đúng và ma trậb các véctơ dự đoán nhãn
    của các đối tượng.

    - lambda_: hệ số chính quy hóa L2

    Đầu ra:
    - weights (np.ndarray): giá trị dự đoán cho các nhãn ở nút lá.

    Tham khảo:
    - https://medium.com/analytics-vidhya/what-makes-xgboost-so-extreme-e1544a4433bb
    - https://arxiv.org/pdf/1603.02754.pdf
    '''
    # gradient = ce_softmax_grad(y, y_pred)
    # hessian = ce_softmax_hess(y, y_pred)

    nominator = ce_softmax_grad(y, y_pred).sum(0)
    denominator = ce_softmax_hess(y, y_pred).sum(0) + lambda_
    # denominator[denominator==0] = 1
    
    score = -nominator / denominator
    #score[np.isnan(score)] = 0

    return score

@njit
def _create_split(X, threshold):
    left_idx = (X < threshold).nonzero()[0]
    right_idx = (X >= threshold).nonzero()[0]

    return left_idx, right_idx

@njit
def _best_split_xgb_node(X, y, y_pred_base, lambda_):
    '''
    Hàm tìm cách phân nhánh tốt nhất cho một nút trong cây thành phần
    của mô hình XGBoost.

    Đầu vào:
    - X, y (np.ndarray): mảng đặc trưng và nhãn đúng các đối tượng trong nút.
    - y_pred_base (np.ndarray): mảng các véctơ dự đoán ban đầu.
    - lambda_: hệ số chính quy hóa L2

    Đầu ra:
    - best_feature, best_threshold (float): chỉ mục đặc trưng và giá trị ngưỡng tốt nhất
    để phân nhánh nút đang xét.
    '''
    best_gain = -1.
    best_feature = -1
    best_threshold = -1.

    node_gain = _gain(y, y_pred_base, lambda_)
    # tạo thứ tự duyệt các cột ngẫu nhiên
    # features = np.random.choice(X.shape[1], X.shape[1], False)
    features = np.random.permutation(X.shape[1])
    for feat in features:
        # feat = features[i]
        X_feat = X[:, feat]

        # xác định các giá trị ngưỡng ở mỗi cột và tính độ lợi
        
        thresholds = np.unique(X_feat)
        for threshold in thresholds:
            left_idx, right_idx = _create_split(X_feat, threshold)

            left_gain = _gain(y[left_idx], y_pred_base[left_idx], lambda_)
            right_gain = _gain(y[right_idx], y_pred_base[right_idx], lambda_)

            current_gain = left_gain + right_gain - node_gain
            if current_gain > best_gain:
                best_gain = current_gain
                best_feature = feat
                best_threshold = threshold

    return best_feature, best_threshold


class XGBRegressionTree:
    def __init__(self, lambda_=1., min_samples_split=2, max_depth=None) -> None:
        self.lambda_ = lambda_
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def _is_finished(self, n_samples, depth):
        '''
        Phương thức kiểm tra điều kiện dừng xây dựng cây.
        Dừng xây dựng cây khi thỏa một trong các điều kiện:
        - độ sâu hiện tại >= độ sâu tối đa
        - chỉ có 1 lớp đối tượng tại nút.
        - số đối tượng tại nút < tối thiểu.

        Đầu vào:
        - depth (int): độ sâu hiện tại của cây.

        Đầu ra:
        - stop (bool): kết quả dừng xây dựng cây hay không
        '''
        if (n_samples < self.min_samples_split
            or depth >= self.max_depth
            or self.n_classes == 1):
            return True
        return False

    def _build_tree(self, X, y, y_pred_base, depth=0):
        '''
        Phương thức đệ quy xây dựng cây quyết định.
        
        Đầu vào:
        - X (np.ndarray): mảng đặc trưng của tập dữ liệu.
        - y (np.ndarray, shape=(n_samples,) hoặc (n_samples, n_classes)):
            mảng giá trị nhãn đúng của dữ liệu. Kích thước mảng tùy theo hàm tính độ lợi.
        - depth (int): độ sâu hiện tại của cây.

        Đầu ra:
        - node (Node): nút đươc tạo
        '''
        # kiểm tra điều kiện dừng
        if self._is_finished(X.shape[0], depth):
            leaf_value = _newton_boosting(y, y_pred_base, self.lambda_)
            return Node(value=leaf_value)

        # tìm cột và ngưỡng tốt nhất
        best_feat, best_thresh = _best_split_xgb_node(X, y, y_pred_base, self.lambda_)

        # xây dựng nút con và trả về.
        left_idx, right_idx = _create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx], y[left_idx], y_pred_base[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], y_pred_base[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def fit(self, X, y, y_pred_base):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, y_pred_base)
        return self

    def _traverse_tree(self, x, node):
        while not node.is_leaf():
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value
    
    def predict(self, X):
        return np.apply_along_axis(self._traverse_tree, 1, X, self.root)

    def _read_tree_dict(self, tree_dict:dict):
        '''
        Tạo nút của cây dựa trên thông tin đã lưu.

        Đầu vào:
        - tree_dict (dict): từ điển lưu các thuộc tính của một nút.

        Đầu ra:
        - node (Node): nút của cây chứa các thông tin đọc được.
        '''
        node_val = tree_dict.get('leaf')

        if node_val is not None:
            node = Node(value=node_val)
        else:
            feature = int(tree_dict['split'][1:])
            left_child = self._read_tree_dict(tree_dict['children'][0])
            right_child = self._read_tree_dict(tree_dict['children'][1])

            node = Node(feature, tree_dict['split_condition'], left_child, right_child)

        return node

    @classmethod
    def load_tree(cls, tree_dict:dict):
        tree = cls()
        tree.root = tree._read_tree_dict(tree_dict)
        return tree