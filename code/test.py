import numpy as np
from XGBoost import XGBClassifier
from argparse import ArgumentParser

def create_argument_parser():
    parser = ArgumentParser()
    
    # parser.add_argument('X_train')
    # parser.add_argument('y_train')

    parser.add_argument('X_test', help='đường dẫn tới tập dữ liệu kiểm tra')
    parser.add_argument('save_dir', help='đường dẫn kết quả dự đoán được lưu lại')
    parser.add_argument('learner_path', help='đường dẫn tập tin các siêu tham số của mô hình đã lưu')
    parser.add_argument('tree_path', help='đường dẫn tập tin cấu trúc các cây thành phần của mô hình đã lưu')
    parser.add_argument('--option', choices=['raw', 'prob', 'label'], default='prob',
                                        help='tùy chọn biểu diễn kết quả dự đoán')

    # parser.add_argument('-n', type=int, default=75)
    # parser.add_argument('--eta', type=float, default=0.3)

    
    # parser.add_argument('--min_split', type=int, default=2)
    # parser.add_argument('-d', '--depth', type=int, default=6)
    # parser.add_argument('-l', '--lambda_', type=float, default=1.)
    
    return parser

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()

    with open(f'{args.X_test}') as f:
        ncols = len(f.readline().split(','))

    X_test = np.loadtxt(f'{args.X_test}',delimiter=',', skiprows=1, usecols=range(ncols-1))

    xgb_model = XGBClassifier.load_model(args.learner_path, args.tree_path)
    if args.option == 'raw':
        y_pred = xgb_model.predict_raw(X_test)
    elif args.option == 'prob':
        y_pred = xgb_model.predict_proba(X_test)
    else:
        y_pred = xgb_model.predict(X_test)

    np.savetxt(f'{args.save_dir}', y_pred, fmt='%f', delimiter=',')
    print(f'predictions saved to {args.save_dir}')
