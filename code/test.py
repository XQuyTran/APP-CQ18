import numpy as np
from xgboost import XGBClassifier
from argparse import ArgumentParser

def create_argument_parser():
    parser = ArgumentParser()
    
    parser.add_argument('X_train')
    parser.add_argument('y_train')
    parser.add_argument('X_test')
    parser.add_argument('save_dir')
    parser.add_argument('--progress', choices=[True, False], type=bool, default=False)
    
    return parser

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()

    with open(f'{args.X_train}') as f:
        ncols = len(f.readline().split(','))

    X_train = np.loadtxt(f'{args.X_train}',delimiter=',', skiprows=1, usecols=range(ncols-1))
    X_test = np.loadtxt(f'{args.X_test}',delimiter=',', skiprows=1, usecols=range(ncols-1))
    y_train = np.load(f'{args.y_train}')

    xgb_model = XGBClassifier(75, 0.2, progress=args.progress)

    xgb_model = xgb_model.fit(X_train, y_train)
    y_pred_prob = xgb_model.predict_proba(X_test)

    np.savetxt(f'{args.save_dir}', y_pred_prob, fmt='%f', delimiter=',')
    print(f'predictions saved to {args.save_dir}')
