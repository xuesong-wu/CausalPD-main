import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
# from utils.tools import StandardScaler
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')
    
class Dataset_Train(Dataset):
    """
    PyTorch Dataset for training/validation/testing pavement distress data.
    Supports static meta features and external features.
    Args:
        root_path: Directory containing the data files.
        flag: 'train', 'val', or 'test'.
        size: [seq_len, label_len, pred_len].
        features: 'S', 'M', or 'MS'.
        data_path: Main data file (npy).
        ext_path: External features file (csv or npy).
        meta_path: Static meta features file (csv).
        meta_dim: If >0, use meta features.
        target: Target column for univariate.
        scale: Whether to standardize data.
        timeenc: 0 for basic time features, 1 for advanced encoding.
        freq: Data frequency string (e.g., 'h', 'd').
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='pavement_distress.npy', ext_path = 'ext.csv',
                 meta_path='meta.csv', meta_dim=0,
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size: [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # flag must be one of ['train', 'test', 'val']
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features  # 'S': univariate, 'M': multivariate, 'MS': multi-input single-output
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_meta = True if meta_dim > 0 else False

        self.root_path = root_path
        self.data_path = data_path
        self.ext_path = ext_path
        self.meta_path = meta_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))

        # Reshape and filter data according to dataset type
        if self.root_path == "./data/GridSZ/":
            df_raw = df_raw[:, :, :, -1]
            T, H, W = df_raw.shape
            df_raw = df_raw.reshape(T, H * W)
            features = [str(i) for i in range(1, H * W + 1)]
            df_raw = pd.DataFrame(df_raw, columns=features)

            column_sums = df_raw.sum()
            non_zero_columns = column_sums[column_sums > T].index.tolist()
            df_raw = df_raw[non_zero_columns]
            df_index = pd.DataFrame(non_zero_columns, columns=["index"])
            df_index.to_csv(os.path.join(self.root_path, "index.csv"), index=False)
        elif self.root_path in ["./data/SegmentSZ/", "./data/Shanghai/"]:
            df_raw = df_raw[:, :, -1]
            T, N = df_raw.shape
            features = [str(i) for i in range(1, N + 1)]
            df_raw = pd.DataFrame(df_raw, columns=features)

        # Load external features
        if self.root_path in ["./data/GridSZ/", "./data/SegmentSZ/"]:
            df_ext = pd.read_csv(os.path.join(self.root_path, "ext.csv"))
        elif self.root_path == "./data/Shanghai/":
            df_ext = np.load(os.path.join(self.root_path, "ext.npy"))

        # Load static meta features if enabled
        if self.use_meta:
            df_meta = pd.read_csv(os.path.join(self.root_path, self.meta_path))
            # Remove 'row_col' column if present
            if 'row_col' in df_meta.columns:
                df_meta = df_meta.drop(columns=['row_col'])
            # Only keep rows corresponding to non-zero columns
            df_meta = df_meta.iloc[non_zero_columns].reset_index(drop=True)
            self.meta_data = df_meta.values
            self.meta_scaler = StandardScaler()
            self.meta_data = self.meta_scaler.fit_transform(self.meta_data)
        else:
            self.meta_data = None
            self.meta_scaler = None

        # Merge date column for time features
        df_raw['date'] = df_ext['date']
        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        # Split data into train/val/test
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select features
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Standardize data if required
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Time feature encoding
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # Normalize external features and concatenate with time features
        df_ext_sub = df_ext.iloc[border1:border2].copy()
        for col in df_ext_sub.columns:
            if col != 'date':
                min_val = df_ext_sub[col].min()
                max_val = df_ext_sub[col].max()
                if max_val - min_val != 0:
                    df_ext_sub[col] = (df_ext_sub[col] - min_val) / (max_val - min_val) - 0.5
                else:
                    df_ext_sub[col] = 0.0
        if 'date' in df_ext_sub.columns:
            df_ext_sub = df_ext_sub.drop(columns='date')
        ext_array = df_ext_sub.values
        data_stamp = np.concatenate([data_stamp, ext_array], axis=1)

        # Final data for model input
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_meta and self.meta_data is not None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, self.meta_data
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size: [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # flag must be 'pred'
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # df_raw.columns: ['date', ...(other features), target feature]
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
