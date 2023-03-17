import torch
import pickle
import numpy as np
import pandas as pd
import torch.utils.data as Data
from torch.utils.data import DataLoader
from os.path import join
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from base import BaseDataLoader

class PCDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, model_dir, data_dir, batch_size, group, shuffle=True, seed=0,
                 validation_split=0.1, test_split=0.2, num_workers=1):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.group = group.split(',')
        self.seed = seed
        self.validation_split = validation_split
        self.df = self._load_data()
        
        df_PC, df_left = self._split_dataset_by_group()
        df_train, df_valid = self._split_and_normalize(df_PC)
        
        print('Negative sampling...')
        df_train = self._negative_sample(df_train)
        df_valid = self._negative_sample(df_valid)
        df_left = self._negative_sample(df_left)

        self.df_train = df_train
        self.df_valid = df_valid

        self.train_dataset = self._create_dataset(df_train, patient_map_save='train_patient2name_dict.pkl')
        self.valid_dataset = self._create_dataset(df_valid, patient_map_save='valid_patient2name_dict.pkl')
        self.test_dataset = self._create_dataset(df_left, patient_map_save='test_patient2name_dict.pkl', normalize=True)

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        self.valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_valid_dataloader(self):
        return self.valid_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader

    def _load_data(self):
        df = pd.read_excel(join(self.data_dir, 'pancreatic_AP_deleted.xlsx'),
                           sheet_name='T_0.1')
        df = df.sample(frac=1, random_state=self.seed)
        return df

    def get_feature_list(self):
        feature_list = [a for a in list(self.df.columns) if a not in ['NAME', 'Group', 'Outcome']]
        return feature_list

    def _negative_sample(self, df):
        negat_df = df[df['Outcome'] == 0]
        to_sample_num = len(df) - len(negat_df) - len(negat_df)
        sample_df = negat_df.sample(n=to_sample_num, random_state=self.seed, replace=True)
        return pd.concat([df, sample_df])

    def _split_dataset_by_group(self):
        df_filter = self.df[self.df['Group'].isin(self.group)]
        df_left = self.df[~self.df['Group'].isin(self.group)]
        return df_filter, df_left

    def _split_and_normalize(self, df):
        df_copy = df.copy()
        df_other = df_copy[['NAME', 'Group', 'Outcome']]
        df_feature = df_copy.drop(['NAME', 'Group', 'Outcome'], axis=1)
        feature = df_feature.to_numpy()

        scaler = StandardScaler()
        feature_normalized = scaler.fit_transform(feature)
        df_feature.loc[:, :] = feature_normalized

        df_copy = pd.concat([df_other, df_feature], axis=1)

        df_train, df_valid = train_test_split(df_copy, 
            test_size=self.validation_split, random_state=self.seed)

        return df_train, df_valid

    def get_feature_num(self):
        return len(self.df.columns) - 3

    def _create_dataset(self, df, patient_map_save, normalize=False):
        df_filter = df.drop('Group', axis=1)
        print('{} positive and {} negative records.'.format(sum(df_filter['Outcome']),
            len(df_filter)-sum(df_filter['Outcome'])))

        patient_list = list(df_filter['NAME'])
        patient_index2name_dict = {i: p_name for i, p_name in enumerate(patient_list)}
        with open(join(self.model_dir, patient_map_save), 'wb') as f:
            pickle.dump(patient_index2name_dict, f)

        feature_table = df_filter.drop('NAME', axis=1)        
        label = np.array(list(feature_table['Outcome']))
        feature_table = feature_table.drop('Outcome', axis=1)

        # print(list(feature_table.columns)[:10])
        # print(list(feature_table.columns)[-10:])
        
        feature = feature_table.to_numpy()

        if normalize:
            scaler = StandardScaler()
            feature_normalized = scaler.fit_transform(feature)
        else:
            feature_normalized = feature

        patient_index_list = torch.from_numpy(np.array(list(patient_index2name_dict.keys())))
        feature = torch.from_numpy(feature_normalized)
        label = torch.from_numpy(label)

        patient = patient_index_list.type(torch.LongTensor)
        feature = feature.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        dataset = Data.TensorDataset(patient, feature, label)

        return dataset


class PC_PCADataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, model_dir, data_dir, batch_size, group, 
                 n_pc=20, shuffle=True, seed=0,
                 validation_split=0.1, test_split=0.2, num_workers=1):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.group = group.split(',')
        self.n_pc = n_pc
        self.seed = seed
        self.validation_split = validation_split
        self.df = self._load_data()
        
        df_PC, df_left = self._split_dataset_by_group()
        df_train, df_valid = self._split(df_PC)

        train_pca_df, valid_pca_df, test_pca_df = self._PCA_feature(
            train_df=df_train, valid_df=df_valid, test_df=df_left)
        
        print('Negative sampling...')
        df_train = self._negative_sample(train_pca_df)
        df_valid = self._negative_sample(valid_pca_df)
        df_left = self._negative_sample(test_pca_df)

        self.train_dataset = self._create_dataset(df_train, patient_map_save='train_patient2name_dict.pkl')
        self.valid_dataset = self._create_dataset(df_valid, patient_map_save='valid_patient2name_dict.pkl')
        self.test_dataset = self._create_dataset(df_left, patient_map_save='test_patient2name_dict.pkl')

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        self.valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_valid_dataloader(self):
        return self.valid_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader

    def _load_data(self):
        df = pd.read_excel(join(self.data_dir, 'pancreatic_AP_deleted.xlsx'),
                           sheet_name='T_0.1')
        df = df.sample(frac=1, random_state=self.seed)
        return df

    def get_feature_list(self):
        feature_list = [a for a in list(self.df.columns) if a not in ['NAME', 'Group', 'Outcome']]
        return feature_list

    def _negative_sample(self, df):
        negat_df = df[df['Outcome'] == 0]
        to_sample_num = len(df) - len(negat_df) - len(negat_df)
        sample_df = negat_df.sample(n=to_sample_num, random_state=self.seed, replace=True)
        return pd.concat([df, sample_df])

    def _split_dataset_by_group(self):
        df_filter = self.df[self.df['Group'].isin(self.group)]
        df_left = self.df[~self.df['Group'].isin(self.group)]
        return df_filter, df_left

    def _split(self, df):
        df_copy = df.copy()
        df_train, df_valid = train_test_split(df_copy, 
            test_size=self.validation_split, random_state=self.seed)
        return df_train, df_valid

    def _PCA_feature(self, train_df, valid_df, test_df):
        def split_feature_other(df):
            df_other = df[['NAME', 'Group', 'Outcome']]
            df_feature = df.drop(['NAME', 'Group', 'Outcome'], axis=1)
            return df_feature, df_other

        def pca_transform(df, pca):
            feature, other = split_feature_other(df)
            other.reset_index(inplace=True)
            feature_pca = pca.transform(feature)
            n_pc = feature_pca.shape[1]
            feature_pca = pd.DataFrame(feature_pca, columns=['PC-'+str(i) for i in range(1, n_pc+1)])

            pca_df = pd.concat([other, feature_pca], axis=1)
            pca_df.drop('index', axis=1, inplace=True)
            return pca_df

        print('PCA...')
        
        pca = PCA(n_components=self.n_pc)
        df_feature, _ = split_feature_other(pd.concat([train_df, valid_df, test_df]))
        pca.fit(df_feature)
        print('{} components'.format(pca.n_components_))

        train_pca_df = pca_transform(df=train_df, pca=pca)

        valid_pca_df = pca_transform(df=valid_df, pca=pca)
        test_pca_df = pca_transform(df=test_df, pca=pca)

        return train_pca_df, valid_pca_df, test_pca_df

    def get_feature_num(self):
        return len(self.df.columns) - 3

    def _create_dataset(self, df, patient_map_save, normalize=False):
        df_filter = df.drop('Group', axis=1)
        print('{} positive and {} negative records.'.format(sum(df_filter['Outcome']),
            len(df_filter)-sum(df_filter['Outcome'])))

        patient_list = list(df_filter['NAME'])
        patient_index2name_dict = {i: p_name for i, p_name in enumerate(patient_list)}
        with open(join(self.model_dir, patient_map_save), 'wb') as f:
            pickle.dump(patient_index2name_dict, f)

        feature_table = df_filter.drop('NAME', axis=1)        
        label = np.array(list(feature_table['Outcome']))
        feature_table = feature_table.drop('Outcome', axis=1)

        print('Num of features', len(feature_table.columns))

        # print(list(feature_table.columns)[:10])
        # print(list(feature_table.columns)[-10:])
        
        feature = feature_table.to_numpy()

        if normalize:
            scaler = StandardScaler()
            feature_normalized = scaler.fit_transform(feature)
        else:
            feature_normalized = feature

        patient_index_list = torch.from_numpy(np.array(list(patient_index2name_dict.keys())))
        feature = torch.from_numpy(feature_normalized)
        label = torch.from_numpy(label)

        patient = patient_index_list.type(torch.LongTensor)
        feature = feature.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        dataset = Data.TensorDataset(patient, feature, label)

        return dataset
