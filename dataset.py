from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import argparse


class MedicalData(Dataset):
    def __init__(self, mode='train', train_file=None, valid_file=None, test_file=None, label_name=None):
        assert mode in ['train', 'valid', 'test'], f"mode should be set to the one of ['train', 'valid', 'test']"
        self.mode = mode
        self.dataset = []
        if mode == 'train':
            self.dataset = self._load_data(train_file, label_name)
            print(f'Number of training dataset: {len(self.dataset)}')
        elif mode == 'valid':
            self.dataset = self._load_data(valid_file, label_name)
            print(f'Number of validation dataset: {len(self.dataset)}')
        else:
            self.dataset = self._load_data(test_file, label_name)
            print(f'Number of test dataset: {len(self.dataset)}.')

    def __getitem__(self, idx):
        features_content = self.dataset[idx]['data']
        if self.mode in ['train', 'valid', 'test']:
            labels = self.dataset[idx]['label']
            return features_content, labels
        else:
            raise NameError

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_function(batch):
        features_content, labels = zip(*batch)
        return features_content, labels

    def _format_label(self,label):
        if np.isnan(label):
            label = 0
        else:
            label = int(label)

        if label > 1:
            label = 1
        return label

    def _format_data(self,data):
        if data == "M":
            return 1.0
        if data == "F":
            return 0.0

        if np.isnan(data):
            return 0.0
        else:
            return float(data)





    def _load_data(self, file_name, label_name):
        # label_name = 'dead'
        df = pd.read_csv('dataset/{}/{}'.format(label_name, file_name))

        common_need = ['Coagulation.dysfunction', 'age', 'age_raw', 'aids', 'alcohol_abuse',
                            'antibiotic.used_kinds_n',
                            'antibiotic_group.used_kinds_n', 'antibiotic_group1', 'antibiotic_group2',
                            'antibiotic_group3',
                            'antibiotic_group4', 'antibiotic_group5', 'antibiotic_group6', 'antibiotic_group7',
                            'antibiotic_group8',
                            'antibiotic_group9', 'apsiii', 'apsiii_prob', 'basic_disease_n', 'blood_loss_anemia',
                            'cardiac_arrhythmias',
                            'chronic_pulmonary', 'coagulopathy', 'congestive_heart_failure', 'deficiency_anemias',
                            'depression', 'diabetes_complicated',
                            'diabetes_uncomplicated', 'drug_abuse', 'expire_flag', 'fluid_electrolyte', 'gender',
                            'hypertension', 'hypothyroidism',
                            'key_disease_group.diabete', 'key_disease_group.heat', 'key_disease_group.liver_disease',
                            'key_disease_group.lung',
                            'key_disease_group.other', 'key_disease_group.renal_failure', 'key_disease_group.tumor',
                            'key_disease_n', 'liver_disease', 'lods', 'los_hospital', 'los_icu', 'lymphoma',
                            'metastatic_cancer', 'obesity', 'other_neurological', 'paralysis', 'peptic_ulcer',
                            'peripheral_vascular',
                            'psychoses', 'pulmonary_circulation', 'qsofa', 'renal_failure', 'rheumatoid_arthritis',
                            'sirs', 'sofa', 'solid_tumor', 'valvular_disease', 'weight_loss']

        result = []
        for index, row in df.iterrows():
            data = {}
            data["day1"] = []
            data["day2"] = []
            data["day3"] = []
            data["day4"] = []
            data['common'] = []
            # data["day5"] = []

            label = {}

            for col in df.columns:
                if col == "dead":
                    label["dead"] = self._format_label(row[col])
                elif col == "shock_Day5":
                    label["shock"] = self._format_label(row[col])
                elif col == "ards_Day5":
                    label["ards"] = self._format_label(row[col])
                elif col == "aki_stage_Day5":
                    label["aki_stage"] = self._format_label(row[col])
                elif col == "liver_injury_Day5":
                    label["liver_injury"] = self._format_label(row[col])
                elif col == "dic_Day5":
                    label["dic"] = self._format_label(row[col])
                else:
                    if col != "icustay_id":
                        if "Day1" in col:
                            data["day1"].append(self._format_data(row[col]))
                        if "Day2" in col:
                            data["day2"].append(self._format_data(row[col]))
                        if "Day3" in col:
                            data["day3"].append(self._format_data(row[col]))
                        if "Day4" in col:
                            data["day4"].append(self._format_data(row[col]))
                        # if "Day5" in col:
                        #     data["day5"].append(self._format_data(row[col]))
                if col in common_need:
                    data["common"].append(self._format_data(row[col]))



            temp = {}
            temp["data"] = data
            temp["label"] = label
            result.append(temp)
        return result






if __name__ == '__main__':
    # train_file = 'train.csv'  # 仅有5天时间变量，无常规变量的数据
    # valid_file = 'valid.csv'
    # test_file = 'test.csv'
    train_file = 'train_add_normal_std_shock.csv'  # 有5天时间变量（90/day）和常规变量的数据
    valid_file = 'valid_add_normal_std_shock.csv'
    test_file = 'test_add_normal_std_shock.csv'
    train = MedicalData(mode="train",train_file=train_file)

    print(len(train))

    print(train.__getitem__(1))

    data,label = train.__getitem__(0)

    for i in data["day1"]:
        print(type(i))

