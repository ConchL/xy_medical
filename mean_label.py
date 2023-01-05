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
            # elif col == "shock_Day5":
            elif 'shock' in col:

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