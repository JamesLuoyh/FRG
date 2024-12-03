from folktables import ACSDataSource, ACSIncome
import pandas as pd
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)

ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)

# ca_features.to_csv('ca_features.csv', index=False)
# ca_labels.to_csv('ca_labels.csv', index=False)
ca_labels = ca_labels.astype(float)

# normalize 
for c in ["SCHL", "AGEP", "OCCP", "POBP", "RELP", "WKHP"]:
    c_max = ca_features[c].max()
    c_min = ca_features[c].min()
    ca_features[c] = (ca_features[c]-c_min)/(c_max-c_min)



#Convert ca_features "COW", "MAR", "RAC1P", "SEX"

SEX_min = ca_features['SEX'].min()
ca_features['SEX'] -= SEX_min

MAR_min = ca_features['MAR'].min()
print(MAR_min)
ca_features['MAR'] -= MAR_min
print(ca_features['MAR'].min())

COW_min = ca_features['COW'].min()
COW_max = ca_features['COW'].max()

for i in range(int(COW_min), int(COW_max) + 1):
    ca_features[f'COW_{i}'] = ca_features['COW'] == i


RAC1P_min = ca_features['RAC1P'].min()
RAC1P_max = ca_features['RAC1P'].max()

for i in range(int(COW_min), int(COW_max) + 1):
    ca_features[f'RAC1P_{i}'] = ca_features['RAC1P'] == i

ca_features = ca_features.drop(columns=['COW', 'RAC1P'])



MAR_min = ca_features['MAR'].min()
MAR_max = ca_features['MAR'].max()

for i in range(int(MAR_min), int(MAR_max) + 1):
    ca_features[f's_{i}'] = ca_features['MAR'] == i

ca_features['y'] = ca_labels

ca_features['MAR'] = ca_features['s_0']
ca_features['label'] = ca_labels

ca_features = ca_features.astype(float)
print(ca_features.max())
# df = pd.concat([ca_features, ca_labels], axis=1)
print(len(ca_features), ca_features["s_0"].sum(),ca_features["s_1"].sum(), ca_features["s_2"].sum(),ca_features["s_3"].sum(),ca_features["s_4"].sum())
# ca_features.to_csv('income.csv', index=False, header=False)
