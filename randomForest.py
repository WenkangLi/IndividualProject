import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have data in some form:
# X is the feature matrix where each row is a sample and each column is a feature.
# y is the label vector, where each entry corresponds to the label of the sample in X.
# Import the Arrhythmia dataset from the library and replace missing values by '?'.
# path_data_training = "/content/drive/MyDrive/Maestro_ML/EEG/subject_3/input_files/EEG_multiclass_horizontal_concatenation_1.csv"# 800 samples
# path_data_training = "/content/drive/MyDrive/Maestro_ML/EEG/subject_3/input_files/EEG_multiclass_horizontal_concatenation_1_50.csv"# 500 samples
# path_data_training = "/content/drive/MyDrive/Maestro_ML/EEG/subject_3/input_files/EEG_multiclass_horizontal_concatenation_1_10.csv"
# path_data_training = "/content/drive/MyDrive/Maestro_ML/EEG/subject_2/input_files/EEG_S2_multiclass_horiz_concat_1_10samp_10channels.csv"
path_data_training ='G:\Code_Multimodal_Deep_learning\Data\Testing010\All_Combine\Subject_5_ECG_EMG_signalsAllMydata.csv'
# ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv(path_data_training, delimiter=',', header=None,)

df_data = df.iloc[:,:-1]
df_class = df.iloc[:,-1]
# Attribute Scaling

# Normalize the values except for the class labels for each attribute using StandardScaler.
# std_scaler = StandardScaler()
# x_scaled = std_scaler.fit_transform(df_data.values) 


#           / Normalize to [0-1]
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df_data.values)

df_data = pd.DataFrame(x_scaled, index = df_data.index)

print(df_data.shape)

X_train, X_test, y_train, y_test = train_test_split(df_data, df_class, test_size=0.2, shuffle = True, stratify = df_class, random_state=0)


# Train a Random Forest classifier.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set.
y_pred = clf.predict(X_test)

# Calculate accuracy.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
