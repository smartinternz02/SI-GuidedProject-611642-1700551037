#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pickle


# In[2]:


train = pd.read_csv("Disease_Prediction_Training.csv")
test = pd.read_csv("Disease_Prediction_Testing.csv")


# In[3]:


train


# In[4]:


train.info()


# In[5]:


train.isnull().sum()


# In[6]:


train = train.drop(columns=['Unnamed: 133'])


# In[7]:


train.isnull().sum()


# In[8]:


train.isnull().sum().sum()


# In[9]:


train


# In[10]:


train.describe()


# In[11]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
itching_counts = train['itching'].value_counts()
colors_itching = ['#66b3ff', '#99ff99']
plt.pie(x=itching_counts, labels=['No', 'Yes'], autopct='%.0f%%', colors=colors_itching)
plt.title("Distribution of Itching")

plt.subplot(1, 2, 2)
sneezing_counts = train['continuous_sneezing'].value_counts()
colors_sneezing = ['#ffcc99', '#ff6666']
plt.pie(x=sneezing_counts, labels=['No', 'Yes'], autopct='%.0f%%', colors=colors_sneezing)
plt.title("Distribution of Continuous Sneezing")

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
joint_pain_counts = train['joint_pain'].value_counts()
colors_joint_pain = ['#c2f0c2', '#ff6666']
plt.pie(x=joint_pain_counts, labels=['No', 'Yes'], autopct='%.0f%%', colors=colors_joint_pain)
plt.title("Distribution of Joint Pain")

plt.subplot(1, 3, 2)
chills_counts = train['chills'].value_counts()
colors_chills = ['#c2c2f0', '#ffb366']
plt.pie(x=chills_counts, labels=['No', 'Yes'], autopct='%.0f%%', colors=colors_chills)
plt.title("Distribution of Chills")

plt.tight_layout()
plt.show()


# In[12]:


plt.figure(figsize=(18, 10))

# Bar chart for 'stomach_pain'
plt.subplot(2, 3, 1)
train['stomach_pain'].value_counts().plot(kind='bar', color=['g', 'r'])
plt.title("Distribution of Stomach Pain")
plt.xlabel("Presence of Stomach Pain")
plt.ylabel("Count")

# Bar chart for 'vomiting'
plt.subplot(2, 3, 2)
train['vomiting'].value_counts().plot(kind='bar', color=['g', 'r'])
plt.title("Distribution of Vomiting")
plt.xlabel("Presence of Vomiting")
plt.ylabel("Count")

# Bar chart for 'ulcers_on_tongue'
plt.subplot(2, 3, 3)
train['ulcers_on_tongue'].value_counts().plot(kind='bar', color=['g', 'r'])
plt.title("Distribution of Ulcers on Tongue")
plt.xlabel("Presence of Ulcers on Tongue")
plt.ylabel("Count")

# Bar chart for 'yellow_crust_ooze'
plt.subplot(2, 3, 4)
train['yellow_crust_ooze'].value_counts().plot(kind='bar', color=['g', 'r'])
plt.title("Distribution of Yellow Crust Ooze")
plt.xlabel("Presence of Yellow Crust Ooze")
plt.ylabel("Count")

# Bar chart for 'joint_pain'
plt.subplot(2, 3, 5)
train['joint_pain'].value_counts().plot(kind='bar', color=['g', 'r'])
plt.title("Distribution of Joint Pain")
plt.xlabel("Presence of Joint Pain")
plt.ylabel("Count")

# Bar chart for 'inflammatory_nails'
plt.subplot(2, 3, 6)
train['inflammatory_nails'].value_counts().plot(kind='bar', color=['g', 'r'])
plt.title("Distribution of Inflammatory Nails")
plt.xlabel("Presence of Inflammatory Nails")
plt.ylabel("Count")

plt.tight_layout(h_pad=2.5, w_pad=2.5)
plt.show()


# In[13]:


a = len(train[train['prognosis'] == 'Fungal infection'])

# Count occurrences of Itching with Fungal Infection
b = len(train[(train['itching'] == 1) & (train['prognosis'] == 'Fungal infection')])

# Create a DataFrame
fi = pd.DataFrame(data=[a, b], columns=['Values'], index=['Fungal Infection', 'Itching while Fungal Infection'])

# Plot the bar chart with different colors
sns.barplot(data=fi, x=fi.index, y=fi['Values'], hue=fi.index, palette=['skyblue', 'orange'], legend=False)
plt.title('Importance of Itching symptom to determine Fungal Infection')
plt.show()


# In[14]:


# Count occurrences of Psoriasis
c = len(train[train['prognosis'] == 'Psoriasis'])

# Count occurrences of Joint Pain with Psoriasis
d = len(train[(train['joint_pain'] == 1) & (train['prognosis'] == 'Psoriasis')])

# Create a DataFrame
psoriasis_joint_pain = pd.DataFrame(data=[c, d], columns=['Values'], index=['Psoriasis', 'Joint Pain with Psoriasis'])

# Plot the bar chart with different colors using hue
sns.barplot(data=psoriasis_joint_pain, x=psoriasis_joint_pain.index, y=psoriasis_joint_pain['Values'], hue=psoriasis_joint_pain.index, palette=['lightblue', 'lightcoral'], legend=False)
plt.title('Importance of Joint Pain symptom to determine Psoriasis')
plt.show()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your DataFrame containing symptoms and prognosis
selected_features = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                      'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'scurring', 'skin_peeling',
                      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
                      'red_sore_around_nose', 'yellow_crust_ooze']

# Assuming your DataFrame is named 'train'
df_subset = train[selected_features]

# Convert non-numeric columns to numeric (if needed) or drop them
df_subset = df_subset.apply(pd.to_numeric, errors='coerce').dropna()

# Calculate the correlation matrix
corr_subset = df_subset.corr()

# Increase the size of the heatmap
plt.figure(figsize=(12, 10))

# Use Seaborn's heatmap with a coolwarm color map
sns.heatmap(corr_subset, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f",
            xticklabels=corr_subset.columns.values, yticklabels=corr_subset.columns.values)

plt.title('Correlation Matrix for Selected Features')
plt.show()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your DataFrame containing symptoms and prognosis
skin_condition_features = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'swelling_joints', 'red_spots_over_body', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# Assuming your DataFrame is named 'train'
df_skin_condition = train[skin_condition_features]

# Convert non-numeric columns to numeric (if needed) or drop them
df_skin_condition = df_skin_condition.apply(pd.to_numeric, errors='coerce').dropna()

# Calculate the correlation matrix
corr_skin_condition = df_skin_condition.corr()

# Increase the size of the heatmap
plt.figure(figsize=(12, 10))

# Use Seaborn's heatmap with a coolwarm color map
sns.heatmap(corr_skin_condition, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f",
            xticklabels=corr_skin_condition.columns.values, yticklabels=corr_skin_condition.columns.values)

plt.title('Correlation Matrix for Skin Condition Features')
plt.show()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your DataFrame containing symptoms and prognosis
gi_issues_features = [
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'burning_micturition',
    'fatigue', 'weight_gain', 'loss_of_appetite', 'pain_behind_the_eyes'
]

# Assuming your DataFrame is named 'train'
df_gi_issues = train[gi_issues_features]

# Convert non-numeric columns to numeric (if needed) or drop them
df_gi_issues = df_gi_issues.apply(pd.to_numeric, errors='coerce').dropna()

# Calculate the correlation matrix
corr_gi_issues = df_gi_issues.corr()

# Increase the size of the heatmap
plt.figure(figsize=(12, 10))

# Use Seaborn's heatmap with a coolwarm color map
sns.heatmap(corr_gi_issues, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f",
            xticklabels=corr_gi_issues.columns.values, yticklabels=corr_gi_issues.columns.values)

plt.title('Correlation Matrix for Gastrointestinal Issues Features')
plt.show()


# In[18]:


print(train.columns.tolist())


# In[19]:


import numpy as np

correlation_threshold = 0.9

numeric_df = train.drop('prognosis', axis=1)
corr_matrix = numeric_df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]


df_filtered = numeric_df.drop(to_drop, axis=1)


# In[20]:


print("Dropped columns With high correlation:")
print(to_drop)

print("\n\nFiltered columns:")
print(df_filtered.columns.tolist())

# import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing features and target variable
selected_prognosis = 'Tuberculosis'
selected_columns = ['weight_loss', 'fatigue', 'prognosis']

# Filter the DataFrame for the selected prognosis
df_tuberculosis = train[train['prognosis'] == selected_prognosis]

# Melt the DataFrame for better visualization
df_melted = pd.melt(df_tuberculosis[selected_columns], id_vars='prognosis', var_name='variable', value_name='value')

# Create a swarmplot
plt.figure(figsize=(10, 6))
sns.swarmplot(data=df_melted, x='variable', y='value', hue='prognosis', palette='Set2')
plt.title(f'Swarmplot for {selected_prognosis}')
plt.show()

# In[21]:


test.info()


# In[22]:


def preprocess_test_data(test, to_drop):
    test.drop(columns=to_drop, inplace=True)
    return test


# In[23]:


test_data = preprocess_test_data(test, to_drop)


# In[24]:


X = df_filtered
y = train['prognosis']


# In[25]:


X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size =0.8, random_state=42)


# In[26]:


print("Training set - X:", X_train.shape, "y:", y_train.shape)
print("Validation set - X:", X_val.shape, "y:", y_val.shape)
print("Testing set - X:", X_test.shape, "y:", y_test.shape)


# In[27]:


def evaluate_model(classifier):
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_val)
    yt_pred = classifier.predict(X_train)
    y_pred1 = classifier.predict(X_test)
    print('Training Accuracy: ', accuracy_score(y_train, yt_pred))
    print('Validation Accuracy: ', accuracy_score(y_val, y_pred))
    print('Testing Accuracy: ', accuracy_score(y_test, y_pred1))
    return [(accuracy_score(y_train, yt_pred)), (accuracy_score(y_val, y_pred)), (accuracy_score(y_test, y_pred1))]


# In[28]:


KNN = KNeighborsClassifier(n_neighbors=10)
KNN_result = evaluate_model(KNN)


# In[29]:


SVM = SVC(C=1)
SVM_result = evaluate_model(SVM)


# In[30]:


DTC = DecisionTreeClassifier(max_features=10)
DTC_result = evaluate_model(DTC)


# In[31]:


RFC = RandomForestClassifier(max_depth = 13)
RFC_result = evaluate_model(RFC)




# In[33]:


RFC_feature_importances = pd.DataFrame(RFC.feature_importances_,
                                   index=X_train.columns,
                                   columns=['Importance']).sort_values('Importance', ascending=False)

RFC_feature_importances.head(50)


# In[34]:


print("Random Forest Classifier Model\n")
for num_features in range(1, 91, 10):
    # Select top N features
    top_features = RFC_feature_importances.head(num_features).index
    X_train_selected = X_train[top_features]
    X_val_selected = X_val[top_features]

    # Train the model
    RFC_selected = RandomForestClassifier()
    RFC_selected.fit(X_train_selected, y_train)

    # Evaluate accuracy on both training and validation sets
    train_accuracy = RFC_selected.score(X_train_selected, y_train)
    val_accuracy = RFC_selected.score(X_val_selected, y_val)

    print(f"Accuracy with {num_features} features - Training Accuracy: {train_accuracy} , Validation Accuracy: {val_accuracy}")


# In[35]:


print("KNN Model\n")
for num_features in range(1, 91, 10):
    # Select top N features
    top_features = RFC_feature_importances.head(num_features).index
    X_train_selected = X_train[top_features]
    X_val_selected = X_val[top_features]

    # Train the model
    KNN_selected = KNeighborsClassifier()
    KNN_selected.fit(X_train_selected, y_train)

    # Evaluate accuracy on both training and validation sets
    train_accuracy = KNN_selected.score(X_train_selected, y_train)
    val_accuracy = KNN_selected.score(X_val_selected, y_val)
    print(f"Accuracy with {num_features} features - Training Accuracy: {train_accuracy} , Validation Accuracy: {val_accuracy}")


# In[36]:


# Choosing the top 50 features as there is little change in accuracy from 50 to 80 features

top_features = RFC_feature_importances.head(50).index
X1 = X[top_features]
y1 = y
X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y1, train_size=0.8, random_state=42)
X_test1 = X_test[top_features]
X1.columns


# In[37]:


# Adding serial numbers to the list of columns
columns_with_serial = list(enumerate(X1.columns, start=1))

# Displaying the serial numbers and corresponding column names
for serial, column in columns_with_serial:
    print(f"{serial}. {column}")


# In[38]:


RFC_model = RandomForestClassifier()
RFC_model.fit(X_train1, y_train1)
y_train_pred1 = RFC_model.predict(X_train1)
y_val_pred1 = RFC_model.predict(X_val1)
y_test_pred1 = RFC_model.predict(X_test1)

print('Training Accuracy: ', accuracy_score(y_train1, y_train_pred1))
print('Validation Accuracy: ', accuracy_score(y_val1, y_val_pred1))
print('Testing Accuracy: ', accuracy_score(y_test, y_test_pred1))


# In[39]:


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train1, y_train1)
y_train_pred1 = knn_model.predict(X_train1)
y_val_pred1 = knn_model.predict(X_val1)
y_test_pred1 = knn_model.predict(X_test1)

print('Training Accuracy: ', accuracy_score(y_train1, y_train_pred1))
print('Validation Accuracy: ', accuracy_score(y_val1, y_val_pred1))
print('Testing Accuracy: ', accuracy_score(y_test, y_test_pred1))


# In[40]:


test_predictions = pd.DataFrame(y_test_pred1, columns=["predicted"])
result_df = test.join(test_predictions)[["prognosis", "predicted"]]
result_df.head(10)


# In[44]:


pickle.dump(knn_model , open('knn_model.pkl' ,'wb'))




# In[ ]:




