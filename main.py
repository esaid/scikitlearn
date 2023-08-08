import sklearn
# dataset tumeurs cancer classification
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from  sklearn.metrics import accuracy_score

# load dataset
data = load_breast_cancer()
# organize our data
label_names = data['target_names']
labels = data['target']
features_names = data['feature_names']
features = data['data']

# look at our data
indice = 0
print(label_names)
print(labels[indice])
print(features_names[indice])
print(features[indice])

# Split our data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

# initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)

# Make predictions
preds = gnb.predict(test)
print('prediction : ', preds)

# Evaluate accuracy
print(accuracy_score(test_labels, preds))
