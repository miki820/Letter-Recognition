import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import seaborn as sns

data = pd.read_csv('letter-recognition.data', header=None)
print(data.shape)
print(data.head(5))
data.info()

class_frequency = (data.groupby(0).size())
print(class_frequency)

plt.bar(class_frequency.index, class_frequency.values)
plt.xlabel('Litery')
plt.ylabel('Częstość')
plt.title('Rozkład klas')
plt.show()

X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train_normalized, y_train)
knn_pred = knn_classifier.predict(X_test_normalized)
knn_acc = accuracy_score(y_test, knn_pred)
knn_prec = precision_score(y_test, knn_pred, average='weighted')
knn_recall = recall_score(y_test, knn_pred, average='weighted')
knn_f1 = f1_score(y_test, knn_pred, average='weighted')
plt.figure(figsize=(8, 6))
knn_matrix = pd.crosstab(y_test, knn_pred, rownames=['Prawdziwe'], colnames=['Przewidziane'])
sns.heatmap(knn_matrix, annot=True, cmap='Blues')
plt.title('Macierz pomyłek - KNN')
plt.show()
print(f'Dokładność modelu KNN: {knn_acc * 100:.2f}%')
print(f'Precyzja modelu KNN: {knn_prec * 100:.2f}%')
print(f'Pełność modelu KNN: {knn_recall * 100:.2f}%')
print(f'Miara-F1 modelu KNN: {knn_f1 * 100:.2f}%')

naiveBayes_classifier = GaussianNB()
naiveBayes_classifier.fit(X_train_normalized, y_train)
naiveBayes_pred = naiveBayes_classifier.predict(X_test_normalized)
naiveBayes_acc = accuracy_score(y_test, naiveBayes_pred)
naiveBayes_prec = precision_score(y_test, naiveBayes_pred, average='weighted')
naiveBayes_recall = recall_score(y_test, naiveBayes_pred, average='weighted')
naiveBayes_f1 = f1_score(y_test, naiveBayes_pred, average='weighted')
plt.figure(figsize=(8, 6))
naiveBayes_matrix = pd.crosstab(y_test, naiveBayes_pred, rownames=['Prawdziwe'], colnames=['Przewidziane'])
sns.heatmap(naiveBayes_matrix, annot=True, cmap='Blues')
plt.title('Macierz pomyłek - Naive Bayes')
plt.show()
print(f'Dokładność modelu Naive Bayes: {naiveBayes_acc * 100:.2f}%')
print(f'Precyzja modelu Naive Bayes: {naiveBayes_prec * 100:.2f}%')
print(f'Pełność modelu Naive Bayes: {naiveBayes_recall * 100:.2f}%')
print(f'Miara-F1 modelu Naive Bayes: {naiveBayes_f1 * 100:.2f}%')

neuralNetwork = MLPClassifier(hidden_layer_sizes=(40, 40, 40), max_iter=1000)
neuralNetwork.fit(X_train_normalized, y_train)
neuralNetwork_pred = neuralNetwork.predict(X_test_normalized)
neuralNetwork_acc = accuracy_score(y_test, neuralNetwork_pred)
neuralNetwork_prec = precision_score(y_test, neuralNetwork_pred, average='weighted')
neuralNetwork_recall = recall_score(y_test, neuralNetwork_pred, average='weighted')
neuralNetwork_f1 = f1_score(y_test, neuralNetwork_pred, average='weighted')
plt.figure(figsize=(8, 6))
neuralNetwork_matrix = pd.crosstab(y_test, neuralNetwork_pred, rownames=['Prawdziwe'], colnames=['Przewidziane'])
sns.heatmap(neuralNetwork_matrix, annot=True, cmap='Blues')
plt.title('Macierz pomyłek - Neural Network')
plt.show()
print(f'Dokładność Neural Network: {neuralNetwork_acc * 100:.2f}%')
print(f'Precyzja Neural Network: {neuralNetwork_prec * 100:.2f}%')
print(f'Pełność Neural Network: {neuralNetwork_recall * 100:.2f}%')
print(f'Miara-F1 Neural Network: {neuralNetwork_f1 * 100:.2f}%')

methods = ['KNN', 'Naive Bayes', 'Neural Network']

plt.bar(methods, [knn_acc, naiveBayes_acc, neuralNetwork_acc])
plt.xlabel('Metoda')
plt.ylabel('Dokładność')
plt.title('Dokładności w różnych modelach')
plt.show()

plt.bar(methods, [knn_prec, naiveBayes_prec, neuralNetwork_prec])
plt.xlabel('Metoda')
plt.ylabel('Precyzja')
plt.title('Precyzja w różnych modelach')
plt.show()

plt.bar(methods, [knn_recall, naiveBayes_recall, neuralNetwork_recall])
plt.xlabel('Metoda')
plt.ylabel('Pełność')
plt.title('Pełność w różnych modelach')
plt.show()

plt.bar(methods, [knn_f1, naiveBayes_f1, neuralNetwork_f1])
plt.xlabel('Metoda')
plt.ylabel('Miara f1')
plt.title('Miara f1 w różnych modelach')
plt.show()