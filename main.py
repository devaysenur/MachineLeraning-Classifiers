import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Veri setini yükleme ve optimizasyn:
file = 'data.csv'
data = pd.read_csv(file, delimiter=';')

# veri setine dair bilgilerin gösterilmesi:
data.info(), data.head()

# features(özellikler) ve target ayrımı:
X = data.drop(columns=['outcome'])
y = data['outcome']

# eksik verilerin mean value ile doldurulması:
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Özelliklerin normalizasyonu:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_optimized = pd.DataFrame(X_scaled, columns=X.columns)

# Naive Bayes Sınıflandırıcısı:
# Veri setini %70 eğitim, %30 test olarak ayırma işlemi:
X_train, X_test, y_train, y_test = train_test_split(X_optimized, y, test_size=0.3, random_state=42)

# Naive Bayes sınıflandırıcısının uygulanması:
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# tahminler:
y_pred_train_nb = nb_model.predict(X_train)
y_pred_test_nb = nb_model.predict(X_test)

print("Naive Bayes-Eğitim seti doğruluğu: ", accuracy_score(y_train, y_pred_train_nb))
print("Naive Bayes-Test seti doğrulğu: ", accuracy_score(y_test, y_pred_test_nb))

# K-En yakın komşuluk sınıflandırıcısı:
# Veri seti %70 eğitim, %30 test olarak ayrıldı:
X_train, X_test, y_train, y_test = train_test_split(X_optimized, y, test_size=0.3, random_state=42)

# En iyi k değeri belirlendi:
best_k = 1
best_score = 0

for k in range(1, 21):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    score = knn_model.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k

# En iyi k değeri ile model oluşturma:
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)

# tahminler:
y_pred_train_knn = knn_model.predict(X_train)
y_pred_test_knn = knn_model.predict(X_test)

print("K-NN - En iyi K değeri: ", best_k)
print("K-NN - Eğitim seti doğruluğu: ", accuracy_score(y_train, y_pred_train_knn))
print("K-NN - Test seti doğrulğu: ", accuracy_score(y_test, y_pred_test_knn))

# MLP ve SVM sınıflandırıcıları
# Veri setinin %70 eğitim, %30 test olarak ayrımı:
X_train, X_test, y_train, y_test = train_test_split(X_optimized, y, test_size=0.3, random_state=42)

# MLP sınıflandırıcısı:
mlp_model = MLPClassifier(max_iter=1000, random_state=42)  # max_iter artırıldı. (min 300 uyarısı vermişti)
mlp_model.fit(X_train, y_train)

# tahminler:
y_pred_train_mlp = mlp_model.predict(X_train)
y_pred_test_mlp = mlp_model.predict(X_test)

print("MLP - Eğitim Seti Doğruluğu: ", accuracy_score(y_train, y_pred_train_mlp))
print("MLP - Test Seti Doğruluğu: ", accuracy_score(y_test, y_pred_test_mlp))

# SVM sınıflandırıcısı:
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# tahminler:
y_pred_train_svm = svm_model.predict(X_train)
y_pred_test_svm = svm_model.predict(X_test)

print("SVM - Eğitim Seti Doğruluğu: ", accuracy_score(y_train, y_pred_train_svm))
print("SVM - Test Seti Doğruluğu: ", accuracy_score(y_test, y_pred_test_svm))


#Her bir sınıflandırıcı iç,n konfüzyon matrisleri ve bazı parametreler:

# Naive Bayes için
print("Naive Bayes:")
conf_matrix_nb = confusion_matrix(y_test, y_pred_test_nb)
print("Konfüzyon Matrisi:")
print(conf_matrix_nb)
print("Hassasiyet(Precision):", precision_score(y_test, y_pred_test_nb))
print("Özgüllük(Recall):", recall_score(y_test, y_pred_test_nb))
print("F1-Skoru:", f1_score(y_test, y_pred_test_nb))


# K-NN için
print("\nK-NN:")
conf_matrix_knn = confusion_matrix(y_test, y_pred_test_knn)
print("Konfüzyon Matrisi:")
print(conf_matrix_knn)
print("Precision:", precision_score(y_test, y_pred_test_knn))
print("Recall:", recall_score(y_test, y_pred_test_knn))
print("F1-Skoru:", f1_score(y_test, y_pred_test_knn))


# MLP için
print("\nMLP:")
conf_matrix_mlp = confusion_matrix(y_test, y_pred_test_mlp)
print("Confusion Matrix:")
print(conf_matrix_mlp)
print("Precision:", precision_score(y_test, y_pred_test_mlp))
print("Recall:", recall_score(y_test, y_pred_test_mlp))
print("F1-Skoru:", f1_score(y_test, y_pred_test_mlp))


# SVM için
print("\nSVM:")
conf_matrix_svm = confusion_matrix(y_test, y_pred_test_svm)
print("Confusion Matrix:")
print(conf_matrix_svm)
print("Hassasiyet (Precision):", precision_score(y_test, y_pred_test_svm))
print("Özgüllük (Recall):", recall_score(y_test, y_pred_test_svm))
print("F1-Skoru:", f1_score(y_test, y_pred_test_svm))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_test_svm))
