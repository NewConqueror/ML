# Kütüphaneleri içe aktar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

# Veriyi yükle ve keşifsel veri analizi
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/diabetes_pima_indians.csv")


print(df.columns)

print(df.describe())

# Sıfırların mantıksız olduğu sütunlar
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Bu sütunlarda 0 olan değerleri NaN yapalım
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

# Her bir sütunun medyanı ile dolduralım
df[cols_to_replace] = df[cols_to_replace].fillna(df[cols_to_replace].median())



# Korelasyon ısısı grafiği
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Features Correlation Matrix")
plt.show()


def detect_outliers_iqr(df):

    outlier_map = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # True olarak işaretle
        outlier_map[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

    # Satır bazında kaç outlier var?
    outlier_counts = outlier_map.sum(axis=1)

    # En az 2 outlier olan satırların index'leri
    outlier_indices = outlier_counts[outlier_counts >= 2].index

    # Bu satırları veri setinden çıkar
    df_cleaned = df.drop(index=outlier_indices).reset_index(drop=True)

    return df_cleaned, outlier_indices

df_cleaned, outlier_indices = detect_outliers_iqr(df)

# Before outlier removal - Boxplot
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.boxplot(data=df)
plt.title("Before Outlier Removal")

# After outlier removal - Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(data=df_cleaned)
plt.title("After Outlier Removal")

plt.tight_layout()
plt.show()

# Eğitim ve Test Setlerine Ayırma
X = df_cleaned.drop(["Outcome"], axis=1)
y = df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Özellik Seçim yöntemleri

# 1. PCA
pca = PCA(n_components=3)  # En fazla 3 ana bileşen alıyoruz
X_pca = pca.fit_transform(X_train_scaled)
X_pca_test = pca.transform(X_test_scaled)

# 2. LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_train_scaled, y_train)
X_lda_test = lda.transform(X_test_scaled)

# 3. ANOVA
selector_anova = SelectKBest(score_func=f_classif, k=5)  # En iyi 5 özellik seçiliyor
X_anova = selector_anova.fit_transform(X_train_scaled, y_train)
X_anova_test = selector_anova.transform(X_test_scaled)


# Model listesi
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "KNN": KNeighborsClassifier(),
    "GaussianNB": GaussianNB()
}

# Özellik setleri
feature_sets = {
    "PCA": (X_pca, X_pca_test),
    "LDA": (X_lda, X_lda_test),
    "ANOVA": (X_anova, X_anova_test)
}

# Modeli eğitme ve değerlendirme fonksiyonu
def train_and_evaluate(X_train_resampled, X_test, y_train_resampled, y_test, model, model_name, method_name):
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    print(f"--- {method_name} + {model_name} ---")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("------------------------------------------------\n")


    if isinstance(model, DecisionTreeClassifier):
        plt.figure(figsize=(35, 25))
        plot_tree(model, filled=True, feature_names=X.columns, class_names=["Negative", "Positive"], rounded=True, fontsize=12)
        plt.title(f"Decision Tree Visualization - {method_name} + {model_name}")
        plt.show()


# Her kombinasyon için model eğitimi ve değerlendirmesi
for method_name, (X_tr, X_te) in feature_sets.items():
    for model_name, model in models.items():
        train_and_evaluate(X_tr, X_te, y_train, y_test, model, model_name, method_name)

# Skorları toplama fonksiyonu
results_table = []

def train_and_collect_scores(X_train, X_test, y_train, y_test, model, model_name, method_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results_table.append({"Method": method_name, "Model": model_name, "Accuracy": acc,
                          "Precision": precision, "recall": recall, "f1-score": f1})

# Skorları toplama
for method_name, (X_tr, X_te) in feature_sets.items():
    for model_name, model in models.items():
        train_and_collect_scores(X_tr, X_te, y_train, y_test, model, model_name, method_name)

results_df = pd.DataFrame(results_table)


# Skorları görselleştirme fonksiyonu
def plot_metric_comparison(results_df, metric):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Model", y=metric, hue="Method")
    plt.ylim(0.0, 1.0)
    plt.title(f"Model Başarı Karşılaştırması (Farklı Özellik Seçim Yöntemleri) - {metric}")
    plt.ylabel(metric)
    plt.legend(title="Özellik Seçimi")
    plt.show()

# Accuracy, Precision, Recall ve F1-score için grafikler
for metric in ["Accuracy", "Precision", "recall", "f1-score"]:
    plot_metric_comparison(results_df, metric)


# Confusion Matrix görselleştirme fonksiyonu
def plot_confusion_matrix(X_train, X_test, y_train, y_test, models, feature_set_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (model_name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
                    ax=axes[i], annot_kws={"size": 16})

        axes[i].set_title(f"{feature_set_name} + {model_name}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.tight_layout()
    plt.show()

# Her bir özellik seti için Confusion Matrix'leri çizme
for method_name, (X_tr, X_te) in feature_sets.items():
    plot_confusion_matrix(X_tr, X_te, y_train, y_test, models, method_name)

