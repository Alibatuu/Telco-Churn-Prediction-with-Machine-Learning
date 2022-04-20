import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def load():
    data = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    return data


#####################################
# Keşifçi Veri Analizi
#####################################

df = load()
check_df(df)

# Adım 1 : Numerik ve kategorik değişkenleri yakalayınız

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 2 : Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

for col in cat_cols:
    cat_summary(df, col)
for col in num_cols:
    num_summary(df, col, plot=False)

# Adım 4 : Kategorik değişkenler ile hedef değişken incelemesini yapınız.

for col in cat_cols:
    print(pd.DataFrame({"TARGET_COUNT": df.groupby(col)["Churn"].count(),
                       "TARGET_RATIO": df.groupby(col)["Churn"].count() / df.shape[0]}), end="\n\n")

# Adım 5 : Aykırı gözlem var mı inceleyiniz

for col in num_cols:
    print(col, check_outlier(df, col)) # Aykırı değer bulunmamaktadır.

# Adım 6 : Eksik gözlem var mı inceleyiniz

missing_values_table(df) # TotalCharges değişkeninde 11 adet eksik veri bulunmaktadır.

#########################################################
# Görev 2 : Feature Engineering
#########################################################

# Adım 1 : Eksik ve aykırı değerler için gerekli işlemleri yapınız.

df["TotalCharges"].fillna(0, inplace=True)
missing_values_table(df)
# Aykırı değer bulunmadığı için herhangi bir işlem yapılmamıştır.

# Adım 2 : Yeni değişkenler oluşturunuz.

df.loc[(df["PhoneService"] == "Yes") &
       (df["InternetService"] != "No") &
       (df["StreamingTV"] == "Yes") &
       (df["StreamingMovies"] == "Yes"),
       ["New_Using_All_Services"]] = "Yes"
df["New_Using_All_Services"].fillna("No", inplace=True)

df.loc[(df["OnlineSecurity"] == "Yes") &
       (df["OnlineBackup"] == "Yes") &
       (df["DeviceProtection"] == "Yes"),
       ["New_Safe_Customer"]] = "Yes"
df["New_Safe_Customer"].fillna("No",inplace=True)

# Adım 3 : Encoding işlemlerini gerçekleştiriniz

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
df.head()
num_cols = [col for col in num_cols if "customerID" not in col]

# Adım 4 : Numerik değişkenler için standartlaştırma yapınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#########################################################
# Görev 3 : Modelleme
#########################################################

# Adım 1:  Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

y = df["Churn"]
X = df.drop(["customerID", "Churn"], axis=1)

# Random Forest Classifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# 0.7889256980596309

# Logistic Regression

df.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
log_model = LogisticRegression(solver='liblinear').fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1"])
cv_results['test_accuracy'].mean()
# 0.805339255758436

# CART
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
# 0.74

# GBM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8032090780050325

# XGBoost

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7803508492483386

# LightGBM

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7941227054971289

# CatBoost

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7965362684689334




# GBM, LightGBM, CatBoost, Logistic Regression

# Adım 2 : Seçtiğiniz modeller ile hiperparametreoptimizasyonu gerçekleştirin ve bulduğunuz hiparparametrelerile modeli tekrar kurunuz.

# Logistic Regression

log_model.get_params()
log_modelparams = {"max_iter": [50, 100, 250, 1000],
             "penalty": ["l1", "l2", "elasticnet", "none"],
             "tol": [0.00001, 0.0001, 0.001]}
log_model_bestgrid = GridSearchCV(log_model, log_modelparams, cv=5, n_jobs=-1, verbose=2).fit(X, y)

log_model_bestgrid.best_params_
# {'max_iter': 100, 'penalty': 'l1', 'tol': 0.001}

log_model_final = log_model.set_params(**log_model_bestgrid.best_params_).fit(X,y)
log_results = cross_validate(log_model_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1","roc_auc","precision","recall"])
log_results['test_accuracy'].mean() # 0.8051973111168463
log_results["test_f1"].mean() # 0.6015566898542817
log_results['test_roc_auc'].mean()  # 0.8452093406991331
log_results['test_recall'].mean()  # 0.5543131998107554
log_results['test_precision'].mean() # 0.6577250419363587

# GBM

gbm_model.get_params()

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_
# {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.5}
gbm_model_final = gbm_model.set_params(**gbm_best_grid.best_params_).fit(X,y)
gbm_results = cross_validate(gbm_model_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1","roc_auc","precision","recall"])

gbm_results['test_accuracy'].mean() # 0.8061910244209305
gbm_results["test_f1"].mean() # 0.589261953264261
gbm_results['test_roc_auc'].mean()  # 0.8477943338367695
gbm_results['test_recall'].mean()  # 0.5238118449914697
gbm_results['test_precision'].mean() # 0.6734813238404648

# LightGBM

lgbm_model.get_params()
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [100, 200, 300, 350],
               "colsample_bytree": [0.9, 0.8, 1]}
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_
# {'colsample_bytree': 0.9, 'learning_rate': 0.01, 'n_estimators': 300}
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X, y)

lgbm_results = cross_validate(lgbm_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1","roc_auc","precision","recall"])
lgbm_results['test_accuracy'].mean() # 0.803635516807536
lgbm_results["test_f1"].mean() # 0.5736315395099871
lgbm_results['test_roc_auc'].mean()  # 0.8451521925159117
lgbm_results['test_recall'].mean()  # 0.49759286605209957
lgbm_results['test_precision'].mean() # 0.6778456890882039

# CatBoost

catboost_model.get_params()

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
# {'depth': 6, 'iterations': 500, 'learning_rate': 0.01}
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_).fit(X, y)

catboost_results = cross_validate(catboost_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1","roc_auc","precision","recall"])

catboost_results['test_accuracy'].mean() # 0.8059073367636623
catboost_results['test_f1'].mean() # 0.5839397648228942
catboost_results['test_roc_auc'].mean() # 0.8485990468230069
catboost_results['test_recall'].mean() # 0.5131123568120888
catboost_results['test_precision'].mean() # 0.6777152228484864

# Adım 3 : Modele en çok etki eden değişkenleri gösteriniz ve
# önem sırasına göre kendi belirlediğiniz kriterlerde değişken
# seçimi yapıp seçtiğiniz değişkenler ile modeli tekrar
# çalıştırıp bir önceki model skoru arasındaki farkı gözlemleyiniz.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(gbm_model_final, X)

importance_cols = X.columns[gbm_model_final.feature_importances_ > 0.045]
X_new = X[importance_cols]

gbm_new_final = gbm_model.set_params(**{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.5}).fit(X_new,y)
gbm_new_results = cross_validate(gbm_new_final,
                            X_new, y,
                            cv=5,
                            scoring=["accuracy", "f1","roc_auc","precision","recall"])

gbm_new_results['test_accuracy'].mean() # 0.7979563197625652
gbm_new_results["test_f1"].mean() # 0.5659768770354925
gbm_new_results['test_roc_auc'].mean()  # 0.8364531203364203
gbm_new_results['test_recall'].mean()  # 0.49706527504982007
gbm_new_results['test_precision'].mean() # 0.6575784061316436

# BONUS

# Random Oversampling

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')
X_random, y_random = oversample.fit_resample(X, y)

gbm_oversample = gbm_model.set_params(**{'learning_rate': 0.01,  'max_depth': 3,
                                 'min_samples_split': 10, 'n_estimators': 1000}).fit(X_random, y_random)

gbm_oversample_results = cross_validate(gbm_oversample, X_random, y_random, cv=5, n_jobs=-1,
                                      scoring=["accuracy", "f1", "roc_auc", "recall", "precision"], verbose=True)
gbm_oversample_results['test_accuracy'].mean() # 0.7870142405839129
gbm_oversample_results['test_f1'].mean()  # 0.7961127861904391
gbm_oversample_results['test_roc_auc'].mean()  # 0.8655530352572276
gbm_oversample_results['test_recall'].mean()  # 0.831660546258141
gbm_oversample_results['test_precision'].mean()  # 0.7636179092146336

# Dengesiz veri problemi için random oversampling metodu kullanılmıştır. Bu metod ile
# azınlık sınıf olan 1 sınıfından rastgele seçilen örneklerin çoğaltılmasıyla
# veri seti dengelenmiştir. Sonuçlara bakıldığında accuracy değeri az miktarda düşmüş
# fakat diğer tüm metriklerde artış gözlenmiştir.

