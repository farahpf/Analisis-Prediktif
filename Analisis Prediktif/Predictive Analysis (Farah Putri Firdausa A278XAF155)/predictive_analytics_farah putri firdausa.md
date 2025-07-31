# Predictive Analytics Project - ESG Company Classification

## Domain Proyek

Dalam beberapa tahun terakhir, isu keberlanjutan semakin menjadi perhatian utama dalam dunia bisnis dan investasi. Perusahaan dituntut tidak hanya untuk mencetak keuntungan, tetapi juga untuk beroperasi secara bertanggung jawab terhadap lingkungan, sosial, dan tata kelola perusahaan. Oleh karena itu, ESG (Environmental, Social, Governance) menjadi indikator penting dalam menilai keberlanjutan perusahaan.

Namun, mengidentifikasi perusahaan dengan performa ESG yang baik membutuhkan proses evaluasi yang kompleks dan tidak selalu efisien jika dilakukan secara manual. Di sinilah machine learning dapat berperan: dengan memanfaatkan data ESG dan keuangan, kita dapat membangun model prediktif untuk mengklasifikasikan apakah sebuah perusahaan tergolong berkelanjutan atau tidak.

Model prediktif semacam ini dapat menjadi alat bantu bagi investor, analis keuangan, dan lembaga pemerintahan dalam pengambilan keputusan strategis, seperti memilih portofolio investasi ramah lingkungan atau menyusun kebijakan keberlanjutan.

Studi ini mengambil data perusahaan yang memuat skor ESG serta indikator keuangan untuk membangun model klasifikasi yang dapat mengidentifikasi perusahaan dengan skor ESG tinggi.

> Referensi:
> - World Economic Forum. (2022). Why ESG is the future of investing.
> - MSCI. (2021). ESG Ratings Methodology.

## Problem Statement
Dalam dunia bisnis modern, keberlanjutan menjadi faktor penting dalam menilai kualitas dan tanggung jawab sebuah perusahaan. Proyek ini bertujuan untuk memprediksi apakah sebuah perusahaan tergolong **berkelanjutan** atau tidak, berdasarkan skor ESG (Environmental, Social, Governance). Target klasifikasi ditentukan dari nilai `ESG_Overall > 60`.

## Goal
Membangun model machine learning untuk memprediksi status keberlanjutan (`HighESG`) sebuah perusahaan berdasarkan fitur finansial, lingkungan, dan kategori wilayah serta industri.

## Solution Statement

Untuk mencapai tujuan klasifikasi status keberlanjutan (HighESG), proyek ini mengusulkan dua pendekatan solusi:

1. **Random Forest Classifier**  
   Algoritma ensemble berbasis pohon yang mampu menangani data numerik dan kategorikal dengan baik, serta memberikan interpretasi melalui feature importance.

2. **XGBoost Classifier**  
   Algoritma boosting yang dikenal efisien dan memiliki performa tinggi pada berbagai permasalahan klasifikasi, terutama pada data non-linear.

Evaluasi kedua model dilakukan menggunakan metrik: **precision**, **recall**, **f1-score**, dan **ROC AUC**.  
Model dengan performa terbaik akan dipilih sebagai solusi utama dan dapat digunakan sebagai sistem pendukung keputusan untuk evaluasi ESG perusahaan.

## Data Understanding
### Deskripsi Dataset 
Dataset yang digunakan berisi informasi terkait nilai ESG (Environmental, Social, Governance) dan indikator keuangan dari berbagai perusahaan. Dataset ini memiliki **11.000 entri (baris)** dan **16 kolom (fitur)**. Dataset bersumber dari Kaggle dan dapat diakses publik melalui tautan berikut:

> [Kaggle - ESG & Financial Data for Companies]
https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset  

### Fitur dalam Dataset

| Fitur              | Deskripsi                                                                 | Status        |
|-------------------|---------------------------------------------------------------------------|---------------|
| CompanyID         | ID unik perusahaan                                                        | Dihapus       |
| CompanyName       | Nama perusahaan                                                           | Dihapus       |
| Industry          | Sektor industri perusahaan                                                | Digunakan     |
| Region            | Wilayah operasional perusahaan                                            | Digunakan     |
| Year              | Tahun data dicatat                                                        | Dihapus       |
| Revenue           | Pendapatan tahunan perusahaan (dalam juta USD)                            | Digunakan     |
| ProfitMargin      | Margin laba bersih (%)                                                    | Digunakan     |
| MarketCap         | Kapitalisasi pasar (dalam juta USD)                                       | Digunakan     |
| GrowthRate        | Laju pertumbuhan pendapatan (%)                                           | Digunakan     |
| ESG_Overall       | Skor ESG keseluruhan (0–100)                                              | Dihapus       |
| ESG_Environmental | Skor ESG pilar lingkungan                                                 | Digunakan     |
| ESG_Social        | Skor ESG pilar sosial                                                     | Digunakan     |
| ESG_Governance    | Skor ESG pilar tata kelola                                                | Digunakan     |
| CarbonEmissions   | Emisi karbon tahunan (ton)                                                | Digunakan     |
| WaterUsage        | Penggunaan air (meter kubik)                                              | Digunakan     |
| EnergyConsumption | Konsumsi energi tahunan (kilowatt-jam)                                    | Digunakan     |

### Kondisi Data

- Dataset tidak memiliki nilai kosong (`missing value`) pada fitur utama.
- Tidak ditemukan duplikasi berdasarkan `CompanyID` dan `Year`.
- Kolom `ESG_Overall` digunakan sebagai dasar pembentukan target klasifikasi (`HighESG`) di tahap data preparation.
- Terdapat perbedaan skala signifikan antar fitur numerik (misal `MarketCap`, `CarbonEmissions`) → perlu dilakukan skaling.
- Variabel kategorikal seperti `Industry` dan `Region` perlu di-*encode* agar bisa diproses oleh model.

### Visualisasi Awal

Distribusi nilai `ESG_Overall` menunjukkan bahwa mayoritas perusahaan memiliki skor antara 50–70.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.histplot(df['ESG_Overall'], bins=30, kde=True, color='skyblue')
plt.title('Distribusi ESG Overall Score')
plt.xlabel('Skor ESG')
plt.ylabel('Jumlah Perusahaan')
plt.show()

### Struktur Data

Fitur-fitur utama dalam dataset ini meliputi:
- **Finansial**: `Revenue`, `ProfitMargin`, `MarketCap`, `GrowthRate`
- **ESG**: `ESG_Environmental`, `ESG_Social`, `ESG_Governance`, `ESG_Overall`
- **Lingkungan**: `CarbonEmissions`, `WaterUsage`, `EnergyConsumption`
- **Kategori**: `Industry`, `Region`
- **Identitas**: `CompanyID`, `CompanyName`, `Year`
### Pemeriksaan Kualitas Data

- **Missing Value**:
  - Kolom `GrowthRate` memiliki 1.000 missing value (≈9% dari data).
  - Kolom lainnya tidak memiliki nilai kosong.

- **Data Duplikat**:
  - Tidak ditemukan baris duplikat dalam dataset (`df.duplicated().sum() = 0`).

- **Outlier**:
  - Outlier terdeteksi terutama pada kolom numerik seperti `Revenue`, dengan nilai maksimum jauh di atas kuartil ke-3 (Q3), mengindikasikan keberadaan outlier ekstrim.
  - Deteksi dilakukan menggunakan metode IQR dan divisualisasikan melalui boxplot.

Contoh visualisasi boxplot pada kolom `Revenue`:
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Revenue'])
plt.title("Boxplot Revenue")
plt.show()

## Data Preparation

Tahapan data preparation dilakukan untuk membersihkan dan menyiapkan data agar sesuai dengan kebutuhan model machine learning.

### 1. Penghapusan Kolom Tidak Relevan
Kolom `CompanyID`, `CompanyName`, dan `Year` dihapus karena tidak memberikan kontribusi prediktif terhadap target dan berpotensi menjadi sumber bias atau kebocoran data.

```python
df.drop(columns=['CompanyID', 'CompanyName', 'Year'], inplace=True)

### 2. Membuat Target Klasifikasi 
Dibuat kolom target ***HighESG*** untuk klasifikasi biner, berdasarkan nilai ***ESG_Overall***. Perusahaan dianggap memiliki skor ESG tinggi jika nilai diatas 60 
df['HighESG'] = (df['ESG_Overall'] >= 60).astype(int)
df.drop(columns='ESG_Overall', inplace=True)
Alasan pemilihan ambang batas 60 adalah karena mayoritas distribusi ESG berkisar antara 50-70, sehingga treshold ini merepresentasikan kelompok perusahaan dengan skor di atas rata-rata 

### 3. Penanganan Fitur Kategorikal
Fitur ***Industry*** dan Region diencoding menggunakan OneHotEncoder agar dapat digunakan dalam model berbasis numerik.
```python
from sklearn.preprocessing import OneHotEncoder

categorical_features = ['Industry', 'Region']
encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(df[categorical_features])

### 4. Skaling Fitur Numerik 
Karena terdapat perbedaan skala yang besar antar fitur numerik (misalnya ***MarketCap***, ***CarbonEmissions***), maka digunakan StandardScaler untuk normalisasi.

```python
from sklearn.preprocessing import StandardScaler

numerical_features = ['Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate',
                      'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
                      'CarbonEmissions', 'WaterUsage', 'EnergyConsumption']
scaler = StandardScaler()
scaled = scaler.fit_transform(df[numerical_features])

### 5.Pembagian Data Latih dan Uji 
Dataset dibagi menjadi data latih dan uji dengan proporsi 80:20 menggunakan ***train_test_split***. Digunakan parameter ***stratify=y*** untuk memastikan distribusi kelas target seimbang.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

### 6. Outlier Removal

Outlier dapat menyebabkan bias dalam pelatihan model, terutama pada fitur numerik seperti `Revenue` dan `ProfitMargin` yang memiliki nilai ekstrem.Untuk mengurangi bias model akibat nilai ekstrim, dilakukan penghapusan outlier pada kolom `ProfitMargin` dan `Revenue` menggunakan metode IQR (Interquartile Range). Outlier berpotensi mendistorsi distribusi data dan mempengaruhi akurasi prediksi model.

Metode IQR (Interquartile Range) digunakan untuk mendeteksi dan menghapus outlier dari dua fitur tersebut. Langkah ini bertujuan untuk mengurangi noise dalam data dan meningkatkan generalisasi model.

Berikut jumlah data sebelum dan sesudah penghapusan outlier :
- Sebelum penghapusan outlier: 11.000 entri
- Setelah penghapusan outlier: 10.620 entri
- Total data yang dihapus: 600 (≈5.5%)

```python
# Contoh penghapusan outlier pada ProfitMargin
Q1 = df['ProfitMargin'].quantile(0.25)
Q3 = df['ProfitMargin'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['ProfitMargin'] >= Q1 - 1.5 * IQR) & (df['ProfitMargin'] <= Q3 + 1.5 * IQR)]

Langkah ini bertujuan agar model lebih stabil dan generalisasi terhadap data baru dapat meningkat.


## Modeling
Pada tahap ini, dilakukan pelatihan model machine learning untuk memprediksi apakah sebuah perusahaan termasuk ke dalam kategori ESG tinggi (`HighESG = 1`) atau tidak (`HighESG = 0`). Dua model yang digunakan adalah:

### 1. Random Forest Classifier

**Random Forest** adalah algoritma ensembel yang membangun banyak pohon keputusan (decision tree) secara acak, kemudian menggabungkan hasil prediksi mereka untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dilatih pada subset data dan subset fitur yang berbeda.

Model Random Forest dibangun dengan parameter utama berikut:
- `n_estimators=100`: jumlah pohon keputusan yang dibangun dalam ensemble. Semakin banyak pohon, semakin stabil prediksi yang dihasilkan, meskipun dengan peningkatan waktu komputasi.

- `max_depth=None`: tidak membatasi kedalaman pohon, sehingga pohon dapat tumbuh sampai seluruh daun bersifat murni. Jika tidak diatur, model bisa mengalami overfitting, terutama pada data dengan banyak fitur.

- `class_weight='class_weight_dict'`: memberikan bobot berbeda untuk masing-masing kelas agar model tidak bias terhadap kelas mayoritas. Ini penting untuk menangani data yang tidak seimbang (imbalance).

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,
                            max_depth=None,
                            class_weight=class_weight_dict,
                            random_state=42)
rf.fit(X_train, y_train)

Alasan pemilihan Random Forest :
- Tahan terhadap outlier dan data yang tidak terdistribusi normal.
- Memiliki interpretabilitas fitur penting melalui feature importance 

### 1. XGBoost Classifier

**XGBoost (Extreme Gradient Boosting)** adalah algoritma boosting yang membangun model secara bertahap. Setiap model baru fokus untuk memperbaiki kesalahan dari model sebelumnya. Teknik ini sangat powerful dan telah terbukti memberikan hasil terbaik di berbagai kompetisi ML.

Parameter utama:
- `n_estimators=100`: jumlah boosting round
- `max_depth=6`: kedalaman pohon, dikontrol agar tidak overfitting 

- `learning_rate= 0.1`: ukuran langkah pembaruan model.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.1,
                    max_depth=6,
                    n_estimators=100,
                    use_label_encoder=False,
                    eval_metric='logloss')
xgb.fit(X_train, y_train)


Alasan pemilihan XGBoost:
- Lebih efisien dibanding model boosting biasa. 
- Mendukung regularisasi untuk mencegah overfitting 
Kinerja tinggi dalam menangani data klasifikasi tabular.

## Evaluation Metrics Explanation
Berikut adalah metrik evaluasi yang digunakan beserta formula dan alasannya:

- **Precision**:  
  $\text{Precision} = \frac{TP}{TP + FP}$  
  Mengukur seberapa akurat prediksi model terhadap kelas positif. Penting untuk meminimalkan false positive, khususnya pada investasi berkelanjutan yang salah klasifikasi bisa berdampak besar.

- **Recall**:  
  $\text{Recall} = \frac{TP}{TP + FN}$  
  Mengukur seberapa banyak kasus positif yang berhasil ditemukan oleh model. Dalam konteks ESG, recall tinggi membantu memastikan perusahaan yang benar-benar berkelanjutan tidak terlewatkan.

- **F1-Score**:  
  $\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$  
  Rata-rata harmonis dari precision dan recall. Berguna saat diperlukan keseimbangan antara keduanya.

- **ROC AUC Score**:  
  Mengukur kemampuan model dalam membedakan kelas 0 dan 1 pada berbagai threshold. Semakin tinggi nilainya (maksimum 1), semakin baik model dalam mengklasifikasikan dua kelas.

Dipilihnya metrik-metrik ini karena permasalahan ESG melibatkan klasifikasi biner dengan potensi ketidakseimbangan kelas, sehingga dibutuhkan pengukuran yang lebih dari sekadar akurasi.

## Evaluation

Untuk mengukur performa model klasifikasi yang dibangun, digunakan beberapa metrik evaluasi yang sesuai dengan konteks klasifikasi biner, yaitu:

- **Accuracy**: proporsi prediksi yang benar dari seluruh prediksi.
- **Precision**: ketepatan model saat memprediksi kelas positif.
- **Recall**: kemampuan model menangkap semua instance positif yang sebenarnya.
- **F1-Score**: harmonic mean dari precision dan recall.
- **ROC AUC Score**: kemampuan model membedakan antara kelas positif dan negatif secara keseluruhan.

### Hasil Evaluasi Random Forest

| Metrik       | Nilai   |
|--------------|---------|
| Accuracy     | 0.97    |
| Precision    | 0.96    |
| Recall       | 0.96    |
| F1-Score     | 0.96    |
| ROC AUC      | 0.997   |

### Hasil Evaluasi XGBoost

| Metrik       | Nilai   |
|--------------|---------|
| Accuracy     | 0.98    |
| Precision    | 0.98    |
| Recall       | 0.97    |
| F1-Score     | 0.97    |
| ROC AUC      | 0.999   |

### Interpretasi

Model XGBoost menunjukkan performa yang lebih unggul dibanding Random Forest dalam semua metrik, terutama pada ROC AUC sebesar 0.999 yang menandakan kemampuan pemisahan kelas sangat baik. Hal ini berarti model sangat efektif dalam mengidentifikasi perusahaan dengan skor ESG tinggi dan rendah.

Tingkat precision dan recall yang seimbang menunjukkan bahwa model tidak hanya akurat, tetapi juga stabil dalam menghadapi ketidakseimbangan kelas. Hal ini penting bagi pengguna seperti analis ESG atau investor karena mereka ingin menghindari kesalahan dalam mengklasifikasikan perusahaan yang sebenarnya tidak layak.

XGBoost dipilih sebagai model terbaik tidak hanya karena performa metriknya, tetapi juga karena keunggulan teknisnya. Model ini memiliki kemampuan untuk menangani missing value secara internal, mendukung regularisasi (L1 dan L2) untuk mencegah overfitting, serta menggabungkan banyak pohon keputusan secara efisien melalui boosting. Keunggulan-keunggulan tersebut membuat XGBoost sangat cocok untuk dataset dengan variabel kompleks seperti yang digunakan dalam proyek ini.

## Conclusion

Model Random Forest dan XGBoost berhasil membedakan perusahaan yang tergolong berkelanjutan (HighESG) berdasarkan kombinasi fitur ESG dan finansial.

Berdasarkan hasil evaluasi:
*   XGBoost mencatat skor tertinggi dengan **Accuracy 0.98, Precision 0.98, F1-Score 0.97,** dan **ROC AUC 0.999**
*   Random Forest menyusul dengan performa yang juga baik (**Accuracy 0.97, F1-Score 0.96,** dan **ROC AUC 0.997**)

Hal ini menunjukkan bahwa kedua model mampu membedakan kelas dengan sangat baik. Namun, **XGBoost lebih unggul** secara keseluruhan. XGBoost cocok digunakan dalam kasus yang memerlukan keseimbangan antara false positive dan false negative.

Dengan performa tersebut, pendekatan machine learning ini efektif untuk mendukung pengambilan keputusan dalam sektor keuangan berkelanjutan dan investasi hijau.


Model dapat dikembangkan lebih lanjut dengan:
- Menambahkan data historis atau eksternal seperti kebijakan regional
- Menerapkan interpretabilitas model (SHAP)
- Menggunakan data real-time untuk sistem rekomendasi ESG


## Referensi
- Dataset : https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset?resource=download 
- https://en.wikipedia.org/wiki/Environmental,_social,_and_corporate_governance

## Closing Statement

Proyek ini menunjukkan bahwa pendekatan machine learning dapat digunakan sebagai alat bantu pengambilan keputusan dalam menilai keberlanjutan perusahaan. Dengan memanfaatkan kombinasi data ESG dan finansial, model yang dibangun mampu memberikan prediksi yang akurat dan dapat diterapkan dalam berbagai konteks seperti investasi hijau, pemeringkatan perusahaan, serta pelaporan keberlanjutan.

Pendekatan ini dapat dikembangkan lebih lanjut dengan menambahkan data historis, eksternal (misalnya kebijakan regional), maupun integrasi dengan sistem rekomendasi untuk mendukung kebijakan perusahaan dan investor masa depan.
