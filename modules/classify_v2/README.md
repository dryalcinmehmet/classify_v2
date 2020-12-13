# Nasıl Çalıştırılır?
***
* Kütüphanelerin yüklenmesi için aşağıdaki komut çalıştırılır.

`
pip install -r requirements.txt
`
***

* Ön işleme adımları, model kaydetme tercihleri ve dosya yolları **'param_config.py'** dosyası içindeki parametreler değiştirilerek belirlenmektedir.

***
* Eğer, [`Zemberek`](https://github.com/cbilgili/zemberek-nlp-server) kullanılacaksa,
```
docker pull cbilgili/zemberek-nlp-server
docker run -p 4567:4567 -d cbilgili/zemberek-nlp-server
```
komutları çalıştırılarak docker ayağa kaldırılır.

***

### İçerik

* clana_.py: 'Clana' fonksiyonuyla sınıflar ve karışıklık matrisi dosyaya kaydedilerek clana'ya beslenmektedir. Çıkarılan sınıflar görselleştirilecek sınıflardır.

* **classify.py:** Ana fonksiyondur. _Train_ ve _Predict_ burada yapılmaktadır. 

_'Train'_ fonksiyonuyla data/datanın sırasıyla; preprocess edilir, train-test olarak ayrılır, tf-idf özellikleri çıkarılır, eğitilir, performans metrikleri çıkarılır. Sınıflar ve karışıklık matrisi dosyaya kaydedilip Clana'ya beslenir. Clana tarafından çıkarılan sınıflarla yeni bir karışıklık matrisi elde edilir. Son olarak, model ve tf-idf vektörü kaydedilir.

_'Predict'_ fonksiyonuyla, sırasıyla; data preprocess edilir, kaydedilen tf-idf vektörü okunarak özellikleri çıkarılır. Kaydedilen model okunarak datanın sınıfı tahmin edilir. Ayrıca, diğer sınıflarda da bulunma olasılıkları büyükten küçüğe doğru sıralanır.

* _grid_search.py : Şu an, geliştirmeye dahil değildir._

* _intent_keywords.py : Şu an, geliştirmeye dahil değildir._

* performance_results.py : Model performans metrikleri 'get_performance_results' fonksiyonuyla 'cls_reports' sözlüğüne yazılmaktadır.

* preprocess.py : 'Cleaner/cleaner.py' dosyasındaki preprocess fonksiyonları, 'param_config' dosyasındaki adımlara göre, 'preprocess.py' tarafından 'Preprocessor' fonksiyonuyla çağırılmaktadır.

* vectorization.py: 'train_vectorizer' fonksiyonuyla metinlerin Tf-idf özellikleri çıkarılmaktadır. 'predict_vectorizer' fonksiyonu ise predict aşamasında çağırılmaktadır.

***

### Clana

Sınıf sayısının çok fazla olduğu durumlarda oluşan karışıklık matrisi görüntüleme problemi için [`Clana`](https://github.com/MartinThoma/clana) kütüphanesinden yararlanılmıştır. Birbirine en çok benzeyen bu yüzden birbirine en çok karışan sınıfları saptamaktadır. Böylece, daha optimal bir matris elde edilmektedir. Clana kütüphanesi'nin 4.0 sürümü üzerinde geliştirmeler yapılarak 'clana/' olarak koda gömülmüştür. 

* Log ve görselleştirme ayarları, **'config.yaml'** dosyasından yapılmaktadır. 'disable_existing_loggers=true' ise çıktılar log dosyalarına kaydedilmez, terminale basılmaz. (Local)

* 'config.yaml' dosya yol ayarları 'utils.load_config' fonksiyonundan yapılmaktadır.

* Loglar, 'logs/clana.error.log' ve 'logs/clana.info.log' dosyası olarak tutulmaktadır.

* Clana tarafından çıkarılan dosyalar 'save/' dosyasına basılmaktadır.

* 'main.py' içerisindeki 'clanaMain' fonksiyonu Clana ana fonksiyonudur.İçerisine;

    -- Karışıklık matrisi(cm_),
    
    -- Model adı(cm_name),
    
    -- İterasyon sayısı(steps),
    
    -- Sınıf listesi(labels_),
    
    -- Dosya kayıt parametreleri(save_cm_plot, save_score_plot, save_hierarchy)

    alır. Clana tarafından çıkarılan sınıfları liste olarak döndürür.

* 'save/.clana' dosyası modele ve modelin çalışma tekrarına göre dosyadan okunup değiştirilmektedir. (DB'de tutulması gerekiyor)

* Her defasında çağırılan bazı fonksiyonlar, dosya kayıt parametreleri, config dosyası ve log şablonu 'conf.py' dosyasından okunmaktadır.

* 


























