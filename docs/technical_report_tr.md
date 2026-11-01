### 1. Projenin temel amacı
Projemizin çözmeye çalıştığı problem şu:

Müşterinin doğal dilde ifade ettiği ürün beklentilerini, ürün konfigürasyon sisteminin anlayabileceği yapılandırılmış gereksinimlere dönüştürmek ve mevcut ürün portföyündeki uygun ürünleri belirlemek.

Bunun için iki farklı yaklaşımı birleştirdik:
- Makine öğrenmesi: Müşteri talebini yorumlamak.
- Deterministik algoritma: Yorumlanan gereksinimlere göre ürünleri eşleştirmek.
Bu ayrım çok önemli. Ürün eşleştirmesini yapay zekâya bırakmadık. Yapay zekâyı, müşteri dilini teknik gereksinimlere dönüştürmek için kullandık.
### 2. Neden iki NLP modeli kullandık?
Müşteri talebini anlamak aslında iki ayrı problem içeriyor.

NLP 1 — Müşteri ne istiyor?

Örneğin:

I need a 45 cm steam oven.

Bu cümleden şu bilgiler çıkarılıyor:
Karakteristik	Değer
OVEN_TYPE	STEAM
WIDTH_CM	45


Bu, çok etiketli sınıflandırma (multi-label classification) problemidir. Çünkü aynı müşteri cümlesinde birden fazla ürün özelliği bulunabilir.

NLP 1 modelimizde TF-IDF ve Linear SVM kullandık.

NLP 2 — Müşteri bu özelliği ne kadar zorunlu görüyor?

Örneğin:

I definitely need a steam oven, but I would prefer a black control panel.

Burada iki farklı öncelik var:
Gereksinim	Öncelik
Steam oven	MUST
Black control panel	PREFERRED


NLP 2 modelimizde TF-IDF ve Logistic Regression kullandık.

Bu iki görevi ayrı modellerle çözmemiz, her birinin başarısını bağımsız olarak ölçmemizi sağladı.
### 3. Modellerimiz ne kadar başarılı?
NLP 1 — Gereksinim çıkarma
Metrik	Sonuç
Precision	%94,16
Recall	%91,86
Micro F1	%92,99
Macro F1	%91,30
Exact Match	%64,77


Burada iki ölçümü özellikle anlamalısın.

Micro F1: Modelin bütün karakteristik–değer tahminleri genelinde ne kadar başarılı olduğunu gösterir.

Exact Match: Bir müşteri talebindeki bütün etiketlerin eksiksiz ve hatasız tahmin edilmesini gerektirir.

Dolayısıyla %92,99 Micro F1 elde etmemize rağmen Exact Match'in %64,77 olması çelişki değildir.

NLP 2 — Öncelik sınıflandırma
Metrik	Sonuç
Accuracy	%63,48
Macro F1	%60,73


NLP 2, projemizin geliştirmeye en açık bileşenidir.

MUST ve PREFERRED ifadeleri, yalnızca belirli kelimelere değil, cümlenin bağlamına da bağlıdır. TF-IDF temelli modelimiz bu bağlamı her zaman doğru yorumlayamıyor.

Bu nedenle uygulamaya kullanıcı onayı ekledik.
### 4. Ürün eşleştirme algoritmamız güvenilir mi?
Burada iki farklı güvenilirlik kavramını ayırmamız gerekiyor.

Birincisi: Doğru gereksinimler verildiğinde algoritma doğru ürünleri buluyor mu?

1.967 talep üzerinde yaptığımız bağımsız karşılaştırmada eşleştirme ve tercih puanlama sonuçları arasında tutarsızlık bulmadık.

Bu, eşleştirme kodumuzun test edilen veri ve kurallar kapsamında tutarlı çalıştığını gösteriyor.

İkincisi: NLP modelleri doğru gereksinimleri üretiyor mu?

Burada hatalarımız var.

298 test talebinde elde ettiğimiz uçtan uca sonuçlar:
Metrik	Sonuç
Product Set Exact Match	%31,54
Eligibility Precision	%27,58
Eligibility Recall	%86,57
Availability Agreement	%95,64


Projemizin en önemli bulgusu burada:

Bir sistemin NLP bileşeninde yüksek F1 başarısı elde etmesi, nihai ürün önerilerinin de aynı derecede başarılı olacağı anlamına gelmiyor.

Özellikle MUST/PREFERRED hataları, uygun ürün kümesini önemli ölçüde değiştirebiliyor.

Bu nedenle uygulamamız henüz tamamen otonom bir ürün öneri sistemi olarak değerlendirilmemeli.
### 5. LLM kullanımı projemizde nerede?
Bu konuyu eğitmene açıklamamız önemli.

Projemizde LLM desteğini sentetik müşteri taleplerinin hazırlanması ve dilsel çeşitliliğinin artırılması aşamasında kullandık.

Ancak son uygulamada çalışan tahmin modelleri LLM değil:
- NLP 1: TF-IDF + Linear SVM
- NLP 2: TF-IDF + Logistic Regression
Bu bilinçli bir mimari ayrım olarak sunulabilir; ancak LLM tabanlı bir tahmin modeliyle karşılaştırmalı deney yapmadığımızı da belirtmeliyiz.

Dolayısıyla LLM'nin daha iyi veya daha kötü sonuç vereceğini henüz bilmiyoruz.
## 6. Model Seçimleri ve Teknik Gerekçeler

### 6.1. Neden TF-IDF kullandık?

TF-IDF (Term Frequency–Inverse Document Frequency), metinleri makine öğrenmesi algoritmalarının işleyebileceği sayısal vektörlere dönüştürür.

Bu yöntem, kelimelerin bir metindeki önemini değerlendirirken bütün veri setindeki yaygınlıklarını da dikkate alır.

Projemizde TF-IDF tercih etmemizin nedenleri:

- Geleneksel makine öğrenmesi modelleriyle kullanılabilmesi.
- Görece düşük hesaplama maliyeti.
- Küçük ve orta ölçekli metin veri setleri için uygun bir başlangıç yöntemi olması.
- Modellerin yerel ortamda çalıştırılabilmesi.

Ancak TF-IDF, kelimelerin bağlamsal anlamlarını doğrudan modellemez. Bu durum özellikle gereksinim önceliklerinin belirlenmesinde sınırlama oluşturur.

### 6.2. Neden Linear SVM kullandık?

NLP 1 problemimizde müşteri taleplerinden birden fazla karakteristik–değer etiketi çıkarılması gerekiyor.

Bu nedenle problemimizi çok etiketli sınıflandırma olarak ele aldık.

Linear SVM, TF-IDF ile oluşturulan yüksek boyutlu metin vektörleri üzerinde kullanılabilen bir sınıflandırma algoritmasıdır.

Modelimizi One-vs-Rest yaklaşımıyla kullanarak her karakteristik–değer etiketi için ayrı bir karar mekanizması oluşturduk.

Karar eşiğini -0,3 olarak belirledik.

Bu eşik, modelin hangi etiketleri tahmin olarak kabul edeceğini etkiler. Eşiğin değiştirilmesi precision ve recall arasındaki dengeyi değiştirebilir.

### 6.3. Neden Logistic Regression kullandık?

NLP 2 problemimizde her gereksinimin MUST veya PREFERRED olarak sınıflandırılması gerekiyor.

Bu nedenle ikili sınıflandırma yaklaşımını kullandık.

Logistic Regression, metin sınıflandırma problemlerinde kullanılabilen, görece basit ve yorumlanabilir bir temel modeldir.

Ancak deney sonuçlarımız, mevcut yaklaşımın gereksinim önceliklerini yeterli güvenilirlikle belirleyemediğini gösterdi.

Özellikle aynı müşteri talebinde hem zorunlu hem tercih edilen özellikler bulunması, modelin doğru karar vermesini zorlaştırıyor.

## 7. Hata Analizi

Projemizdeki temel hata kaynakları üç grupta değerlendirilebilir.

### 7.1. Gereksinim çıkarma hataları

NLP 1 bazı müşteri taleplerinde:

- Gerçekte belirtilen bir gereksinimi çıkaramayabilir.
- Metinde bulunmayan bir gereksinimi tahmin edebilir.
- Karakteristik veya değer kodunu yanlış belirleyebilir.

Bu hatalar, ürün eşleştirme algoritmasının değerlendirdiği koşulları doğrudan değiştirir.

### 7.2. Öncelik sınıflandırma hataları

NLP 2'nin MUST ve PREFERRED tahminlerinde hata yapması, ürün eşleştirme sonucunu önemli ölçüde etkileyebilir.

Örneğin gerçekte PREFERRED olan bir gereksinimin MUST olarak sınıflandırılması, bazı uygun alternatiflerin gereksiz yere elenmesine neden olabilir.

Tersi durumda ise gerçekte zorunlu olan bir koşulu karşılamayan ürünler önerilebilir.

Bu nedenle öncelik tahmini, sistemin kritik geliştirme alanıdır.

### 7.3. Uçtan uca hata birikimi

Ürün eşleştirme algoritmamız yapılandırılmış gereksinimlere göre çalışır.

Dolayısıyla algoritma kendi kurallarını doğru uygulasa bile yanlış yapılandırılmış gereksinimler yanlış ürün sonuçları oluşturabilir.

Test sonuçlarında NLP 1'in Micro F1 değeri %92,99 iken ürün kümesi Exact Match değerinin %31,54 olması bu etkiyi göstermektedir.

## 8. Kullanıcı Kontrolünün Önemi

Geliştirdiğimiz Streamlit uygulaması, NLP modellerinin tahminlerini doğrudan nihai karar olarak kullanmaz.

Kullanıcıya tahmin edilen gereksinimleri gösterir ve mevcut arayüzün desteklediği alanlarda düzeltme yapma imkânı verir.

Ürün eşleştirme işlemi kullanıcı onayından sonra gerçekleştirilir.

Bu yaklaşım, insan denetimli bir karar destek sistemi olarak değerlendirilebilir.

Ancak mevcut uygulamada karakteristik ve değer kodlarının doğrudan düzenlenememesi önemli bir sınırlamadır.

## 9. Projenin Kapsamı ve Sınırlamaları

Bu proje akademik amaçlı bir prototiptir.

Çalışmanın başlıca sınırlamaları:

1. Ürün ve müşteri talebi verilerinin sentetik olması.
2. Gerçek müşteri talepleriyle doğrulama yapılmamış olması.
3. NLP 2 öncelik sınıflandırma başarısının sınırlı olması.
4. Mevcut ürün portföyünün 500 ürünle sınırlı olması.
5. Yeni ürün konfigürasyonu oluşturma özelliğinin bulunmaması.
6. Kurumsal ürün konfigürasyon sistemleriyle entegrasyon yapılmamış olması.

Bu sınırlamalar, projenin başarısız olduğu anlamına gelmez. Prototipin hangi koşullarda değerlendirildiğini ve sonuçların hangi kapsamda geçerli olduğunu açıklar.

## 10. Gelecek Çalışmalar

Projenin sonraki aşamalarında aşağıdaki geliştirmeler değerlendirilebilir:

- LLM tabanlı gereksinim çıkarma yöntemleriyle karşılaştırmalı deneyler.
- MUST/PREFERRED sınıflandırmasının bağlamı daha iyi anlayan modellerle geliştirilmesi.
- Karakteristik ve değer tahminlerinin kullanıcı tarafından düzenlenebilmesi.
- Çelişkili müşteri taleplerinin tespit edilmesi.
- Hiçbir ürün eşleşmediğinde alternatif konfigürasyon önerilmesi.
- Mevcut bir ürünün müşteri taleplerine göre yeniden yapılandırılması.
- Gerçek müşteri talepleriyle performans değerlendirmesi.
- Kurumsal ürün konfigürasyon sistemleriyle entegrasyon.

## 11. Genel Sonuç

Bu projede, doğal dilde ifade edilen müşteri taleplerinin makine öğrenmesiyle yapılandırılmış ürün gereksinimlerine dönüştürülebileceğini ve bu gereksinimlerin deterministik bir ürün eşleştirme algoritmasında kullanılabileceğini gösteren çalışan bir prototip geliştirdik.

Çalışmanın önemli sonuçlarından biri, gereksinim çıkarma başarısının yüksek olmasının tek başına doğru ürün önerileri için yeterli olmadığının görülmesidir.

Özellikle gereksinimlerin zorunluluk derecesinin doğru belirlenmesi, ürün öneri sisteminin performansında kritik rol oynamaktadır.

Sonuç olarak proje, yapay zekâ destekli ürün konfigürasyonu alanında geliştirilebilecek daha kapsamlı uygulamalar için bir başlangıç noktası oluşturmaktadır.