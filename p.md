Instruksi Submission

Submission

Proyek Analisis Sentimen

Pengantar Kriteria Utama Penilaian Ketentuan Berkas Submission Lainnya
Hai! Selamat datang di Proyek Analisis Sentimen! Akhirnya, Diana, Ryan, dan Anda telah mencapai tahap submission pertama. Banyak sekali, ya, ilmu yang sudah didapatkan pada modul-modul sebelumnya? Semoga bermanfaat dan memberikan banyak pengetahuan baru untuk Anda, ya.

Perjalanan belajar yang panjang dan penuh ilmu telah membawa Anda semua menuju tahapan ini. Anda akan menunjukkan pemahaman teknis dalam menerapkan salah satu implementasi dari NLP, yaitu analisis sentimen.

Analisis sentimen, sebagian dari klasifikasi teks, bertujuan untuk mengidentifikasi dan mengevaluasi opini, sikap, atau emosi yang terkandung dalam teks. Topik ini bisa menjadi alat yang sangat berguna baik dalam memahami pandangan pelanggan, tanggapan publik terhadap suatu peristiwa, maupun sentimen pasar terhadap suatu merek atau produk.

Dalam proyek ini, Anda diberikan kebebasan untuk memilih topik atau pembahasan yang ingin dianalisis sentimennya, serta platform yang ingin digunakan sebagai sumber data untuk analisis. Namun, ada persyaratan tertentu yang harus Anda penuhi. 

Pertama, topik yang Anda pilih haruslah sesuai dengan etika dan norma, serta tidak boleh mengandung unsur kontroversial atau isu-isu sensitif yang dapat menimbulkan masalah. Selain itu, Anda harus mematuhi kebijakan dan pedoman dari platform sumber data dan menghindari tindakan yang dapat dianggap sebagai pencurian data.

Dengan memahami pentingnya penghormatan terhadap etika dan kebijakan saat melakukan analisis sentimen, Anda siap untuk melangkah maju dalam proyek ini. Dengan kreativitas dan ketelitian, Anda akan dapat mempelajari data teks dari platform yang dipilih, serta menghasilkan wawasan yang berharga mengenai sentimen dan opini yang terkandung di dalamnya. Mari kita mulai!

Instruksi Submission

Submission

Proyek Analisis Sentimen

Pengantar Kriteria Utama Penilaian Ketentuan Berkas Submission Lainnya
Untuk memenuhi kriteria submission yang telah ditetapkan, beberapa hal harus dipertimbangkan.

Kriteria 1: Data merupakan hasil scraping secara mandiri
Anda diberi kebebasan untuk mengambil data atau scraping menggunakan bahasa pemrograman Python dari berbagai sumber, seperti platform PlayStore, X, Instagram, komentar pada penilaian barang di e-commerce, dan lain-lain. Jumlah dataset minimal yang harus diperoleh adalah 3.000 sampel. 

Kriteria 2: Melakukan tahapan ekstraksi fitur dan pelabelan data
Metode yang digunakan bebas sesuai dengan preferensi masing-masing peserta. Tahapan ini penting untuk mempersiapkan data sehingga dapat diolah lebih lanjut dalam proses pelatihan model.

Kriteria 3: Menggunakan algoritma pelatihan machine learning
Pilihan algoritma pelatihan ini haruslah sesuai dengan tujuan analisis sentimen yang ingin dicapai.

Kriteria 4: Akurasi testing set yang didapatkan minimal harus mencapai 85%
Hal ini menunjukkan bahwa model yang dikembangkan memiliki kinerja yang baik dalam mengklasifikasikan sentimen dari data yang diberikan.

Dengan memperhatikan semua kriteria ini, Anda diharapkan dapat menghasilkan model analisis sentimen yang berkualitas tinggi dan bisa dipertanggungjawabkan.

Instruksi Submission

Submission

Proyek Analisis Sentimen

Pengantar Kriteria Utama Penilaian Ketentuan Berkas Submission Lainnya
Submission Anda akan dinilai oleh reviewer dengan skala 1–5 berdasarkan dari parameter yang ada. Anda dapat menerapkan beberapa saran untuk mendapatkan nilai tinggi. Berikut sarannya.

Menggunakan algoritma deep learning.
Akurasi pada training set dan testing set di atas 92%. 
Dataset yang digunakan untuk melatih model minimal memiliki tiga kelas.
Memiliki jumlah data minimal 10.000 sampel data.
Melakukan 3 percobaan skema pelatihan yang berbeda. Skema ini dapat dibedakan dari variasi algoritma pelatihan, metode ekstraksi fitur, pelabelan, dan pembagian data dengan memilih minimal 2 kombinasi.
Catatan:
Jika Anda tidak menerapkan saran kedua, pastikan ketiga percobaan skema pelatihan yang dilakukan memiliki akurasi testing set minimal 85%. Lalu jika Anda mencoba lebih dari tiga skema pelatihan, pastikan setidaknya ketiga percobaan di antaranya memiliki akurasi testing set minimal 85%.
Jika Anda juga ingin menerapkan saran kedua, pastikan percobaan pelatihan yang dilakukan memiliki akurasi pada training set dan testing set di atas 92%. Lalu jika Anda mencoba lebih dari tiga skema pelatihan, pastikan setidaknya salah satu percobaan di antaranya memiliki akurasi pada training set dan testing set di atas 92% dan sisanya 85%.
Berikut contoh dari 3 percobaan skema pelatihan dengan adanya 2 kombinasi yang berbeda.
Pelatihan: SVM,    Ekstraksi Fitur: TF-IDF,    Pembagian Data: 80/20
Pelatihan: RF,    Ekstraksi Fitur: Word2Vec,    Pembagian Data: 80/20
Pelatihan: RF,    Ekstraksi Fitur: TF-IDF,    Pembagian Data: 70/30    
Melakukan inference atau testing dalam file .ipynb atau .py yang menghasilkan output berupa kelas kategorikal (contoh: negatif, netral, dan positif).
Pastikan menyertakan bukti inferensi baik itu dalam bentuk screenshot atau output pada notebook
Berikut adalah detail penilaian submission:

rating-default-1
Semua kriteria utama terpenuhi, tetapi penulisan kode masih perlu banyak diperbaiki atau terindikasi melakukan plagiat.

rating-default-2
Semua kriteria utama terpenuhi, tetapi penulisan kode masih perlu diperbaiki.

rating-default-3
Semua kriteria utama terpenuhi, tetapi tidak terdapat saran yang terpenuhi.

rating-default-4
Semua kriteria utama terpenuhi dan menerapkan minimal 3 dari seluruh saran yang ada di atas.

rating-default-5
Semua kriteria utama terpenuhi dan menerapkan semua saran yang ada di atas.

Catatan: Jika submission Anda ditolak, tidak ada penilaian. Kriteria penilaian bintang di atas hanya berlaku jika submission Anda lulus.

Instruksi Submission

Submission

Proyek Analisis Sentimen

Pengantar Kriteria Utama Penilaian Ketentuan Berkas Submission Lainnya
Beberapa poin ini perlu diperhatikan ketika mengirimkan berkas submission.

Menggunakan bahasa pemrograman Python.
File yang dikumpulkan adalah berikut:
Kriteria Utama
Notebook pelatihan model dengan format .ipynb.

File kode scraping dengan format .py atau .ipynb.

File requirements.txt.

Dataset hasil scraping dengan format .csv atau .json.

Kriteria Opsional

Notebook pelatihan model berisi cell inference jika menerapkan saran ke-6.

Mengirimkan pekerjaan Anda dalam 1 folder yang telah di-zip.

File .ipynb yang dikirim telah dijalankan terlebih dahulu sehingga output ada tanpa reviewer perlu menjalankan ulang notebook.

Instruksi Submission

Submission

Proyek Analisis Sentimen

Pengantar Kriteria Utama Penilaian Ketentuan Berkas Submission Lainnya
Tips
Untuk membuat file requirements.txt terdapat beberapa cara salah satunya menggunakan pip freeze atau pipreqs. Berikut cara penggunaan dan perbedaannya.
pip freeze
pip freeze menghasilkan daftar semua library Python yang diinstal di lingkungan saat ini beserta versinya.
 pip freeze requirements.txt
pipreqs
pipreqs menghasilkan file requirements.txt yang hanya mencantumkan library yang digunakan dalam proyek berdasarkan impor yang ada dalam file kode.
pipreqs /path/to/your/project
Tentunya kedua cara tersebut memiliki kelebihan dan kekurangan, untuk mengetahui lebih lengkap terkait freeze dan pipreqs Anda dapat membaca di tautan berikut: Ternyata Mengelola Dependensi Proyek Python Semudah Ini, lo!.
Untuk export project yang Anda kerjakan di Colaboratory sebagai berkas ipynb, klik tombol file yang berada di pojok kiri atas Colaboratory dan pilih download .ipynb serta download .py.dos-bfec37d9c5666e7e83c911cda7fd7a1220240628164934.jpeg


Resources
Untuk analisis sentimen, platform yang kaya dengan opini dan ulasan sangat ideal. Berikut adalah beberapa sumber platform yang dapat di-scraping untuk tujuan analisis sentimen.

Media Sosial

X/Twitter: Platform ini sangat kaya dengan opini pengguna yang sering diperbarui. Data tweet, retweet, dan hashtag dapat digunakan untuk menganalisis sentimen dalam berbagai topik.

Facebook: Komentar dan posting di halaman publik atau grup dapat memberikan wawasan tentang sentimen pengguna.

Instagram: Analisis sentimen dapat dilakukan melalui komentar pada posting dan caption pengguna.

Platform Ulasan Produk

Amazon: Ulasan produk memberikan banyak data yang berguna untuk analisis sentimen tentang berbagai produk.

Tokopedia: Ulasan dan rating produk dalam platform e-commerce Indonesia.

Shopee: Ulasan dan rating produk dari pengguna di Asia Tenggara.

Female Daily: Ulasan produk kecantikan dan perawatan kulit, serta diskusi di forum yang kaya dengan opini pengguna.

Google Play Store: Ulasan aplikasi dan game, memberikan data sentimen tentang performa, bug, dan fitur aplikasi.

Situs Ulasan dan Direktori

Yelp: Ulasan restoran, toko, dan layanan lokal yang memberikan banyak opini pengguna.

TripAdvisor: Ulasan tentang destinasi wisata, hotel, dan restoran.

Google Reviews: Ulasan dari berbagai layanan dan produk yang diposting oleh pengguna.

Portal Lowongan Kerja dan Ulasan Perusahaan

Glassdoor: Ulasan perusahaan, budaya kerja, dan gaji dari karyawan.

Indeed: Ulasan perusahaan dan pengalaman kerja dari karyawan.

Forum Online

Stack Overflow: Diskusi dan komentar mengenai berbagai topik pemrograman yang dapat memberikan pandangan sentimen tentang teknologi atau bahasa pemrograman tertentu.

Quora: Jawaban dan komentar tentang berbagai topik yang dapat dianalisis untuk sentimen.

Platform Konten Video

YouTube: Komentar pada video bisa memberikan data sentimen tentang berbagai topik yang dibahas dalam video tersebut.

Situs Berita dan Blog

BBC, CNN, Reuters, Kompas: Komentar pada artikel berita bisa digunakan untuk menganalisis sentimen publik terhadap berbagai peristiwa.

Medium: Komentar pada artikel blog untuk mengukur sentimen tentang topik yang dibahas.



Submission yang Tidak Sesuai Kriteria
Jika tidak sesuai dengan kriteria, submission Anda akan ditolak oleh reviewer. Berikut poin-poinnya.

Tidak melampirkan kode dan proses data scraping.
Akurasi dari model Anda di bawah 85%.
Tidak melampirkan 4 file kriteria utama yang tertera pada tab “Ketentuan Berkas Submission”.
Menggunakan data yang sudah tersedia pada open source.


Forum Diskusi
Jika mengalami kesulitan, Anda bisa bertanya langsung ke forum diskusi. https://www.dicoding.com/academies/185/discussions.



Ketentuan Proses Review
Beberapa hal yang perlu Anda ketahui mengenai proses review.

Tim penilai akan mengulas submission Anda dalam waktu selambatnya 3 hari kerja, tidak termasuk hari Sabtu, Minggu, dan libur nasional.
Tidak disarankan untuk melakukan submit berkali-kali karena akan memperlama proses penilaian yang dilakukan tim penilai.
Anda akan mendapat notifikasi hasil pengumpulan submission via email atau dapat mengecek status submission pada akun Dicoding.

https://www.dicoding.com/blog/ternyata-mengelola-dependensi-proyek-python-semudah-ini-lo/

![alt text](image.png)