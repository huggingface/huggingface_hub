<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Cache'i anlama

`huggingface_hub`, öğelerin yeniden indirilmesini önlemek için yerel diski iki cache olarak kullanır. İlk cache, Hub'dan indirilen tek tek dosyaları saklayan dosya tabanlı bir cache'tir ve bir depo güncellendiğinde aynı dosyanın yeniden indirilmemesini sağlar. İkinci cache ise chunk cache'tir; her chunk bir dosyadan bir bayt aralığını temsil eder ve dosyalar arasında paylaşılan chunk'ların yalnızca bir kez indirilmesini sağlar.

> [!TIP]
> Bu rehber, `huggingface_hub` tarafından sağlanan Python'a özgü cache yönetim araçlarını kapsar. Hugging Face Hub cache sisteminin dil bağımsız bir genel bakışı için [yerel cache hakkındaki Hub dokümantasyonuna](https://huggingface.co/docs/hub/local-cache) bakabilirsin.

## Dosya tabanlı cache

Hugging Face Hub cache sistemi, Hub'a bağımlı kütüphaneler arasında paylaşılan merkezi cache olacak şekilde tasarlanmıştır. Aynı dosyaların revision'lar arasında yeniden indirilmesini önlemek için v0.8.0'da güncellenmiştir.

Cache sistemi şöyle tasarlanmıştır:

```
<CACHE_DIR>
├─ <MODELS>
├─ <DATASETS>
├─ <SPACES>
├─ <KERNELS>
```

Varsayılan `<CACHE_DIR>`, `~/.cache/huggingface/hub` konumudur. Ancak tüm metotlarda `cache_dir` argümanıyla veya `HF_HOME` ya da `HF_HUB_CACHE` ortam değişkenini belirterek özelleştirilebilir.

Modeller, dataset'ler, Space'ler ve kernel'ler ortak bir kökü paylaşır. Bu depoların her biri depo türünü, varsa namespace'i (organizasyon veya kullanıcı adı) ve depo adını içerir:

```
<CACHE_DIR>
├─ models--julien-c--EsperBERTo-small
├─ models--lysandrejik--arxiv-nlp
├─ models--bert-base-cased
├─ datasets--glue
├─ datasets--huggingface--DataMeasurementsFiles
├─ spaces--dalle-mini--dalle-mini
```

Hub'dan indirilen tüm dosyalar artık bu klasörlerin içinde saklanır. Cache, bir dosya zaten varsa ve güncellenmediyse iki kez indirilmemesini sağlar; ancak güncellendiyse ve en son dosyayı istiyorsan, en son dosyayı indirir (yeniden ihtiyacın olursa diye önceki dosyayı bozulmadan tutarak).

Bunu sağlamak için tüm klasörler aynı iskeleti içerir:

```
<CACHE_DIR>
├─ datasets--glue
│  ├─ refs
│  ├─ blobs
│  ├─ snapshots
│  ├─ trees
...
```

Her klasör aşağıdakileri içerecek şekilde tasarlanmıştır:

### Refs

`refs` klasörü, verilen referansın en son revision'ını gösteren dosyaları içerir. Örneğin, daha önce bir deponun `main` dalından bir dosya indirdiysek, `refs` klasöründe `main` adında bir dosya bulunur ve bu dosya mevcut head'in commit tanımlayıcısını içerir.

`main` dalının en son commit'inin tanımlayıcısı `aaaaaa` ise, dosya `aaaaaa` içerir.

Aynı dal `bbbbbb` tanımlayıcılı yeni bir commit ile güncellenirse, o referanstan bir dosyayı yeniden indirmek `refs/main` dosyasını `bbbbbb` içerecek şekilde günceller.

### Blobs

`blobs` klasörü, indirdiğimiz asıl dosyaları içerir. Her dosyanın adı hash'idir.

### Snapshots

`snapshots` klasörü, yukarıda bahsedilen blob'lara giden symlink'leri içerir. Kendisi de birkaç klasörden oluşur: bilinen her revision için bir tane!

Yukarıdaki açıklamada önce `aaaaaa` revision'ından bir dosya indirmiş, ardından `bbbbbb` revision'ından bir dosya indirmiştik. Bu durumda `snapshots` klasöründe artık iki klasör olurdu: `aaaaaa` ve `bbbbbb`.

Bu klasörlerin her birinde, indirdiğimiz dosyaların adlarına sahip symlink'ler bulunur. Örneğin, `aaaaaa` revision'ında `README.md` dosyasını indirdiysek şu yola sahip olurduk:

```
<CACHE_DIR>/<REPO_NAME>/snapshots/aaaaaa/README.md
```

Bu `README.md` dosyası aslında, dosyanın hash'ine sahip blob'a işaret eden bir symlink'tir.

İskeleti bu şekilde oluşturarak dosya paylaşımı mekanizmasını açarız: aynı dosya `bbbbbb` revision'ında da indirildiyse aynı hash'e sahip olur ve dosyanın yeniden indirilmesine gerek kalmaz.

### Trees

`trees` klasörü, bir deponun belirli bir commit'te içerdiği dosya listesini cache'ler. Bir commit değişmez olduğundan dosya listesi asla değişmez. Bu da listenin Hub'a karşı bir daha kontrol edilmesine gerek kalmadan sonsuza kadar cache'lenebileceği anlamına gelir.

Her cache'lenmiş liste bir commit hash'inden sonra adlandırılır ve JSON dosyası olarak saklanır; örneğin `trees/aaaaaa.json`. O commit'teki depodaki her dosya için, dosyayı indirmek için gerekenleri kaydeder: yolu, boyutu ve hash'i. Bu, Hub'ın aksi halde döndüreceği aynı bilgidir; ancak normalde bunu almak dosya başına bir ağ çağrısına mal olur.

Bu cache [`snapshot_download`] tarafından yazılır. Bir commit'i ilk indirdiğinde dosya listesi bir kez alınır ve buraya kaydedilir. Aynı commit'i bir sonraki indirdiğinde liste yeniden alınmak yerine diskten okunur. Sonuç olarak, her şey zaten cache'teyken bir indirmeyi yeniden çalıştırmak tek bir ağ çağrısına mal olur: dal veya etiket adını bir commit hash'ine çözümlemek için gereken çağrı.

Hem [`snapshot_download`] hem de [`hf_hub_download`] ağ çağrılarından kaçınmak için bu cache'i okur. Revision olarak bir commit hash'iyle dosya indirdiğinde (bu, [`snapshot_download`]'ın dahili olarak her dosya için tam olarak yaptığı şeydir), indirme meta verileri cache'lenmiş dosya listesinden okunur ve dosya başına ağ çağrısı atlanır. Bu, tek bir dosya için yapılan bir [`hf_hub_download`]'ın da aynı commit için daha önce [`snapshot_download`] tarafından kaydedilmiş bir dosya listesinden yararlandığı anlamına gelir.

Cache'lenmiş dosya listesi bir commit'in tam olarak ne içermesi gerektiğini tanımladığından, [`snapshot_download`] yerel bir snapshot'ın tamamlanıp tamamlanmadığını da anlayabilir. Hub'a ulaşılamıyorsa (çevrimdışısın, bağlantı başarısız oluyor veya `local_files_only=True` geçtin) ve beklenen bazı dosyalar yerel snapshot'ta eksikse, [`snapshot_download`] kısmi bir klasör döndürmek yerine [`~errors.IncompleteSnapshotError`] yükseltir. Bundan önce eksik bir snapshot sessizce döndürülürdü; bu da farkında olmadan eksik dosyalarla çalışmana yol açabilirdi. `allow_patterns` veya `ignore_patterns` ile hariç tutulan dosyalar eksik sayılmaz. İstisna, `snapshot_path` özniteliği üzerinden eksik snapshot'ın yolunu açığa çıkarır; böylece gerekirse kısmen cache'lenmiş dosyaları yine de bulabilirsin.

### .no_exist (ileri seviye)

`blobs`, `refs` ve `snapshots` klasörlerine ek olarak cache'inde bir `.no_exist` klasörü de bulabilirsin. Bu klasör, bir kez indirmeyi denediğin ancak Hub'da bulunmayan dosyaları takip eder. Yapısı, bilinen her revision için 1 alt klasör içeren `snapshots` klasörüyle aynıdır:

```
<CACHE_DIR>/<REPO_NAME>/.no_exist/aaaaaa/config_that_does_not_exist.json
```

`snapshots` klasörünün aksine dosyalar basit boş dosyalardır (symlink yok). Bu örnekte, `"config_that_does_not_exist.json"` dosyası `"aaaaaa"` revision'ı için Hub'da yoktur. Yalnızca boş dosyalar sakladığı için bu klasör disk kullanımı açısından ihmal edilebilir düzeydedir.

Peki bu bilgi neden önemli olabilir?
Bazı durumlarda bir framework, bir model için isteğe bağlı dosyaları yüklemeyi dener. İsteğe bağlı dosyaların var olmadığını kaydetmek, olası her isteğe bağlı dosya için 1 HTTP çağrısından tasarruf ederek model yüklemeyi hızlandırır. Bu örneğin `transformers`'ta geçerlidir; her tokenizer ek dosyaları destekleyebilir. Tokenizer'ı makinende ilk yüklediğinde, sonraki başlatmalarda yükleme süresini kısaltmak için hangi isteğe bağlı dosyaların var olduğunu (ve hangilerinin olmadığını) cache'ler.

Bir dosyanın yerel olarak cache'lenip cache'lenmediğini test etmek için (hiç HTTP isteği yapmadan) [`try_to_load_from_cache`] yardımcısını kullanabilirsin. Ya dosya yolunu (varsa ve cache'teyse), `_CACHED_NO_EXIST` nesnesini (yokluk cache'teyse) ya da `None` (bilmiyorsak) döndürür.

```python
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

filepath = try_to_load_from_cache()
if isinstance(filepath, str):
    # file exists and is cached
    ...
elif filepath is _CACHED_NO_EXIST:
    # non-existence of file is cached
    ...
else:
    # file is not cached
    ...
```

### Pratikte

Pratikte cache'in şu ağaca benzer görünmelidir:

```text
    [  96]  .
    └── [ 160]  models--julien-c--EsperBERTo-small
        ├── [ 160]  blobs
        │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
        │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
        ├── [  96]  refs
        │   └── [  40]  main
        ├── [ 128]  snapshots
        │   ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
        │   │   ├── [  52]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
        │   │   └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        │   └── [ 128]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
        │       ├── [  52]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
        │       └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        └── [  96]  trees
            ├── [ 521]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f.json
            └── [ 521]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48.json
```

### CACHEDIR.TAG

`huggingface_hub`, cache dizininde otomatik olarak bir
[`CACHEDIR.TAG`](https://bford.info/cachedir/) dosyası oluşturur. Bu
etiket *Cache Directory Tagging Standard*'ı izler ve yedekleme araçlarına (ör. Borg,
restic, rsync) dizinin yeniden indirilebilir cache verisi içerdiğini ve yedeklemelerden
güvenle hariç tutulabileceğini söyler.

### Sınırlamalar

Verimli bir cache sistemi için `huggingface-hub` symlink kullanır. Ancak
symlink'ler tüm makinelerde desteklenmez. Bu, özellikle Windows'ta bilinen bir
sınırlamadır. Bu durumda `huggingface_hub`, `blobs/` dizinini kullanmaz; bunun yerine
dosyaları doğrudan `snapshots/` dizininde saklar. Bu geçici çözüm, kullanıcıların
Hub'dan dosyaları tıpkı aynı şekilde indirmesine ve cache'lemesine olanak tanır. Cache'i
inceleme ve silme araçları (aşağıya bak) da desteklenir. Ancak cache sistemi daha az
verimlidir çünkü aynı deponun birden fazla revision'ı indirildiğinde tek bir dosya
birkaç kez indirilebilir.

Windows bir makinede symlink tabanlı cache sisteminden yararlanmak istiyorsan,
ya [Geliştirici Modunu etkinleştirmen](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)
ya da Python'u yönetici olarak çalıştırman gerekir.

Symlink'siz cache modunu önceden kullanmak istiyorsan (ör. symlink'leri iyi
işlemeyen paylaşımlı bir dosya sisteminde), [`HF_HUB_DISABLE_SYMLINKS`](../package_reference/environment_variables#hfhubdisablesymlinks) ortam değişkenini `1` olarak ayarlayabilirsin. Dosyalar `blobs/`'a symlink oluşturmak yerine doğrudan `snapshots/` içine kopyalanır.

Symlink'ler desteklenmediğinde, kullanıcıya bozulmuş bir cache sistemi
kullandıklarını bildirmek için bir uyarı mesajı gösterilir. Bu uyarı,
`HF_HUB_DISABLE_SYMLINKS_WARNING` ortam değişkeni true olarak ayarlanarak
devre dışı bırakılabilir.

### Depolar arasında paylaşılan blob'lar

Varsayılan olarak Xet dosyaları depolar arasında da tekilleştirilir. `hf_xet` üzerinden indirilen bir Xet dosyası bir kez `<CACHE_DIR>/blobs/<prefix>/<xet_hash>` konumunda saklanır ve deponun `blobs/<etag>` girişi buna göreli bir symlink'tir. Başka bir depo aynı dosyaya ihtiyaç duyduğunda indirme yerine bir symlink alır: bayt aktarılmaz ve ekstra alan kullanılmaz. Snapshot düzeni değişmez. Bir işaretçi dosyası deposu tanımlar; böylece `huggingface_hub` tarafından oluşturulmamış bir `blobs` dizinine asla dokunulmaz.

Her paylaşılan dosyanın, onu kullanan depo blob'larını listeleyen bir `<xet_hash>.refs` manifest'i vardır. `hf cache rm` bir silmeden etkilenen yalnızca paylaşılan dosyaları kontrol etmek için bunu okur ve `hf cache prune`, artık hiçbir cache'lenmiş deponun kullanmadığı paylaşılan dosyaları kaldırır. Manifest yalnızca bir ipucudur: her giriş dosya sistemine karşı kontrol edilir ve eksik ya da okunamayan meta verisi olan bir dosya tutulur. Temizlik, geçerli bir cache girişini bozmaktansa geri kazanılabilir veriyi geride bırakmayı tercih eder.

Depo başına boyutlar mantıksal kalır; yani paylaşılan bir dosya onu kullanan her depo için sayılır. Cache genelindeki `size_on_disk` fizikseldir: her paylaşılan dosya bir kez sayılır; artık hiçbir deponun kullanmadığı dosyalar dahil.

Daha eski istemciler (`huggingface_hub`, `huggingface.js`, `hf-hub`, `llama.cpp` ve symlink'leri izleyen her şey) aynı depo klasörlerinde normal şekilde okumaya ve indirmeye devam eder. İki sınırlama vardır:
- Cache silme araçları, diğer depolar tarafından hâlâ kullanılan bir paylaşılan dosyayı silebilir. Etkilenen dosyalar bir sonraki kullanımda yeniden indirilir. `hf cache rm`, `hf cache prune` ve programatik silme için güncel bir `huggingface_hub` kullan.
- Cache tarayıcıları üst düzey `blobs` dizinini bilinmeyen bir giriş olarak raporlayabilir.

Depo, symlink tabanlı cache düzenini gerektirir ve `HF_HUB_DISABLE_XET=1` ile devre dışı bırakılır. Desteklenmeyen bir dosya sistemi, bir izin hatası veya önceden var olan işaretsiz bir `blobs` dizini gibi bir dosyayı paylaşma başarısızlığı, sessizce düzenli depo-yerel depolamaya geri düşer. Tamamen vazgeçmek için [`HF_HUB_DISABLE_SHARED_BLOBS=1`](../package_reference/environment_variables#hfhubdisablesharedblobs) ayarla.

## Bir revision sabitleme (ileri seviye)

> [!TIP]
> Hub'ı bir ML kütüphanesine entegre ediyorsan, tek bir [`snapshot_download`] çağrısı hâlâ önerilen yaklaşımdır: revision'ı bir kez çözümler, her şeyi paralel indirir ve dosya listesini cache'ler. Aşağıdakiler yalnızca birçok bileşeni ayrı ayrı indirip yükleyen (config, ağırlıklar, tokenizer, processor, adapter, ...) ve tek bir çağrı kullanamayan karmaşık kütüphaneler için yararlıdır.

Bir kütüphane birkaç dosyayı tek tek indirdiğinde, her çağrının `revision="main"` ifadesini yeniden bir commit hash'ine çözmesi gerekir. Bu, dosya başına bir HTTP çağrısına mal olur ve daha kötüsü, birkaç saniye arayla yapılan iki çağrı, arada depo güncellenirse iki farklı commit'e düşebilir.

[`HfApi.resolve_revision`], revision'ı bir kez çözer ve bir [`ResolvedRevision`] döndürür:

```py
>>> from huggingface_hub import resolve_revision
>>> revision = resolve_revision("openai-community/gpt2")
>>> revision
ResolvedRevision(initial=None, resolved='607a30d783dfa663caf39e06633721c8d4cfcd7e')
```

[`ResolvedRevision`] bir `str` alt sınıfıdır; bu yüzden `revision` argümanı alan herhangi bir `huggingface_hub` metoduna geçilebilir. String değeri kullanıcının başlangıçta istediği şeydir (burada `"main"`, dolayısıyla okunabilir hata mesajları), `.resolved` ise commit hash'ini tutar:

```py
>>> revision == "main"
True
>>> revision.resolved
'607a30d783dfa663caf39e06633721c8d4cfcd7e'
```

İndirme yardımcıları ([`hf_hub_download`], [`snapshot_download`], [`get_cached_repo_tree`]) bir [`ResolvedRevision`] algılar ve commit hash'ini doğrudan kullanır. Her dosyanın aynı commit'ten geldiği garantilenir ve dosyalar bir kez cache'lendikten sonra hiç HTTP çağrısına gerek kalmaz:

```py
>>> from huggingface_hub import hf_hub_download
>>> config = hf_hub_download("openai-community/gpt2", "config.json", revision=revision)
>>> weights = hf_hub_download("openai-community/gpt2", "model.safetensors", revision=revision)
```

`revision` → `commit hash` eşlemesi ayrıca cache'in `refs/` klasörüne de yazılır (bkz. [Refs](#refs)). Bu, daha sonra Hub'a ulaşılamazsa (çevrimdışı mod, bağlantı hatası, zaman aşımı, Hub kesintisi) [`HfApi.resolve_revision`]'ın saydam biçimde cache'lenmiş değere geri düştüğü anlamına gelir. Hiçbir şey cache'te de yoksa bir [`~errors.RevisionResolutionError`] yükseltilir.

Bir commit hash'i yalnızca çözüldüğü depo için bir şey ifade eder ve indirme yardımcıları onu olduğu gibi kullanır. Bu yüzden bir [`ResolvedRevision`] yalnızca çözüldüğü depoya geçilmelidir. Bir kütüphane başka bir depodan da indiriyorsa (bir temel model, bir adapter, kendi deposunda yaşayan bir bileşen, ...), o depo için çözülmüş bir revision'a ihtiyaç duyar. [`ResolvedRevision`]'ı yalnızca [`HfApi.resolve_revision`]'a geri geç: hangi depoya ait olduğunu hatırlar ve başlangıçta istenen revision'ı (burada `"main"`) yeni depo için yeniden çözer.

```py
>>> other_revision = resolve_revision("openai-community/gpt2-medium", revision=revision)  # resolves "main" again
>>> other_revision.resolved
'6dcaa7a952f72f9298047fd5137cd6e4f05f41da'
>>> config = hf_hub_download("openai-community/gpt2-medium", "config.json", revision=other_revision)
```

## Chunk tabanlı cache (Xet)

Daha verimli dosya aktarımları sağlamak için `hf_xet`, mevcut `huggingface_hub` cache'ine bir `xet` dizini ekleyerek chunk tabanlı tekilleştirmeyi mümkün kılan ek bir cache katmanı oluşturur. Bu cache, chunk'ları (dosyaların ~64KB boyutundaki değişmez bayt aralıkları) ve shard'ları (dosyaları chunk'lara eşleyen bir veri yapısı) tutar. Xet Storage sistemi hakkında daha fazla bilgi için bu [bölüme](https://huggingface.co/docs/hub/xet/index) bakabilirsin.

Varsayılan olarak `~/.cache/huggingface/xet` konumunda bulunan `xet` dizini, yüklemeler ve indirmeler için kullanılan iki cache içerir. Yapısı şöyledir:

```bash
<CACHE_DIR>
├─ xet
│  ├─ environment_identifier
│  │  ├─ chunk_cache
│  │  ├─ shard_cache
│  │  ├─ staging
```

`environment_identifier` dizini kodlanmış bir string'tir (makinende `https___cas_serv-tGqkUaZf_CBPHQ6h` olarak görünebilir). Bu, geliştirme sırasında cache'in yerel ve üretim sürümlerinin aynı anda yan yana var olmasına olanak tanımak için kullanılır. Ayrıca farklı [depolama bölgelerinde](https://huggingface.co/docs/hub/storage-regions) bulunan depolardan indirirken de kullanılır. `xet` dizininde her biri farklı bir ortama karşılık gelen birden fazla böyle giriş görebilirsin; ancak iç yapıları aynıdır.

İç dizinler şu amaçlara hizmet eder:
* `chunk-cache`, indirmeleri hızlandırmak için kullanılan cache'lenmiş veri chunk'larını içerir.
* `shard-cache`, yükleme yolunda kullanılan cache'lenmiş shard'ları içerir.
* `staging`, sürdürülebilir yüklemeleri desteklemek için tasarlanmış bir çalışma alanıdır.

Bunlar aşağıda belgelenmiştir.

`xet` cache sistemi, `hf_xet`'in geri kalanı gibi `huggingface_hub` ile tamamen entegredir. Cache'lenmiş varlıklarla etkileşim için mevcut API'leri kullanıyorsan iş akışını güncellemen gerekmez. `xet` cache'leri, mevcut `hf_xet` chunk tabanlı tekilleştirme ve `huggingface_hub` cache sistemi üzerine bir optimizasyon katmanı olarak inşa edilmiştir.


### `chunk_cache`

Bu cache indirme yolunda kullanılır. Cache dizin yapısı, her Xet etkin deponun arkasındaki içerik adresli depodan (CAS) gelen base-64 kodlanmış bir hash'e dayanır. Bir CAS hash'i, verinin nerede saklandığının ofsetlerini aramak için anahtar görevi görür. Not: `hf_xet` 1.2.0 itibarıyla chunk_cache varsayılan olarak devre dışıdır. Etkinleştirmek için Python sürecini başlatmadan önce `HF_XET_CHUNK_CACHE_SIZE_BYTES` ortam değişkenini uygun boyuta ayarla.

En üst düzeyde, base 64 kodlanmış CAS hash'inin ilk iki harfi `chunk_cache` içinde bir alt dizin oluşturmak için kullanılır (bu ilk iki harfi paylaşan anahtarlar burada gruplanır). İç düzeyler, dizin adı olarak tam anahtara sahip alt dizinlerden oluşur. En altta, cache'lenmiş chunk'ları içeren blok aralıkları olan cache öğeleri bulunur.

```bash
<CACHE_DIR>
├─ xet
│  ├─ chunk_cache
│  │  ├─ A1
│  │  │  ├─ A1GerURLUcISVivdseeoY1PnYifYkOaCCJ7V5Q9fjgxkZWZhdWx0
│  │  │  │  ├─ AAAAAAEAAAA5DQAAAAAAAIhRLjDI3SS5jYs4ysNKZiJy9XFI8CN7Ww0UyEA9KPD9
│  │  │  │  ├─ AQAAAAIAAABzngAAAAAAAPNqPjd5Zby5aBvabF7Z1itCx0ryMwoCnuQcDwq79jlB

```

Bir dosya istendiğinde `hf_xet`'in yaptığı ilk şey, yeniden oluşturma bilgisi için Xet storage'ın içerik adresli deposu (CAS) ile iletişim kurmaktır. Yeniden oluşturma bilgisi, dosyayı bütünüyle indirmek için gereken CAS anahtarları hakkında bilgi içerir.

CAS anahtarları için istekler yürütülmeden önce `chunk_cache`'e danışılır. Cache'teki bir anahtar bir CAS anahtarıyla eşleşirse, o içerik için istek göndermeye gerek yoktur. `hf_xet` bunun yerine dizinde saklanan chunk'ları kullanır.

`chunk_cache` salt bir optimizasyon olduğundan, garanti olmadığından, `hf_xet` hesaplama açısından verimli bir boşaltma politikası kullanır. `chunk_cache` dolduğunda (aşağıdaki `Sınırlar ve sınırlamalar`'a bak), `hf_xet` bir boşaltma adayı seçerken rastgele boşaltma politikası uygular. Bu, sağlam bir cache sistemini (ör. LRU) yönetmenin ek yükünü önemli ölçüde azaltırken chunk'ları cache'lemenin faydalarının çoğunu yine de sağlar.

### `shard_cache`

Bu cache Hub'a içerik yüklerken kullanılır. Dizin düzdür; yalnızca shard dosyalarından oluşur ve her biri shard adı için bir ID kullanır.

```sh
<CACHE_DIR>
├─ xet
│  ├─ shard_cache
│  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
│  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  ├─ ceeeb7ea4cf6c0a8d395a2cf9c08871211fbbd17b9b5dc1005811845307e6b8f.mdb
│  │  ├─ e8535155b1b11ebd894c908e91a1e14e3461dddd1392695ddc90ae54a548d8b2.mdb
```

`shard_cache` şu shard'ları içerir:

- Yerelde üretilmiş ve CAS'a başarıyla yüklenmiş olanlar
- Global tekilleştirme algoritmasının parçası olarak CAS'tan indirilenler

Shard'lar dosyalar ile chunk'lar arasında bir eşleme sağlar. Yüklemeler sırasında her dosya chunk'lara bölünür ve chunk'ın hash'i kaydedilir. Ardından cache'teki her shard'a danışılır. Bir shard, yüklenen yerel dosyada bulunan bir chunk hash'i içeriyorsa, o chunk CAS'ta zaten saklandığı için atılabilir.

Tüm shard'ların indirildikleri andan itibaren 3-4 haftalık bir son kullanma tarihi vardır. Süresi dolmuş shard'lar yükleme sırasında yüklenmez ve süre dolduktan bir hafta sonra silinir.

### `staging`

Bir yükleme, yeni içerik depoya commit edilmeden önce sonlanırsa dosya aktarımına devam etmen gerekir. Ancak kesintiden önce bazı chunk'ların başarıyla yüklenmiş olması mümkündür.

Baştan yeniden başlamak zorunda kalmaman için `staging` dizini yüklemeler sırasında bir çalışma alanı olarak hareket eder ve başarıyla yüklenen chunk'lar için meta veriyi saklar. `staging` dizininin şekli şöyledir:

```
<CACHE_DIR>
├─ xet
│  ├─ staging
│  │  ├─ shard-session
│  │  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  │  ├─ xorb-metadata
│  │  │  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
```

Dosyalar işlenip chunk'lar başarıyla yüklendikçe meta verileri `xorb-metadata` içinde bir shard olarak saklanır. Bir yükleme oturumuna devam edildiğinde her dosya yeniden işlenir ve bu dizindeki shard'lara danışılır. Başarıyla yüklenen herhangi bir içerik atlanır ve yeni içerik yüklenir (ve meta verisi kaydedilir).

Bu arada `shard-session`, işlenen dosyalar için dosya ve chunk bilgisini saklar. Bir yükleme başarıyla tamamlandığında bu shard'lardaki içerik daha kalıcı olan `shard-cache`'e taşınır.

### Sınırlar ve sınırlamalar

`chunk_cache` boyutta 10GB ile sınırlıyken `shard_cache`'in 4GB'lık yumuşak bir sınırı vardır. Tasarım gereği her iki cache'in de üst düzey API'leri yoktur; ancak boyutları `HF_XET_CHUNK_CACHE_SIZE_BYTES` ve `HF_XET_SHARD_CACHE_SIZE_LIMIT` ortam değişkenleriyle yapılandırılabilir.

Bu cache'ler öncelikle bir dosyanın yeniden oluşturulmasını (indirme) veya yüklenmesini kolaylaştırmak için kullanılır. Varlıkların kendileriyle etkileşim kurmak için [`huggingface_hub` cache sistemi API'lerini](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) kullanman önerilir.

Her iki cache tarafından kullanılan alanı geri kazanman veya olası cache ile ilgili sorunları ayıklaman gerekiyorsa, `rm -rf ~/<cache_dir>/xet` çalıştırarak `xet` cache'ini tamamen kaldır; burada `<cache_dir>` Hugging Face cache'inin konumudur, tipik olarak `~/.cache/huggingface`

Örnek tam `xet` cache dizin ağacı:

```sh
<CACHE_DIR>
├─ xet
│  ├─ chunk_cache
│  │  ├─ L1
│  │  │  ├─ L1GerURLUcISVivdseeoY1PnYifYkOaCCJ7V5Q9fjgxkZWZhdWx0
│  │  │  │  ├─ AAAAAAEAAAA5DQAAAAAAAIhRLjDI3SS5jYs4ysNKZiJy9XFI8CN7Ww0UyEA9KPD9
│  │  │  │  ├─ AQAAAAIAAABzngAAAAAAAPNqPjd5Zby5aBvabF7Z1itCx0ryMwoCnuQcDwq79jlB
│  ├─ shard_cache
│  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
│  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  ├─ ceeeb7ea4cf6c0a8d395a2cf9c08871211fbbd17b9b5dc1005811845307e6b8f.mdb
│  │  ├─ e8535155b1b11ebd894c908e91a1e14e3461dddd1392695ddc90ae54a548d8b2.mdb
│  ├─ staging
│  │  ├─ shard-session
│  │  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  │  ├─ xorb-metadata
│  │  │  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
```

Xet Storage hakkında daha fazla bilgi edinmek için bu [bölüme](https://huggingface.co/docs/hub/xet/index) bakabilirsin.

## Asset'leri cache'leme

Hub'dan dosyaları cache'lemenin yanı sıra, alt seviye kütüphaneler sıkça HF ile ilgili ancak `huggingface_hub` tarafından doğrudan işlenmeyen diğer dosyaları da cache'lemek ister (örnek: GitHub'dan indirilen dosya, ön işlenmiş veri, loglar,...). `asset` adı verilen bu dosyaları cache'lemek için [`cached_assets_path`] kullanılabilir. Bu küçük yardımcı, onu isteyen kütüphanenin adına ve isteğe bağlı olarak bir namespace ile bir alt klasör adına dayalı olarak HF cache'inde birleşik bir şekilde yollar üretir. Amaç, her alt seviye kütüphanenin asset'lerini kendi yolunca yönetmesine izin vermektir (ör. yapı üzerinde kural yok) yeter ki doğru asset klasöründe kalsın. Bu kütüphaneler ardından cache'i yönetmek için `huggingface_hub` araçlarından yararlanabilir; özellikle bir CLI komutuyla asset'lerin parçalarını tarama ve silme.

```py
from huggingface_hub import cached_assets_path

assets_path = cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
something_path = assets_path / "something.json" # Do anything you like in your assets folder !
```

> [!TIP]
> [`cached_assets_path`] asset'leri saklamanın önerilen yoludur ancak zorunlu değildir. Eğer
> kütüphanen zaten kendi cache'ini kullanıyorsa, onu kullanmaktan çekinme!

### Pratikte asset'ler

Pratikte asset cache'in şu ağaca benzer görünmelidir:

```text
    assets/
    └── datasets/
    │   ├── SQuAD/
    │   │   ├── downloaded/
    │   │   ├── extracted/
    │   │   └── processed/
    │   ├── Helsinki-NLP--tatoeba_mt/
    │       ├── downloaded/
    │       ├── extracted/
    │       └── processed/
    └── transformers/
        ├── default/
        │   ├── something/
        ├── bert-base-cased/
        │   ├── default/
        │   └── training/
    hub/
    └── models--julien-c--EsperBERTo-small/
        ├── blobs/
        │   ├── (...)
        │   ├── (...)
        ├── refs/
        │   └── (...)
        └── [ 128]  snapshots/
            ├── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
            │   ├── (...)
            └── bbc77c8132af1cc5cf678da3f1ddf2de43606d48/
                └── (...)
```

## Dosya tabanlı cache'ini yönetme

### Cache'ini inceleme

Şu anda cache'lenmiş dosyalar yerel dizininden asla silinmez: bir dalın yeni bir
revision'ını indirdiğinde, yeniden ihtiyacın olursa diye önceki dosyalar tutulur.
Bu nedenle hangi depoların ve revision'ların en çok disk alanı kapladığını bilmek
için cache dizinini incelemek yararlı olabilir. `huggingface_hub`, `hf` CLI'sinden
veya Python'dan kullanabileceğin yardımcılar sağlar.

**Cache'i terminalden inceleme**

Yerelde neyin saklandığını keşfetmek için `hf cache ls` çalıştır. Varsayılan olarak
komut bilgiyi depoya göre toplar:

```text
➜ hf cache ls
ID                                   SIZE   LAST_ACCESSED LAST_MODIFIED REFS
------------------------------------ ------- ------------- ------------- -------------------
dataset/glue                         116.3K 4 days ago     4 days ago     2.4.0 main 1.17.0
dataset/google/fleurs                 64.9M 1 week ago     1 week ago     main refs/pr/1
model/Jean-Baptiste/camembert-ner    441.0M 2 weeks ago    16 hours ago   main
model/bert-base-cased                  1.9G 1 week ago     2 years ago
model/t5-base                          10.1K 3 months ago   3 months ago   main
model/t5-small                        970.7M 3 days ago     3 days ago     main refs/pr/1

Found 6 repo(s) for a total of 12 revision(s) and 3.4G on disk.
```

Her cache'lenmiş snapshot'ı listelemek için `--revisions` ekle ve önemli olana
odaklanmak için filtreleri zincirle. Filtreler insan dostu boyutları ve süreleri
anlar; böylece `size>1GB` veya `accessed>30d` gibi ifadeler kutudan çıktığı gibi çalışır:

```text
➜ hf cache ls --revisions --filter "size>1GB" --filter "accessed>30d"
ID                                   REVISION            SIZE   LAST_MODIFIED REFS
------------------------------------ ------------------ ------- ------------- -------------------
model/bert-base-cased                6d1d7a1a2a6cf4c2    1.9G  2 years ago
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main

Found 2 repo(s) for a total of 2 revision(s) and 3.0G on disk.
```

Makine dostu çıktı mı lazım? Yapılandırılmış nesneler için `--format json` veya
elektronik tablolar için `--format csv` kullan. Alternatif olarak `--quiet` yalnızca
tanımlayıcıları yazdırır (satır başına bir) böylece onları diğer araçlara boru
edebilirsin. Girişleri `accessed`, `modified`, `name` veya `size`'a göre sıralamak
için `--sort` kullan (`:asc` veya `:desc` ekleyerek sırayı kontrol et) ve sonuçları
üst N girişle sınırlamak için `--limit` kullan. `HF_HOME` dışında saklanan bir
cache'i incelemen gerektiğinde bu seçenekleri `--cache-dir` ile birleştir.

**Yaygın kabuk araçlarıyla filtreleme**

Tablo çıktısı, zaten bildiğin araçları kullanmaya devam edebileceğin anlamına gelir.
Örneğin aşağıdaki snippet, `t5-small` ile ilgili her cache'lenmiş revision'ı bulur:

```text
➜ eval "hf cache ls --revisions" | grep "t5-small"
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main
model/t5-small                       8f3ad1c90fed7a62    820.1M 2 weeks ago   refs/pr/1
```

**Cache'i Python'dan inceleme**

Daha ileri düzey kullanım için, CLI aracı tarafından çağrılan Python yardımcısı
olan [`scan_cache_dir`]'i kullan.

Onu 4 dataclass etrafında yapılandırılmış ayrıntılı bir rapor almak için
kullanabilirsin:

- [`HFCacheInfo`]: [`scan_cache_dir`] tarafından döndürülen tam rapor
- [`CachedRepoInfo`]: cache'lenmiş bir depo hakkında bilgi
- [`CachedRevisionInfo`]: bir depo içindeki cache'lenmiş bir revision (ör. "snapshot") hakkında bilgi
- [`CachedFileInfo`]: bir snapshot'taki cache'lenmiş bir dosya hakkında bilgi

İşte basit bir kullanım örneği. Ayrıntılar için referansa bak.

```py
>>> from huggingface_hub import scan_cache_dir

>>> hf_cache_info = scan_cache_dir()
HFCacheInfo(
    size_on_disk=3398085269,
    repos=frozenset({
        CachedRepoInfo(
            repo_id='t5-small',
            repo_type='model',
            repo_path=PosixPath(...),
            size_on_disk=970726914,
            nb_files=11,
            last_accessed=1662971707.3567169,
            last_modified=1662971107.3567169,
            revisions=frozenset({
                CachedRevisionInfo(
                    commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                    size_on_disk=970726339,
                    snapshot_path=PosixPath(...),
                    # No `last_accessed` as blobs are shared among revisions
                    last_modified=1662971107.3567169,
                    files=frozenset({
                        CachedFileInfo(
                            file_name='config.json',
                            size_on_disk=1197
                            file_path=PosixPath(...),
                            blob_path=PosixPath(...),
                            blob_last_accessed=1662971707.3567169,
                            blob_last_modified=1662971107.3567169,
                        ),
                        CachedFileInfo(...),
                        ...
                    }),
                ),
                CachedRevisionInfo(...),
                ...
            }),
        ),
        CachedRepoInfo(...),
        ...
    }),
    warnings=[
        CorruptedCacheException("Snapshots dir doesn't exist in cached repo: ..."),
        CorruptedCacheException(...),
        ...
    ],
)
```

### Cache'ini doğrulama

`huggingface_hub`, cache'lenmiş dosyalarının Hub'daki checksum'larla eşleştiğini doğrulayabilir. Belirli bir deponun belirli bir revision'ı için dosya tutarlılığını doğrulamak üzere `hf cache verify` CLI'sini kullan:


```bash
>>> hf cache verify meta-llama/Llama-3.2-1B-Instruct
✅ Verified 13 file(s) for 'meta-llama/Llama-3.2-1B-Instruct' (model) in ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
  All checksums match.
```

Belirli bir cache'lenmiş revision'ı doğrula:

```bash
>>> hf cache verify meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659
```

> [!TIP]
> Kullanım ve seçeneklerin eksiksiz listesi hakkında daha fazla ayrıntı için [`hf cache verify` CLI referansına](../package_reference/cli#hf-cache-verify) bak.

### Cache'ini temizleme

Cache'ini taramak ilginçtir ama genellikle asıl yapmak istediğin şey, sürücünde
biraz yer açmak için bazı kısımları silmektir. Bu, `hf cache rm` ve `hf cache prune`
CLI komutlarıyla mümkündür. Ayrıca cache taranırken döndürülen [`HFCacheInfo`]
nesnesinden [`~HFCacheInfo.delete_revisions`] ve [`~HFCacheInfo.delete_files`]
yardımcıları programatik olarak da kullanılabilir.

**Silme stratejisi**

Biraz cache silmek için silinecek revision'ların bir listesini geçmen gerekir. Araç
bu listeye göre alanı boşaltmak için bir strateji tanımlar. Hangi dosya ve klasörlerin
silineceğini tanımlayan bir [`DeleteCacheStrategy`] nesnesi döndürür.
[`DeleteCacheStrategy`], ne kadar alanın boşaltılmasının beklendiğini sana verir.
Silmeyi kabul ettiğinde, silmenin etkili olması için onu yürütmen gerekir.
Tutarsızlıklardan kaçınmak için bir strateji nesnesini elle düzenleyemezsin.

Revision'ları silme stratejisi şöyledir:

- revision symlink'lerini içeren `snapshot` klasörü silinir.
- yalnızca silinecek revision'lar tarafından hedeflenen blob dosyaları da silinir.
- bir revision 1 veya daha fazla `refs`'e bağlıysa, referanslar silinir.
- bir depodaki tüm revision'lar silinirse, cache'lenmiş deponun tamamı silinir.

[`~HFCacheInfo.delete_files`] ile tek tek dosyaları silmek aynı mantığı izler:
snapshot girişleri kaldırılır ve blob'ları yalnızca başka hiçbir cache'lenmiş dosya
onlara başvurmuyorsa silinir. Refs ve snapshot klasörleri tutulur.

> [!TIP]
> Revision hash'leri tüm depolar arasında benzersizdir. Bu yüzden `hf cache rm` ya
> bir depo tanımlayıcısını (örneğin `model/bert-base-uncased`) ya da yalın bir revision
> hash'ini kabul eder; bir hash geçerken depoyu ayrıca belirtmen gerekmez.

> [!WARNING]
> Bir revision cache'te bulunamazsa sessizce yok sayılır. Ayrıca silmeye çalışırken
> bir dosya veya klasör bulunamazsa bir uyarı loglanır ancak hata fırlatılmaz. Silme,
> [`DeleteCacheStrategy`] nesnesinde bulunan diğer yollar için devam eder.

**Cache'i terminalden temizleme**

Cache'inden depoları veya revision'ları kalıcı olarak silmek için `hf cache rm`
kullan. Bir veya daha fazla depo tanımlayıcısı (örneğin `model/bert-base-uncased`)
veya revision hash'i geç:

```text
➜ hf cache rm model/bert-base-cased
About to delete 1 repo(s) totalling 1.9G.
  - model/bert-base-cased (entire repo)
Proceed with deletion? [y/N]: y
Deleted 1 repo(s) and 1 revision(s); freed 1.9G.
```

Ayrıca bir filtreyle tanımlanan girişleri toplu silmek için `hf cache rm`'i
`hf cache ls --quiet` ile birlikte kullanabilirsin:

```bash
>>> hf cache rm $(hf cache ls --filter "accessed>1y" -q) -y
About to delete 2 repo(s) totalling 5.31G.
  - model/meta-llama/Llama-3.2-1B-Instruct (entire repo)
  - model/hexgrad/Kokoro-82M (entire repo)
Delete repo: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
Delete repo: ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
Cache deletion done. Saved 5.31G.
Deleted 2 repo(s) and 2 revision(s); freed 5.31G.
```

Aynı çağrıda depoları ve revision'ları karıştır. Etkiyi önizlemek için `--dry-run`
ekle veya betiklerken onay istemini atlamak için `--yes` kullan:

```text
➜ hf cache rm model/t5-small 8f3ad1c --dry-run
About to delete 1 repo(s) and 1 revision(s) totalling 1.1G.
  - model/t5-small:
      8f3ad1c [main] 1.1G
Dry run: no files were deleted.
```

Tüm depo yerine tek bir dosyayı kaldırmak için, örneğin bir GGUF quantizasyonu,
bir `hf://` dosya URI'si geç. Dosya, deponun her cache'lenmiş revision'ından
kaldırılır ve blob'u yalnızca başka hiçbir cache'lenmiş dosya hâlâ ona
başvurmuyorsa silinir. Revision kullanılabilir kalır ve silinen bir dosya bir
sonraki ihtiyaçta yeniden indirilir. Yollar tam eşleşmelidir: klasörler ve glob
desenleri desteklenmez.

```text
➜ hf cache rm hf://models/unsloth/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf --dry-run
About to delete 1 file(s) totalling 16.5G.
  - model/unsloth/gemma-3-27b-it-GGUF@3f4b5c1d2e6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c/gemma-3-27b-it-Q4_K_M.gguf
Dry run: no files were deleted.
```

Varsayılan cache konumu dışında çalışırken komutu `--cache-dir PATH` ile eşleştir.

Cache çöpünü toplu temizlemek için `hf cache prune` çalıştır. Artık bir dal veya
etiket tarafından referans alınmayan revision'ları, yarıda kesilen indirmelerden
kalan `.incomplete` dosyalarını ve artık hiçbir cache'lenmiş deponun başvurmadığı
paylaşılan blob'ları otomatik olarak siler:

```text
➜ hf cache prune
About to delete 3 unreferenced revision(s) and 2 incomplete download(s) (2.4G total).
  - model/t5-small:
      1c610f6b [refs/pr/1] 820.1M
      d4ec9b72 [(detached)] 640.5M
  - dataset/google/fleurs:
      2b91c8dd [(detached)] 937.6M
Proceed? [y/N]: y
Deleted 3 unreferenced revision(s) and 2 incomplete download(s); freed 2.4G.
```

`.incomplete` dosyaları, bir indirme yarıda kesildiğinde geride kalan kısmi blob'lardır.
Revision tabanlı tarama tarafından izlenmezler; bu yüzden `hf cache ls` onları yalnızca
bir ipucuyla işaretler (`Found X incomplete download(s) ...`) ve aslında onları kaldıran
komut `hf cache prune`'dır. `hf cache rm` onlara asla dokunmaz; yalnızca tüm bir depoyu
sildiğinde hariç.

Her iki komut da `--dry-run`, `--yes` ve `--cache-dir`'i destekler; böylece gerektiğinde
önizleyebilir, otomatikleştirebilir ve alternatif cache dizinlerini hedefleyebilirsin.

**Cache'i Python'dan temizleme**

Daha fazla esneklik için [`~HFCacheInfo.delete_revisions`] metodunu programatik olarak
da kullanabilirsin. İşte basit bir örnek. Ayrıntılar için referansa bak.

```py
>>> from huggingface_hub import scan_cache_dir

>>> delete_strategy = scan_cache_dir().delete_revisions(
...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa"
...     "e2983b237dccf3ab4937c97fa717319a9ca1a96d",
...     "6c0e6080953db56375760c0471a8c5f2929baf11",
... )
>>> print("Will free " + delete_strategy.expected_freed_size_str)
Will free 8.6G

>>> delete_strategy.execute()
Cache deletion done. Saved 8.6G.
```
