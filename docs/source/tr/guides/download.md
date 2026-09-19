# Hub'dan dosya indirme

`huggingface_hub` kütüphanesi, Hub'da depolanan depolardan dosya indirmek için
fonksiyonlar sunar. Bu fonksiyonları bağımsız olarak kullanabilir veya kendi
kütüphanene entegre ederek kullanıcılarının Hub ile etkileşimini kolaylaştırabilirsin.
Bu rehber sana şunları gösterecek:

* Tek bir dosyayı indirme ve cache'leme.
* Bir deponun tamamını indirme ve cache'leme.
* Dosyaları yerel bir klasöre indirme.

## Tek bir dosya indirme

[`hf_hub_download`] fonksiyonu, Hub'dan dosya indirmek için ana fonksiyondur.
Uzak dosyayı indirir, diske (sürüme duyarlı şekilde) cache'ler ve yerel dosya yolunu
döndürür.

> [!TIP]
> Döndürülen dosya yolu, HF yerel cache'ine bir işaretçidir. Bu nedenle, bozuk bir
> cache'den kaçınmak için dosyayı değiştirmemek önemlidir. Dosyaların nasıl
> cache'lendiği hakkında daha fazla bilgi edinmek istersen [cache rehberimize](./manage-cache)
> bakabilirsin.

### En son sürümden

İndirilecek dosyayı `repo_id`, `repo_type` ve `filename` parametreleriyle seç.
Varsayılan olarak dosya bir `model` deposunun parçası kabul edilir.

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
'/root/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade/config.json'

# Download from a dataset
>>> hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
'/root/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34/fleurs.py'
```

### Belirli bir sürümden

Varsayılan olarak `main` dalındaki en son sürüm indirilir. Ancak bazı durumlarda
dosyayı belirli bir sürümde (örneğin belirli bir daldan, bir PR'dan, bir etiket
veya bir commit hash'inden) indirmek istersin.
Bunu yapmak için `revision` parametresini kullan:

```python
# Download from the `v1.0` tag
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")

# Download from the `test-branch` branch
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="test-branch")

# Download from Pull Request #3
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="refs/pr/3")

# Download from a specific commit hash
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
```

**Not:** Commit hash kullanırken, 7 karakterlik kısa hash yerine tam uzunlukta
hash kullanman gerekir.

### Bir indirme URL'si oluşturma

Bir depodan dosya indirmek için kullanılan URL'yi oluşturmak istiyorsan, bir URL
döndüren [`hf_hub_url`] fonksiyonunu kullanabilirsin.
Bunun dahili olarak [`hf_hub_download`] tarafından kullanıldığını unutma.

## Bir deponun tamamını indirme

[`snapshot_download`] belirli bir revision'da bir deponun tamamını indirir. Dahili
olarak [`hf_hub_download`] kullandığı için indirilen tüm dosyalar yerel diskinde
de cache'lenir. Süreci hızlandırmak için indirmeler eşzamanlı (concurrent)
yapılır.

Bir deponun tamamını indirmek için yalnızca `repo_id` ve `repo_type` geçmen yeterlidir:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'

# Or from a dataset
>>> snapshot_download(repo_id="google/fleurs", repo_type="dataset")
'/home/lysandre/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34'
```

[`snapshot_download`] varsayılan olarak en son revision'ı indirir. Belirli bir
depo revision'ı istiyorsan `revision` parametresini kullan:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", revision="refs/pr/1")
```

### İndirilecek dosyaları filtreleme

[`snapshot_download`] bir depoyu indirmenin kolay bir yolunu sunar. Ancak her zaman
bir deponun tüm içeriğini indirmek istemezsin. Örneğin, yalnızca `.safetensors`
ağırlıklarını kullanacağını biliyorsan tüm `.bin` dosyalarının indirilmesini
engellemek isteyebilirsin. Bunu `allow_patterns` ve `ignore_patterns` parametreleriyle
yapabilirsin.

Bu parametreler ya tek bir pattern ya da bir pattern listesi kabul eder. Pattern'ler
[burada](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm) belgelenen
Standart Wildcards (globbing pattern'leri) şeklindedir. Pattern eşleştirme
[`fnmatch`](https://docs.python.org/3/library/fnmatch.html) tabanlıdır.

Örneğin, yalnızca JSON yapılandırma dosyalarını indirmek için `allow_patterns`
kullanabilirsin:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")
```

Öte yandan `ignore_patterns`, belirli dosyaların indirilmesini hariç tutabilir.
Aşağıdaki örnek `.msgpack` ve `.h5` dosya uzantılarını yok sayar:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])
```

Son olarak, indirmeyi hassas şekilde filtrelemek için ikisini birleştirebilirsin.
İşte `vocab.json` hariç tüm json ve markdown dosyalarını indiren bir örnek.

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="gpt2", allow_patterns=["*.md", "*.json"], ignore_patterns="vocab.json")
```

## Dosya(ları) yerel bir klasöre indirme

Varsayılan olarak Hub'dan dosya indirmek için [cache sistemini](./manage-cache)
kullanmanı öneririz. [`hf_hub_download`] ve [`snapshot_download`] içinde `cache_dir`
parametresiyle veya [`HF_HOME`](../package_reference/environment_variables#hf_home)
ortam değişkenini ayarlayarak özel bir cache konumu belirtebilirsin.

Ancak dosyaları belirli bir klasöre indirmen gerekiyorsa, indirme fonksiyonuna bir
`local_dir` parametresi geçebilirsin. Bu, `git` komutunun sunduğuna daha yakın bir
iş akışı elde etmek için kullanışlıdır. İndirilen dosyalar, belirtilen klasör içinde
orijinal dosya yapılarını korur. Örneğin `filename="data/train.csv"` ve
`local_dir="path/to/folder"` ise oluşan dosya yolu `"path/to/folder/data/train.csv"`
olur.

Yerel dizininin kökünde, indirilen dosyalar hakkında metadata içeren bir
`.cache/huggingface/` klasörü oluşturulur. Bu, dosyalar zaten güncel ise yeniden
indirilmelerini önler. Metadata değişmişse yeni dosya sürümü indirilir. Bu,
`local_dir`'i yalnızca en son değişiklikleri çekmeye optimize eder.

İndirme tamamlandıktan sonra artık ihtiyacın yoksa `.cache/huggingface/` klasörünü
güvenle kaldırabilirsin. Ancak bu klasör olmadan script'ini yeniden çalıştırmanın,
metadata kaybolduğu için daha uzun kurtarma sürelerine yol açabileceğini unutma.
Yerel verilerinin bozulmadan ve etkilenmeden kalacağından emin olabilirsin.

> [!TIP]
> Hub'a değişiklik commit ederken `.cache/huggingface/` klasörü konusunda endişelenme!
> Bu klasör hem `git` hem de [`upload_folder`] tarafından otomatik olarak yok sayılır.

## CLI'dan indirme

Hub'dan dosyaları doğrudan indirmek için terminalden `hf download` komutunu
kullanabilirsin.
Dahili olarak yukarıda açıklanan aynı [`hf_hub_download`] ve [`snapshot_download`]
yardımcılarını kullanır ve döndürülen yolu terminale yazdırır.

```bash
>>> hf download gpt2 config.json
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

Birden fazla dosyayı bir kerede indirebilirsin; bu bir ilerleme çubuğu gösterir ve
dosyaların bulunduğu snapshot yolunu döndürür:

```bash
>>> hf download gpt2 config.json model.safetensors
Fetching 2 files: 100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 23831.27it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

Ayrıca tek bir `hf://` URI'si kullanarak depoyu (ve isteğe bağlı olarak bir revision
ve dosyayı) işaret edebilirsin. URI şu grameri izler: `hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]` (tam sözdizimi
için [HF URI'leri referansına](../package_reference/hf_uris) bak) ve yanında
ayarlanamayan `--repo-type` ile `--revision` seçeneklerinin yerini alır:

```bash
# Download a single file from a dataset at a given revision
>>> hf download hf://datasets/google/fleurs@refs/pr/1/fleurs.py

# Download a subfolder (note the trailing slash)
>>> hf download hf://datasets/google/fleurs/data/

# Download an entire repo
>>> hf download hf://datasets/google/fleurs
```

CLI indirme komutu hakkında daha fazla ayrıntı için lütfen [CLI rehberine](./cli#hf-download)
bak.

## Dry-run modu

Bazı durumlarda dosyaları gerçekten indirmeden önce hangilerinin indirileceğini
kontrol etmek istersin. Bunu `--dry-run` parametresiyle kontrol edebilirsin. Depodaki
indirilecek tüm dosyaları listeler ve zaten indirilip indirilmediklerini kontrol eder.
Bu, kaç dosyanın indirilmesi gerektiği ve boyutları hakkında bir fikir verir.

İşte tek bir dosya üzerinde kontrol eden bir örnek:

```sh
>>> hf download openai-community/gpt2 onnx/decoder_model_merged.onnx --dry-run
[dry-run] Will download 1 files (out of 1) totalling 655.2M
File                           Bytes to download
------------------------------ -----------------
onnx/decoder_model_merged.onnx 655.2M
```

Ve dosya zaten cache'de ise:

```sh
>>> hf download openai-community/gpt2 onnx/decoder_model_merged.onnx --dry-run
[dry-run] Will download 0 files (out of 1) totalling 0.0.
File                           Bytes to download
------------------------------ -----------------
onnx/decoder_model_merged.onnx -
```

Ayrıca bir deponun tamamında dry-run çalıştırabilirsin:

```sh
>>> hf download openai-community/gpt2 --dry-run
[dry-run] Fetching 26 files: 100%|█████████████| 26/26 [00:04<00:00,  6.26it/s]
[dry-run] Will download 11 files (out of 26) totalling 5.6G.
File                              Bytes to download
--------------------------------- -----------------
.gitattributes                    -
64-8bits.tflite                   125.2M
64-fp16.tflite                    248.3M
64.tflite                         495.8M
README.md                         -
config.json                       -
flax_model.msgpack                497.8M
generation_config.json            -
merges.txt                        -
model.safetensors                 548.1M
onnx/config.json                  -
onnx/decoder_model.onnx           653.7M
onnx/decoder_model_merged.onnx    655.2M
onnx/decoder_with_past_model.onnx 653.7M
onnx/generation_config.json       -
onnx/merges.txt                   -
onnx/special_tokens_map.json      -
onnx/tokenizer.json               -
onnx/tokenizer_config.json        -
onnx/vocab.json                   -
pytorch_model.bin                 548.1M
rust_model.ot                     702.5M
tf_model.h5                       497.9M
tokenizer.json                    -
tokenizer_config.json             -
vocab.json                        -
```

Ve dosya filtrelemeyle:

```sh
>>> hf download openai-community/gpt2 --include "*.json"  --dry-run
[dry-run] Fetching 11 files: 100%|█████████████| 11/11 [00:00<00:00, 80518.92it/s]
[dry-run] Will download 0 files (out of 11) totalling 0.0.
File                         Bytes to download
---------------------------- -----------------
config.json                  -
generation_config.json       -
onnx/config.json             -
onnx/generation_config.json  -
onnx/special_tokens_map.json -
onnx/tokenizer.json          -
onnx/tokenizer_config.json   -
onnx/vocab.json              -
tokenizer.json               -
tokenizer_config.json        -
vocab.json                   -
```

Son olarak, [`hf_hub_download`] ve [`snapshot_download`] fonksiyonlarına
`dry_run=True` geçerek programatik olarak da dry-run yapabilirsin. Her dosya için
commit hash'ini, dosya adını ve dosya boyutunu, dosyanın cache'de olup olmadığını
ve dosyanın indirilip indirilmeyeceğini içeren bir [`DryRunFileInfo`] (sırasıyla bir
[`DryRunFileInfo`] listesi) döndürür. Pratikte dosya, cache'de değilse veya
`force_download=True` geçilmişse indirilir.

## Daha hızlı indirmeler

Daha hızlı indirmeler ve yüklemeler için chunk tabanlı deduplication sağlayan
[`xet-core`](https://github.com/huggingface/xet-core) kütüphanesinin Python bağlayıcısı
olan `hf_xet` ile daha hızlı indirmelerden yararlan. `hf_xet`, `huggingface_hub` ile
sorunsuz entegre olur; ancak LFS yerine Rust `xet-core` kütüphanesini ve Xet
depolamasını kullanır.

`hf_xet`, dosyaları değişmez chunk'lara ayıran, bu chunk koleksiyonlarını (block veya
xorb denir) uzaktan depolayan ve istendiğinde dosyayı yeniden birleştirmek için
onları alan Xet depolama sistemini kullanır. İndirme sırasında, kullanıcının
dosyalara erişim yetkisi olduğunu doğruladıktan sonra `hf_xet`, bu dosya için LFS
SHA256 hash'iyle Xet content-addressable service'ini (CAS) sorgulayarak bu dosyaları
birleştirmek için reconstruction metadata (xorb'lar içindeki aralıklar) ile birlikte
xorb'ları doğrudan indirmek için presigned URL'ler alır. Ardından `hf_xet` gerekli
xorb aralıklarını verimli şekilde indirir ve dosyaları diske yazar.

Etkinleştirmek için yalnızca `huggingface_hub`'ın en son sürümünü kurman yeterlidir:

```bash
pip install -U "huggingface_hub"
```

`huggingface_hub` 0.32.0 itibarıyla bu, `hf_xet`'i de kurar.

Diğer tüm `huggingface_hub` API'leri herhangi bir değişiklik olmadan çalışmaya devam
edecektir. Xet depolamanın ve `hf_xet`'in faydaları hakkında daha fazla bilgi edinmek
için bu [bölüme](https://huggingface.co/docs/hub/xet/index) bak.

Not: `hf_transfer` eskiden LFS depolama backend'iyle kullanılıyordu ve artık
kullanımdan kaldırılmıştır; bunun yerine `hf_xet` kullan.
