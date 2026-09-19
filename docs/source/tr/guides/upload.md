<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hub'a dosya yükleme

Dosyalarını ve çalışmalarını paylaşmak Hub'ın önemli bir parçasıdır. `huggingface_hub`, dosyalarını Hub'a yüklemek için çeşitli seçenekler sunar. Bu fonksiyonları bağımsız olarak kullanabilir veya kendi kütüphanene entegre ederek kullanıcılarının Hub ile etkileşimini kolaylaştırabilirsin.

Hub'a dosya yüklemek istediğin her zaman Hugging Face hesabına giriş yapman gerekir. Kimlik doğrulama hakkında daha fazla ayrıntı için [bu bölüme](../quick-start#authentication) bak.

## Bir dosya yükleme

[`create_repo`] ile bir depo oluşturduktan sonra, [`upload_file`] kullanarak depona bir dosya yükleyebilirsin.

Yüklenecek dosyanın yolunu, depoda dosyayı nereye yüklemek istediğini ve dosyayı eklemek istediğin deponun adını belirt. Depo türüne bağlı olarak isteğe bağlı olarak depo türünü `dataset`, `model` veya `space` olarak ayarlayabilirsin.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/path/to/local/folder/README.md",
...     path_in_repo="README.md",
...     repo_id="username/test-dataset",
...     repo_type="dataset",
... )
```

## Bir klasör yükleme

Yerel bir klasörü mevcut bir depoya yüklemek için [`upload_folder`] fonksiyonunu kullan. Yüklenecek yerel klasörün yolunu, depoda klasörü nereye yüklemek istediğini ve klasörü eklemek istediğin deponun adını belirt. Depo türüne bağlı olarak isteğe bağlı olarak depo türünü `dataset`, `model` veya `space` olarak ayarlayabilirsin.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
>>> api.upload_folder(
...     folder_path="/path/to/local/space",
...     repo_id="username/my-cool-space",
...     repo_type="space",
... )
```

Varsayılan olarak hangi dosyaların commit edileceğini belirlemek için `.gitignore` dosyası dikkate alınır. Varsayılan olarak bir commit'te `.gitignore` dosyası olup olmadığına bakarız; yoksa Hub'da var olup olmadığını kontrol ederiz. Yalnızca dizinin kökünde bulunan bir `.gitignore` dosyasının kullanılacağını unutma. Alt dizinlerdeki `.gitignore` dosyalarını kontrol etmeyiz.

Sabit kodlanmış bir `.gitignore` dosyası kullanmak istemiyorsan, hangi dosyaların yükleneceğini filtrelemek için `allow_patterns` ve `ignore_patterns` argümanlarını kullanabilirsin. Bu parametreler ya tek bir pattern ya da bir pattern listesi kabul eder. Pattern'ler [burada](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm) belgelenen Standart Wildcards (globbing pattern'leri) şeklindedir. Hem `allow_patterns` hem de `ignore_patterns` sağlanırsa her iki kısıt da uygulanır.

`.gitignore` dosyası ve allow/ignore pattern'lerinin yanı sıra, herhangi bir alt dizinde bulunan `.git/` klasörü yok sayılır.

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder",
...     path_in_repo="my-dataset/train", # Upload to a specific folder
...     repo_id="username/test-dataset",
...     repo_type="dataset",
...     ignore_patterns="**/logs/*.txt", # Ignore all text logs
... )
```

Aynı commit içinde depodan silmek istediğin dosyaları belirtmek için `delete_patterns` argümanını da kullanabilirsin. Bu, uzaktaki bir klasörü dosya push etmeden önce temizlemek istediğinde ve hangi dosyaların zaten var olduğunu bilmediğinde faydalı olabilir.

Aşağıdaki örnek yerel `./logs` klasörünü uzaktaki `/experiment/logs/` klasörüne yükler. Yalnızca txt dosyaları yüklenir ancak bundan önce depodaki önceki tüm log'lar silinir. Tüm bunlar tek bir commit'te gerçekleşir.
```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder/logs",
...     repo_id="username/trained-model",
...     path_in_repo="experiment/logs/",
...     allow_patterns="*.txt", # Upload all local text files
...     delete_patterns="*.txt", # Delete all remote text files before
... )
```

### Dosyalar nasıl yüklenir

`hf_xet` kurulu olduğunda (ki varsayılan durum budur), [`upload_folder`] dosyaları akışlı bir pipeline üzerinden yükler: dosyalar Hub'a karşı kontrol edilir, Xet depolama backend'ine yüklenir (bu backend chunk'lara ayırır, deduplicate eder ve transferleri dahili olarak yeniden dener) ve uyarlanabilir batch'ler halinde commit edilir; tüm bunlar paralel olarak yapılır. Pratikte bu şu anlama gelir:

- **Her boyuttaki klasörler**: küçük klasörler tek bir commit'te yüklenirken, çok sayıda dosya içeren klasörler sunucu limitlerinin altında kalmak için otomatik olarak birkaç commit'e bölünür. Bu olduğunda sonraki commit'lerin mesajına ` (part 2)`, ` (part 3)`, ... soneki eklenir.
- **Devam ettirilebilir**: yükleme herhangi bir nedenle kesilirse, aynı çağrıyı yeniden çalıştırman yeterlidir. Zaten commit edilmiş dosyalar tespit edilip atlanır ve zaten yüklenmiş chunk'lar deduplicate edilir — bunları yeniden yüklemek (neredeyse) hiç veri transfer etmez. Yerel bir durum tutulmaz: hatta farklı bir makineden devam edebilirsin. Bir istisna: `create_pr=True` ile yeniden çalıştırmak yeni bir pull request açar. Yüklemeye devam ederken bunun yerine `revision="refs/pr/N"` ile yeniden çalıştırmanı öneririz.
- **Çift okuma yok**: dosyalar yükleme için chunk'lara ayrılırken tek bir okuma geçişinde hash'lenir. Yükleme başlamadan önce ayrı bir "hashing" aşaması yoktur.

Canlı bir ilerleme göstergesi üç aşamayı takip eder:

```
Found 5,000 files to upload
  Preparing   ████████████████████  5,000 / 5,000 ✓
  Uploading   ██████████████░░░░░░  423 / 603 files  3.8GB · 19.7MB/s
  Committing  ██████████████████░░  4,580 / 5,000  6 commits
```

`hf_xet` kurulu değilse, [`upload_folder`] eski davranışa geri düşer: önce her şeyi hash'ler, HTTP üzerinden yükler, ardından tek bir commit oluşturur. Daha iyi dayanıklılık için `hf_xet`'i kurulu tutmanı her zaman öneririz

## CLI'dan yükleme

Hub'a dosyaları doğrudan yüklemek için terminalden `hf upload` komutunu kullanabilirsin. Dahili olarak yukarıda açıklanan aynı [`upload_file`] ve [`upload_folder`] yardımcılarını kullanır.

Tek bir dosya veya bir klasörün tamamını yükleyebilirsin:

```bash
# Usage:  hf upload [repo_id] [local_path] [path_in_repo]
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors

>>> hf upload Wauplin/my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

`local_path` ve `path_in_repo` isteğe bağlıdır ve örtük olarak çıkarılabilir. `local_path` ayarlanmamışsa, araç `repo_id` ile aynı ada sahip yerel bir klasör veya dosya olup olmadığını kontrol eder. Varsa içeriği yüklenir. Aksi halde kullanıcıdan açıkça `local_path` ayarlamasını isteyen bir exception yükseltilir. Her durumda, `path_in_repo` ayarlanmamışsa dosyalar deponun köküne yüklenir.

Hedef ayrıca `hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]` gramerini izleyen tek bir `hf://` URI'si olarak da ifade edilebilir (tam sözdizimi için [HF URI'leri referansına](../package_reference/hf_uris) bak). Depo türü, revision ve `path_in_repo` o zaman URI'den okunur; bunlar `--repo-type` ve `--revision` seçenekleriyle birlikte kullanılamaz:

```bash
# Upload a single file to a dataset on a specific branch
>>> hf upload hf://datasets/Wauplin/my-cool-dataset@my-branch/data/train.csv ./train.csv
https://huggingface.co/datasets/Wauplin/my-cool-dataset/blob/my-branch/data/train.csv
```

> [!TIP]
> Büyük dosyalarda maksimum yükleme throughput'u için [`HF_XET_HIGH_PERFORMANCE=1`](../package_reference/environment_variables.md#hf_xet_high_performance) ortam değişkenini ayarla. Bu, `hf_xet`'in yüksek performans modunu etkinleştirir; mevcut bant genişliğini ve CPU çekirdeklerini doyurur. Not: eski `HF_HUB_ENABLE_HF_TRANSFER=1` bayrağı artık kullanılmaz çünkü `hf_transfer`, `hf_xet` lehine kaldırılmıştır — bunun yerine `HF_XET_HIGH_PERFORMANCE=1` ayarla.

CLI yükleme komutu hakkında daha fazla ayrıntı için lütfen [CLI rehberine](./cli#hf-upload) bak.

## Büyük bir klasör yükleme

[`upload_folder`] ve `hf upload` komutu, çok büyük klasörler dahil Hub'a dosya yüklemek için başvurulan çözümlerdir. Dosyalar Hub'a birkaç commit'te akışla aktarılır ve süreç kesilirse otomatik olarak devam eder. Aynı çağrıyı yeniden çalıştırman yeterlidir; zaten yüklenmiş dosyalar atlanır.

```py
>>> api.upload_folder(
...     repo_id="HuggingFaceM4/Docmatix",
...     repo_type="dataset",
...     folder_path="/path/to/local/docmatix",
... )
```

veya terminalden:

```sh
hf upload HuggingFaceM4/Docmatix --repo-type=dataset /path/to/local/docmatix
```

> [!WARNING]
> Eski [`upload_large_folder`] metodu ve `hf upload-large-folder` komutu **kullanımdan kaldırılmıştır** ve gelecek bir sürümde silinecektir. Bunun yerine [`upload_folder`] / `hf upload` kullan.

### Büyük yüklemeler için ipuçları ve püf noktaları

Deponda büyük miktarda veriyle çalışırken bilmen gereken bazı sınırlamalar vardır. Veriyi akışla aktarmak zaman aldığı için sürecin sonunda bir yükleme/push'un başarısız olması veya hf.co'da ya da yerelde çalışırken kötüleşmiş bir deneyimle karşılaşmak oldukça sinir bozucu olabilir.

Hub'da depolarını nasıl yapılandıracağın konusunda en iyi uygulamalar için [Repository limitations and recommendations](https://huggingface.co/docs/hub/repositories-recommendations) rehberimize bak. Şimdi yükleme sürecini mümkün olduğunca sorunsuz hale getirmek için bazı pratik ipuçlarına geçelim.

- **Küçük başla**: Yükleme script'ini test etmek için az miktarda veriyle başlamanı öneririz. Başarısızlık yalnızca az zaman aldığında bir script üzerinde yinelemek daha kolaydır.
- **Başarısızlıklara hazır ol**: Büyük miktarda veriyi akışla aktarmak zordur. Ne olabileceğini bilmezsin ama en az bir kez bir şeylerin başarısız olacağını varsaymak her zaman en iyisidir — ister makineninden, ister bağlantından, ister sunucularımızdan kaynaklansın. Örneğin çok sayıda dosya yüklemeyi planlıyorsan, bir sonraki batch'i yüklemeden önce hangi dosyaları zaten yüklediğini yerel olarak takip etmek en iyisidir. Zaten commit edilmiş bir LFS dosyasının asla iki kez yeniden yüklenmeyeceğinden emin olabilirsin ama bunu istemci tarafında kontrol etmek yine de zaman kazandırabilir. [`upload_folder`] senin için bunu yapar.
- **`hf_xet` kullan**: bu, Hub için yeni depolama backend'inden yararlanır, Rust ile yazılmıştır ve artık herkesin kullanımına açıktır. Aslında `hf_xet`, `huggingface_hub` kullanırken varsayılan olarak zaten etkindir! Maksimum performans için [`HF_XET_HIGH_PERFORMANCE=1`](../package_reference/environment_variables.md#hf_xet_high_performance) ortam değişkenini ayarla. Yüksek performans modu etkinleştirildiğinde aracın mevcut tüm bant genişliğini ve CPU çekirdeklerini kullanmaya çalışacağını unutma.

## Gelişmiş özellikler

Çoğu durumda Hub'a dosya yüklemek için [`upload_file`] ve [`upload_folder`]'dan fazlasına ihtiyacın olmaz.
Ancak `huggingface_hub`, işleri kolaylaştırmak için daha gelişmiş özelliklere sahiptir. Onlara bir göz atalım!

### Daha hızlı yüklemeler

Daha hızlı yüklemeler ve indirmeler için chunk tabanlı deduplication sağlayan [`xet-core`](https://github.com/huggingface/xet-core) kütüphanesinin Python bağlayıcısı olan `hf_xet` ile daha hızlı yüklemelerden yararlan. `hf_xet`, `huggingface_hub` ile sorunsuz entegre olur; ancak LFS yerine Rust `xet-core` kütüphanesini ve Xet depolamasını kullanır.

`hf_xet`, dosyaları değişmez chunk'lara ayıran, bu chunk koleksiyonlarını (block veya xorb denir) uzaktan depolayan ve istendiğinde dosyayı yeniden birleştirmek için onları alan Xet depolama sistemini kullanır. Yükleme sırasında, kullanıcının bu depoya yazma yetkisi olduğunu doğruladıktan sonra `hf_xet` dosyaları tarar, onları chunk'larına ayırır ve bu chunk'ları xorb'larda toplar (ve bilinen chunk'lar arasında deduplicate eder), ardından bu xorb'ları Xet content-addressable service'ine (CAS) yükler; CAS xorb'ların bütünlüğünü doğrular, xorb metadata'sını LFS SHA256 hash'iyle birlikte kaydeder (lookup/download'ı desteklemek için) ve xorb'ları uzak depolamaya yazar.

Etkinleştirmek için yalnızca `huggingface_hub`'ın en son sürümünü kurman yeterlidir:

```bash
pip install -U "huggingface_hub"
```

`huggingface_hub` 0.32.0 itibarıyla bu, `hf_xet`'i de kurar.

Diğer tüm `huggingface_hub` API'leri herhangi bir değişiklik olmadan çalışmaya devam edecektir. Xet depolamanın ve `hf_xet`'in faydaları hakkında daha fazla bilgi edinmek için bu [bölüme](https://huggingface.co/docs/hub/xet/index) bak.

**Küme / Dağıtık Dosya Sistemi Yükleme Dikkatleri**

Bir kümeden yükleme yaparken, yüklenen dosyalar genellikle dağıtık veya ağ dosya sisteminde (NFS, EBS, Lustre, Fsx vb.) bulunur. Xet depolama bu dosyaları chunk'lara ayırır ve onları yerel olarak block'lara (xorb da denir) yazar; block tamamlandığında yükler. Dağıtık bir dosya sisteminden yükleme yaparken daha iyi performans için [`HF_XET_CACHE`](../package_reference/environment_variables#hfxetcache) değerini yerel bir diskteki (ör. yerel bir NVMe veya SSD disk) bir dizine ayarladığından emin ol. Xet cache'inin varsayılan konumu `HF_HOME` altındadır (`~/.cache/huggingface/xet`) ve kullanıcının home dizininde olması nedeniyle bu konum da sıklıkla dağıtık dosya sistemi üzerindedir.

### Engellemeyen yüklemeler

Bazı durumlarda ana thread'ini engellemeden veri push etmek istersin. Bu, bir eğitimi sürdürürken log ve artifact yüklemek için özellikle kullanışlıdır. Bunu yapmak için hem [`upload_file`] hem de [`upload_folder`] içinde `run_as_future` argümanını kullanabilirsin. Bu, yüklemenin durumunu kontrol etmek için kullanabileceğin bir [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects) nesnesi döndürür.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.upload_folder( # Upload in the background (non-blocking action)
...     repo_id="username/my-model",
...     folder_path="checkpoints-001",
...     run_as_future=True,
... )
>>> future
Future(...)
>>> future.done()
False
>>> future.result() # Wait for the upload to complete (blocking action)
...
```

> [!TIP]
> `run_as_future=True` kullanıldığında arka plan işleri kuyruğa alınır. Bu, işlerin doğru sırada yürütüleceğinden emin olman anlamına gelir.

Arka plan işleri çoğunlukla veri yüklemek/commit oluşturmak için kullanışlı olsa da, [`run_as_future`] kullanarak istediğin herhangi bir metodu kuyruğa alabilirsin. Örneğin bir depo oluşturmak ve ardından arka planda ona veri yüklemek için kullanabilirsin. Yükleme metodlarındaki yerleşik `run_as_future` argümanı yalnızca bunun etrafında bir alias'tır.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.run_as_future(api.create_repo, "username/my-model", exists_ok=True)
Future(...)
>>> api.upload_file(
...     repo_id="username/my-model",
...     path_in_repo="file.txt",
...     path_or_fileobj=b"file content",
...     run_as_future=True,
... )
Future(...)
```

### Depolar arasında dosya kopyalama

Hub'daki depolar arasında dosya veya klasör kopyalamak için büyük veriyi indirmeden veya yeniden yüklemeden [`copy_files`] kullan. Bu, model varyantları arasında ağırlıkları çoğaltmak, depolar arasında dataset dosyalarını kopyalamak veya depolarındaki dosyaları yeniden düzenlemek istediğinde kullanışlıdır. Altta [`CommitOperationCopy`] işlemleriyle bir commit oluşturur.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Copy a single file between repos
>>> api.copy_files(
...     "hf://username/source-model/weights.safetensors",
...     "hf://username/target-model/weights.safetensors",
... )

# Copy an entire folder
>>> api.copy_files(
...     "hf://datasets/username/source-dataset/data/",
...     "hf://datasets/username/target-dataset/data/",
... )
```

Aynı depo içinde de kopyalama yapabilirsin:

```py
# Duplicate a file in the same repo
>>> api.copy_files(
...     "hf://username/my-model/config.json",
...     "hf://username/my-model/backup/config.json",
... )
```

> [!TIP]
> Bir klasör kopyalarken, kaynakta sondaki `/` rsync tarzı semantik kullanır; yani klasörün *içeriği* kopyalanır, klasörün kendisi iç içe eklenmez. Sondaki `/` olmadan klasörün kendisi hedefte iç içe yerleştirilir.

> [!TIP]
> [`copy_files`] ayrıca dosyaları [Bucket'lara](./buckets) kopyalamayı da destekler. Daha fazla ayrıntı için [Buckets rehberine](./buckets#copy-files-to-bucket) bak.

### Zamanlanmış yüklemeler

Hugging Face Hub, veriyi kaydetmeyi ve sürümlendirmeyi kolaylaştırır. Ancak aynı dosyayı binlerce kez güncellerken bazı sınırlamalar vardır. Örneğin bir eğitim sürecinin log'larını veya konuşlandırılmış bir Space'teki kullanıcı geri bildirimlerini kaydetmek isteyebilirsin. Bu durumlarda veriyi Hub'da bir dataset olarak yüklemek mantıklıdır ama bunu düzgün yapmak zor olabilir. Ana neden, verinin her güncellemesini sürümlendirmek istememendir çünkü bu git deposunu kullanılamaz hale getirir. [`CommitScheduler`] sınıfı bu soruna bir çözüm sunar.

Fikir, yerel bir klasörü düzenli olarak Hub'a push eden bir arka plan işi çalıştırmaktır. Diyelim ki girdi olarak biraz metin alan ve bunun iki çevirisini üreten bir Gradio Space'in var. Ardından kullanıcı tercih ettiği çeviriyi seçebilir. Her çalıştırmada sonuçları analiz etmek için girdiyi, çıktıyı ve kullanıcı tercihini kaydetmek istiyorsun. Bu, [`CommitScheduler`] için mükemmel bir kullanım senaryosudur; veriyi Hub'a kaydetmek istiyorsun (potansiyel olarak milyonlarca kullanıcı geri bildirimi) ama her kullanıcının girdisini gerçek zamanlı kaydetmene _gerek yok_. Bunun yerine veriyi yerel olarak bir JSON dosyasına kaydedebilir ve her 10 dakikada bir yükleyebilirsin. Örneğin:

```py
>>> import json
>>> import uuid
>>> from pathlib import Path
>>> import gradio as gr
>>> from huggingface_hub import CommitScheduler

# Define the file where to save the data. Use UUID to make sure not to overwrite existing data from a previous run.
>>> feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
>>> feedback_folder = feedback_file.parent

# Schedule regular uploads. Remote repo and local folder are created if they don't already exist.
>>> scheduler = CommitScheduler(
...     repo_id="report-translation-feedback",
...     repo_type="dataset",
...     folder_path=feedback_folder,
...     path_in_repo="data",
...     every=10,
... )

# Define the function that will be called when the user submits its feedback (to be called in Gradio)
>>> def save_feedback(input_text:str, output_1: str, output_2:str, user_choice: int) -> None:
...     """
...     Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
...     """
...     with scheduler.lock:
...         with feedback_file.open("a") as f:
...             f.write(json.dumps({"input": input_text, "output_1": output_1, "output_2": output_2, "user_choice": user_choice}))
...             f.write("\n")

# Start Gradio
>>> with gr.Blocks() as demo:
>>>     ... # define Gradio demo + use `save_feedback`
>>> demo.launch()
```

Ve bu kadar! Kullanıcı girdi/çıktıları ve geri bildirimi Hub'da bir dataset olarak kullanılabilir olacak. Benzersiz bir JSON dosya adı kullanarak önceki bir çalıştırmadan veya aynı depoya eşzamanlı push eden başka Space'lerden/replica'lardan gelen verinin üzerine yazmayacağından emin olabilirsin.

[`CommitScheduler`] hakkında bilmen gerekenler şunlardır:
- **append-only:**
    Klasöre yalnızca içerik ekleyeceğin varsayılır. Yalnızca mevcut dosyalara veri eklemeli veya yeni dosyalar oluşturmalısın. Bir dosyayı silmek veya üzerine yazmak deponu bozabilir.
- **git history**:
    Zamanlayıcı klasörü her `every` dakikada bir commit eder. Git deposunu fazla kirletmemek için minimum 5 dakika değeri ayarlaman önerilir. Ayrıca zamanlayıcı boş commit'lerden kaçınacak şekilde tasarlanmıştır. Klasörde yeni içerik tespit edilmezse zamanlanmış commit düşürülür.
- **errors:**
    Zamanlayıcı arka plan thread'i olarak çalışır. Sınıfı örneklediğinde başlar ve asla durmaz. Özellikle yükleme sırasında bir hata oluşursa (örnek: bağlantı sorunu), zamanlayıcı bunu sessizce yok sayar ve bir sonraki zamanlanmış commit'te yeniden dener.
- **thread-safety:**
    Çoğu durumda bir lock dosyası konusunda endişelenmeden bir dosyaya yazabileceğini varsaymak güvenlidir. Zamanlayıcı, yükleme yaparken klasöre içerik yazsan bile çökmez veya bozulmaz. Pratikte, yoğun yüklü uygulamalarda eşzamanlılık sorunlarının _olması mümkündür_. Bu durumda thread-safety sağlamak için `scheduler.lock` kilidini kullanmanı öneririz. Kilit yalnızca zamanlayıcı klasörü değişiklikler için taradığında bloke edilir, veri yüklerken değil. Space'indeki kullanıcı deneyimini etkilemeyeceğini güvenle varsayabilirsin.

#### Space kalıcılık demosu

Bir Space'ten Hub'daki bir Dataset'e veri kalıcı hale getirmek, [`CommitScheduler`] için ana kullanım senaryosudur. Kullanım senaryosuna bağlı olarak verini farklı şekilde yapılandırmak isteyebilirsin. Yapı, eşzamanlı kullanıcılara ve yeniden başlatmalara karşı dayanıklı olmalıdır; bu genellikle UUID üretmeyi gerektirir. Dayanıklılığın yanı sıra, veriyi daha sonra yeniden kullanmak için 🤗 Datasets kütüphanesi tarafından okunabilir bir formatta yüklemelisin. Birkaç farklı veri formatının nasıl kaydedileceğini gösteren bir [Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver) oluşturduk (kendi özel ihtiyaçların için uyarlaman gerekebilir).

#### Özel yüklemeler

[`CommitScheduler`], verinin append-only olduğunu ve "olduğu gibi" yükleneceğini varsayar. Ancak verinin yüklenme şeklini özelleştirmek isteyebilirsin. Bunu [`CommitScheduler`]'dan miras alan bir sınıf oluşturarak ve `push_to_hub` metodunu override ederek yapabilirsin (istediğin gibi override etmekte özgürsün). Her `every` dakikada bir arka plan thread'inde çağrılacağından emin olabilirsin. Eşzamanlılık ve hatalar konusunda endişelenmene gerek yoktur ama boş commit push etmek veya yinelenen veri gibi diğer konulara dikkat etmelisin.

Aşağıdaki (basitleştirilmiş) örnekte, Hub'daki depoyu aşırı yüklememek için tüm PNG dosyalarını tek bir arşivde zip'lemek üzere `push_to_hub`'ı override ediyoruz:

```py
class ZipScheduler(CommitScheduler):
    def push_to_hub(self):
        # 1. List PNG files
          png_files = list(self.folder_path.glob("*.png"))
          if len(png_files) == 0:
              return None  # return early if nothing to commit

        # 2. Zip png files in a single archive
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "train.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip:
                for png_file in png_files:
                    zip.write(filename=png_file, arcname=png_file.name)

            # 3. Upload archive
            self.api.upload_file(..., path_or_fileobj=archive_path)

        # 4. Delete local png files to avoid re-uploading them later
        for png_file in png_files:
            png_file.unlink()
```

`push_to_hub`'ı override ettiğinde [`CommitScheduler`]'ın özniteliklerine ve özellikle şunlara erişimin olur:
- [`HfApi`] istemcisi: `api`
- Klasör parametreleri: `folder_path` ve `path_in_repo`
- Depo parametreleri: `repo_id`, `repo_type`, `revision`
- Thread kilidi: `lock`

> [!TIP]
> Özel zamanlayıcı örnekleri için kullanım senaryolarına göre farklı uygulamalar içeren [demo Space'imize](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
> bak.

### create_commit

[`upload_file`] ve [`upload_folder`] fonksiyonları genellikle kullanışlı olan yüksek seviyeli API'lerdir. Daha düşük seviyede çalışmana gerek yoksa önce bu fonksiyonları denemeni öneririz. Ancak commit seviyesinde çalışmak istiyorsan doğrudan [`create_commit`] fonksiyonunu kullanabilirsin.

[`create_commit`] tarafından desteklenen üç işlem türü vardır:

- [`CommitOperationAdd`] Hub'a bir dosya yükler. Dosya zaten varsa dosya içeriğinin üzerine yazılır. Bu işlem iki argüman kabul eder:

  - `path_in_repo`: dosyanın yükleneceği depo yolu.
  - `path_or_fileobj`: dosya sistemindeki bir dosya yolu veya dosya benzeri bir nesne. Hub'a yüklenecek dosyanın içeriğidir.

- [`CommitOperationDelete`] bir depodan bir dosya veya klasör kaldırır. Bu işlem `path_in_repo` argümanını kabul eder.

- [`CommitOperationCopy`] bir depo içinde veya depolar arasında bir dosya kopyalar. Bu işlem şu argümanları kabul eder:

  - `src_path_in_repo`: kopyalanacak dosyanın depo yolu.
  - `path_in_repo`: dosyanın kopyalanacağı depo yolu.
  - `src_revision`: isteğe bağlı - farklı bir dal/revision'dan dosya kopyalamak istiyorsan kopyalanacak dosyanın revision'ı.
  - `src_repo_id`: isteğe bağlı - kopyalanacak kaynak depo (ör. `"username/source-model"`). Varsayılan olarak hedef depodur.
  - `src_repo_type`: isteğe bağlı - kaynak deponun türü (`"model"`, `"dataset"` veya `"space"`). `src_repo_id` ayarlandığında zorunludur.

Örneğin bir Hub deposunda iki dosya yüklemek ve bir dosya silmek istiyorsan:

1. Bir dosya eklemek veya silmek ve bir klasör silmek için uygun `CommitOperation`'ı kullan:

```py
>>> from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
>>> api = HfApi()
>>> operations = [
...     CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj="~/repo/LICENSE.md"),
...     CommitOperationAdd(path_in_repo="weights.h5", path_or_fileobj="~/repo/weights-final.h5"),
...     CommitOperationDelete(path_in_repo="old-weights.h5"),
...     CommitOperationDelete(path_in_repo="logs/"),
...     CommitOperationCopy(src_path_in_repo="image.png", path_in_repo="duplicate_image.png"),
... ]
```

2. İşlemlerini [`create_commit`]'e geç:

```py
>>> api.create_commit(
...     repo_id="lysandre/test-model",
...     operations=operations,
...     commit_message="Upload my model weights and license",
... )
```

[`upload_file`] ve [`upload_folder`]'a ek olarak aşağıdaki fonksiyonlar da altta [`create_commit`] kullanır:

- [`delete_file`] Hub'daki bir depodan tek bir dosya siler.
- [`delete_folder`] Hub'daki bir depodan bir klasörün tamamını siler.
- [`metadata_update`] bir deponun metadata'sını günceller.

Daha ayrıntılı bilgi için [`HfApi`] referansına bak.

### Commit öncesi LFS dosyalarını önceden yükleme

Bazı durumlarda commit çağrısını yapmadan **önce** büyük dosyaları S3'e yüklemek isteyebilirsin. Örneğin bellekte üretilen birkaç shard halinde bir dataset commit ediyorsan, bellek yetersizliği sorunundan kaçınmak için shard'ları tek tek yüklemen gerekir. Bir çözüm, her shard'ı depoda ayrı bir commit olarak yüklemektir. Tamamen geçerli olsa da bu çözümün, onlarca commit üreterek git geçmişini bozma potansiyeli gibi bir dezavantajı vardır. Bu sorunu aşmak için dosyalarını tek tek S3'e yükleyebilir ve sonunda tek bir commit oluşturabilirsin. Bu, [`preupload_lfs_files`]'ı [`create_commit`] ile birlikte kullanarak mümkündür.

> [!WARNING]
> Bu bir power-user metodudur. Dosyaları önceden yüklemenin düşük seviyeli mantığını ele almak yerine doğrudan [`upload_file`], [`upload_folder`] veya [`create_commit`] kullanmak vakaların büyük çoğunluğunda doğru yoldur. [`preupload_lfs_files`]'ın ana uyarısı, commit gerçekten yapılana kadar yüklenen dosyaların Hub'daki depoda erişilebilir olmamasıdır. Bir sorun varsa Discord'umuzda veya bir GitHub issue'sunda bize ping atmaktan çekinme.

Dosyaları önceden yüklemenin nasıl yapılacağını gösteren basit bir örnek:

```py
>>> from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo

>>> repo_id = create_repo("test_preupload").repo_id

>>> operations = [] # List of all `CommitOperationAdd` objects that will be generated
>>> for i in range(5):
...     content = ... # generate binary content
...     addition = CommitOperationAdd(path_in_repo=f"shard_{i}_of_5.bin", path_or_fileobj=content)
...     preupload_lfs_files(repo_id, additions=[addition])
...     operations.append(addition)

>>> # Create commit
>>> create_commit(repo_id, operations=operations, commit_message="Commit all shards")
```

Önce [`CommitOperationAdd`] nesnelerini tek tek oluşturuyoruz. Gerçek dünya örneğinde bunlar üretilen shard'ları içerirdi. Her dosya bir sonraki üretilmeden önce yüklenir. [`preupload_lfs_files`] adımı sırasında **`CommitOperationAdd` nesnesi mutate edilir**. Onu yalnızca doğrudan [`create_commit`]'e geçirmek için kullanmalısın. Nesnenin ana güncellemesi, **binary içeriğin ondan kaldırılmasıdır**; yani başka bir referans saklamazsan garbage-collect edilir. Bu beklenen bir durumdur çünkü zaten yüklenmiş içeriği bellekte tutmak istemeyiz. Son olarak tüm işlemleri [`create_commit`]'e geçirerek commit'i oluşturuyoruz. Henüz işlenmemiş ek işlemler (add, delete veya copy) de geçebilirsin ve bunlar doğru şekilde ele alınır.
