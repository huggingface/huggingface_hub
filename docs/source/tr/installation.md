<!--⚠️ Bu dosya Markdown biçiminde olsa da dokümantasyon oluşturucumuza özgü söz dizimi içerir (MDX'e benzer).
Bu nedenle Markdown görüntüleyicinizde düzgün işlenmeyebilir.
-->

# Kurulum

Başlamadan önce uygun paketleri yükleyerek ortamınızı hazırlamanız gerekir.

`huggingface_hub`, **Python 3.10+** sürümlerinde test edilir.

## pip ile kurulum

`huggingface_hub` kütüphanesini bir [sanal
ortama](https://docs.python.org/3/library/venv.html) kurmanız önemle tavsiye edilir.
Python sanal ortamlarına aşina değilseniz bu [rehbere](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
göz atın. Sanal ortam; farklı projeleri yönetmeyi kolaylaştırır ve bağımlılıklar
arasındaki uyumluluk sorunlarını önler.

İlk olarak proje dizininizde bir sanal ortam oluşturun:

```bash
python -m venv .venv
```

Sanal ortamı etkinleştirin. Linux ve macOS'ta:

```bash
source .venv/bin/activate
```

Windows'ta:

```bash
.venv/Scripts/activate
```

Artık `huggingface_hub` kütüphanesini [PyPI kayıt
sisteminden](https://pypi.org/project/huggingface-hub/) yüklemeye hazırsınız:

```bash
pip install --upgrade huggingface_hub
```

İşlem tamamlandığında kurulumun doğru çalıştığını [kontrol edin](#check-installation).

### İsteğe bağlı bağımlılıkları yükleme

`huggingface_hub` kütüphanesinin bazı bağımlılıkları
[isteğe bağlıdır](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies);
çünkü kütüphanenin temel özelliklerini çalıştırmak için bunlara ihtiyaç duyulmaz. Ancak
isteğe bağlı bağımlılıklar kurulmazsa bazı `huggingface_hub` özellikleri kullanılamayabilir.

İsteğe bağlı bağımlılıkları `pip` ile yükleyebilirsiniz:

```bash
# Install dependencies for both torch-specific and MCP-specific features.
pip install 'huggingface_hub[mcp,torch]'
```

`huggingface_hub` içindeki isteğe bağlı bağımlılıklar şunlardır:

- `fastai`, `torch`: framework'e özgü özellikleri çalıştırmak için gereken bağımlılıklar.
- `dev`: kütüphaneye katkıda bulunmak için gereken bağımlılıklar. Testleri çalıştırmak
  için `testing`, tür denetleyicisini çalıştırmak için `typing` ve linter'ları çalıştırmak
  için `quality` bağımlılıklarını içerir.

### Kaynak koddan kurulum

Bazı durumlarda `huggingface_hub` kütüphanesini doğrudan kaynak koddan kurmak
isteyebilirsiniz. Böylece en son kararlı sürüm yerine en güncel `main` sürümünü
kullanabilirsiniz. Örneğin bir hata son resmi sürümden sonra giderilmiş ancak henüz yeni
bir sürüm yayımlanmamışsa `main` sürümü güncel gelişmeleri takip etmek için yararlıdır.

Ancak bu, `main` sürümünün her zaman kararlı olmayabileceği anlamına gelir. `main`
sürümünü çalışır durumda tutmaya özen gösteriyoruz ve çoğu sorun genellikle birkaç saat
veya bir gün içinde çözülüyor. Bir sorunla karşılaşırsanız daha hızlı çözebilmemiz için
lütfen bir issue açın!

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

Kaynak koddan kurulum yaparken belirli bir dalı da seçebilirsiniz. Bu seçenek, henüz
birleştirilmemiş yeni bir özelliği veya hata düzeltmesini test etmek istediğinizde
yararlıdır:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

İşlem tamamlandığında kurulumun doğru çalıştığını [kontrol edin](#check-installation).

### Düzenlenebilir kurulum

Kaynak koddan kurulum, [düzenlenebilir bir
kurulum](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)
oluşturmanıza da olanak tanır. `huggingface_hub` projesine katkıda bulunmayı ve kod
değişikliklerini test etmeyi planlıyorsanız bu daha gelişmiş kurulum yöntemini
kullanabilirsiniz. Bunun için `huggingface_hub` deposunun yerel bir kopyasını
bilgisayarınıza klonlamanız gerekir.

```bash
# First, clone repo locally
git clone https://github.com/huggingface/huggingface_hub.git

# Then, install with -e flag
cd huggingface_hub
pip install -e .
```

Bu komutlar, klonladığınız klasörü Python kütüphane yollarınıza bağlar. Python bundan
sonra normal kütüphane yollarının yanı sıra klonladığınız klasörün içine de bakar.
Örneğin Python paketleriniz genellikle `./.venv/lib/python3.13/site-packages/` dizinine
kuruluyorsa Python, klonladığınız `./huggingface_hub/` klasöründe de arama yapar.

## Hugging Face CLI aracını yükleme

Python ortamınıza dokunmadan `hf` CLI aracını kurmak için tek satırlık yükleyicilerimizi kullanın.

macOS ve Linux'ta:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Windows'ta:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Mevcut bir kurulumu yükseltmek için `hf update` komutunu çalıştırın. Bu komut `hf`
kurulum yöntemini (bağımsız kurucu, Homebrew veya pip) algılar ve uygun komutu çalıştırır.

## conda ile kurulum

conda'yı daha iyi biliyorsanız `huggingface_hub` kütüphanesini [conda-forge
kanalını](https://anaconda.org/conda-forge/huggingface_hub) kullanarak yükleyebilirsiniz:

```bash
conda install -c conda-forge huggingface_hub
```

İşlem tamamlandığında kurulumun doğru çalıştığını [kontrol edin](#check-installation).

## Kurulumu kontrol etme [[check-installation]]

Kurulumdan sonra aşağıdaki komutu çalıştırarak `huggingface_hub` kütüphanesinin düzgün
çalıştığını kontrol edin:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

Bu komut, [gpt2](https://huggingface.co/gpt2) modeli hakkındaki bilgileri Hub'dan alır.
Çıktı aşağıdakine benzer:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Windows sınırlamaları

Makine öğrenmesini her yerde erişilebilir kılma hedefimiz doğrultusunda `huggingface_hub`
kütüphanesini platformlar arası çalışacak, özellikle hem Unix tabanlı sistemlerde hem de
Windows'ta düzgün çalışacak biçimde geliştirdik. Ancak
`huggingface_hub` Windows'ta çalıştırıldığında bazı sınırlamalara sahiptir. Bilinen
sorunların tam listesini aşağıda bulabilirsiniz. Belgelenmemiş bir sorunla
karşılaşırsanız [GitHub'da bir issue
açarak](https://github.com/huggingface/huggingface_hub/issues/new/choose) lütfen bize
bildirin.

- `huggingface_hub` önbellek sistemi, Hub'dan indirilen dosyaları verimli biçimde
  önbelleğe almak için sembolik bağlantılardan yararlanır. Windows'ta sembolik
  bağlantıları etkinleştirmek için geliştirici modunu açmanız veya betiğinizi yönetici
  olarak çalıştırmanız gerekir. Bunlar etkin değilse önbellek sistemi çalışmaya devam
  eder ancak daha az verimli olur. Ayrıntılı bilgi için
  [önbellek sınırlamaları](./guides/manage-cache#limitations) bölümünü okuyun.
- Hub'daki dosya yollarında özel karakterler bulunabilir (örneğin
  `"path/to?/my/file"`). Windows'un [özel
  karakterler](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)
  konusundaki daha katı kuralları bu dosyaların indirilmesini engeller. Neyse ki bu
  durumla nadiren karşılaşılır. Bunun bir hata olduğunu düşünüyorsanız depo sahibiyle
  veya bir çözüm bulabilmemiz için bizimle iletişime geçin.

## Sonraki adımlar

`huggingface_hub` bilgisayarınıza düzgün biçimde kurulduktan sonra başlangıç için [ortam
değişkenlerini yapılandırabilir](package_reference/environment_variables) veya
[rehberlerimizden birine](guides/overview) göz atabilirsiniz.
