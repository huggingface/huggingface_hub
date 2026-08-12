<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hızlı başlangıç

[Hugging Face Hub](https://huggingface.co/), makine öğrenmesi modellerini,
demolarını, veri kümelerini ve metriklerini paylaşmak için başvurulan temel
platformdur. `huggingface_hub` kütüphanesi, geliştirme ortamından ayrılmadan Hub ile
etkileşim kurmana yardımcı olur. Kolayca depo oluşturup yönetebilir, dosya indirip
yükleyebilir ve Hub'daki modeller ile veri kümeleri hakkında yararlı meta veriler
edinebilirsin.

## Kurulum

Başlamak için `huggingface_hub` kütüphanesini yükle:

```bash
pip install --upgrade huggingface_hub
```

Daha fazla bilgi için [kurulum](installation) rehberine göz at.

> [!TIP]
> `huggingface_hub`, Hub ile doğrudan terminalden etkileşim kurmanı sağlayan bir
> [`hf` CLI](./guides/cli) ile birlikte gelir. Yapay zekâ ajanları (Claude Code,
> Codex, Cursor, ...) kullanıyorsan ajanının CLI'ı kullanabilmesi için Skill'i yükle:
> ```bash
> # Codex, Cursor, OpenCode, Pi ve skill dosyalarını `.agents/skills` içinden yükleyen diğer ajanlar için
> hf skills add
> # yukarıdakilere ek olarak Claude Code'u da kapsar
> hf skills add --claude
> ```
> Daha fazla bilgi için [Yapay zekâ ajanları için Hugging Face
> CLI](https://huggingface.co/docs/hub/agents-cli) rehberine göz at.

## Dosya indirme

Hub'daki depolar Git ile sürüm kontrolü altındadır. Kullanıcılar tek bir dosyayı
veya tüm depoyu indirebilir. Dosya indirmek için [`hf_hub_download`] fonksiyonunu
kullanabilirsin. Bu fonksiyon dosyayı indirir ve yerel diskinde önbelleğe alır.
Aynı dosyaya bir sonraki ihtiyaç duyduğunda dosya önbellekten yüklenir; böylece
dosyayı tekrar indirmen gerekmez.

İndirmek istediğin dosyanın depo kimliğine ve dosya adına ihtiyacın vardır. Örneğin
[Pegasus](https://huggingface.co/google/pegasus-xsum) modelinin yapılandırma
dosyasını indirmek için:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

Dosyanın belirli bir sürümünü indirmek için dal adını, etiketi veya commit hash'ini
`revision` parametresiyle belirt. Commit hash'ini kullanırsan 7 karakterlik kısa
hash yerine tam uzunluktaki hash'i vermelisin:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

Daha fazla bilgi ve seçenek için [`hf_hub_download`] API referansına bak.

<a id="login"></a> <!-- backward compatible anchor -->

## Kimlik doğrulama

Çoğu durumda Hub ile etkileşim kurmak için Hugging Face hesabınla kimliğini
doğrulaman gerekir. Buna özel depoları indirmek, dosya yüklemek ve PR oluşturmak
dahildir. Henüz hesabın yoksa [bir hesap oluştur](https://huggingface.co/join).

### Oturum açma komutu

Kimliğini doğrulamanın en kolay yolu [`login`] komutunu kullanmaktır:

```bash
hf auth login
```

Zaten oturum açtıysan komut hemen tamamlanır. Yeniden oturum açmaya zorlamak için
`hf auth login --force` komutunu kullan. Oturum açmadıysan tarayıcı üzerinden oturum
açman istenir: terminalde gösterilen URL'yi aç, kısa kodu gir ve isteği onayla. Bir
erişim token'ı alınarak `HF_HOME` dizinine kaydedilir (varsayılan konum
`~/.cache/huggingface/token`). Token'ın süresi bir süre sonra dolar ancak Hub'ı
kullanmaya devam ettiğin sürece otomatik olarak yenilenir. Hub ile etkileşim kuran
tüm betikler ve kütüphaneler istek gönderirken bu token'ı kullanır. Alternatif
olarak [Ayarlar sayfasından](https://huggingface.co/settings/tokens) oluşturduğun
bir [User Access Token](https://huggingface.co/docs/hub/security-tokens)
yapıştırabilirsin.

> [!TIP]
> User Access Token'lar `read` veya `write` iznine sahip olabilir. Depo oluşturmak
> ya da düzenlemek istiyorsan `write` erişimli bir token kullandığından emin ol.
> Aksi durumda token'ın yanlışlıkla sızması hâlindeki riski azaltmak için `read`
> erişimli bir token oluşturmak daha iyidir.

Alternatif olarak bir notebook ya da betik içinde [`login`] kullanarak program
aracılığıyla oturum açabilirsin:

```py
>>> from huggingface_hub import login
>>> login()
```

Aynı anda yalnızca bir hesapta oturum açabilirsin. Yeni bir hesapta oturum açtığında
önceki hesaptaki oturumun otomatik olarak kapatılır. Etkin hesabını öğrenmek için
`hf auth whoami` komutunu çalıştırman yeterlidir.

> [!WARNING]
> Oturum açtıktan sonra, kimlik doğrulaması gerektirmeyen metotlar da dahil olmak
> üzere Hub'a yapılan tüm istekler varsayılan olarak erişim token'ını kullanır.
> Token'ın örtük kullanımını devre dışı bırakmak için `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`
> ortam değişkenini ayarla ([referansa](../package_reference/environment_variables#hfhubdisableimplicittoken)
> bak).

### Birden fazla yerel token'ı yönetme

Her token ile [`login`] komutunu kullanarak ayrı ayrı oturum açabilir ve makinen
üzerinde birden fazla token kaydedebilirsin. Bu token'lar arasında geçiş yapmak için
[`auth switch`] komutunu kullan:

```bash
hf auth switch
```

Bu komut, kayıtlı token listesinden birini seçmeni ister. Seçtiğin token etkin hâle
gelir ve Hub ile yapılan tüm etkileşimlerde kullanılır.

Kullanılabilir tüm erişim token'larını `hf auth list` komutuyla listeleyebilirsin.

### Ortam değişkeni

Kimlik doğrulamak için `HF_TOKEN` ortam değişkenini de kullanabilirsin. Bu yöntem,
`HF_TOKEN` değerini bir [Space secret](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)
olarak tanımlayabildiğin Space'lerde özellikle kullanışlıdır.

> [!TIP]
> **YENİ:** Google Colaboratory, notebook'larında [gizli
> anahtarlar](https://twitter.com/GoogleColab/status/1719798406195867814)
> tanımlamana olanak tanır. Otomatik kimlik doğrulama için bir `HF_TOKEN` secret'ı
> tanımla!

Ortam değişkeni veya secret ile yapılan kimlik doğrulama, makinede kayıtlı token'a
göre önceliklidir.

### Metot parametreleri

Son olarak `token` parametresini kabul eden herhangi bir metoda token'ını vererek de
kimliğini doğrulayabilirsin.

```
from huggingface_hub import whoami

user = whoami(token=...)
```

Token'ı kalıcı olarak saklamak istemediğin veya aynı anda birden fazla token'ı
yönetmen gereken ortamlar dışında bu yöntem genellikle önerilmez.

> [!WARNING]
> Token'ları parametre olarak geçirirken dikkatli ol. Token'ı koduna sabitlemek
yerine güvenli bir kasadan yüklemek her zaman daha iyi bir uygulamadır. Koda
sabitlenmiş token'lar, kodunu yanlışlıkla paylaşman hâlinde ciddi bir sızıntı
riski oluşturur.

## Depo oluşturma

Kaydolup oturum açtıktan sonra [`create_repo`] fonksiyonuyla bir depo oluştur:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

Deponun özel olmasını istiyorsan:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

Özel depolar yalnızca senin tarafından görülebilir.

> [!TIP]
> Depo oluşturmak veya Hub'a içerik göndermek için `write` iznine sahip bir User
> Access Token sağlamalısın. İzni, token'ı [Ayarlar
> sayfasında](https://huggingface.co/settings/tokens) oluştururken seçebilirsin.

## Dosya yükleme

Yeni oluşturduğun depoya dosya eklemek için [`upload_file`] fonksiyonunu kullan.
Şunları belirtmen gerekir:

1. Yüklenecek dosyanın yolu.
2. Dosyanın depo içindeki yolu.
3. Dosyanın ekleneceği deponun kimliği.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

Aynı anda birden fazla dosya yüklemek için Git kullanan ve kullanmayan çeşitli
yöntemleri açıklayan [Yükleme](./guides/upload) rehberine göz at.

## Sonraki adımlar

`huggingface_hub` kütüphanesi, kullanıcıların Hub ile Python üzerinden kolayca
etkileşim kurmasını sağlar. Hub'daki dosya ve depolarını nasıl yönetebileceğin
hakkında daha fazla bilgi edinmek için [nasıl yapılır
rehberlerimizi](./guides/overview) okumanı öneririz:

- [Deponu yönet](./guides/repository).
- Hub'dan dosya [indir](./guides/download).
- Hub'a dosya [yükle](./guides/upload).
- İstediğin modeli veya veri kümesini bulmak için Hub'da [arama yap](./guides/search).
- Hugging Face Hub'da barındırılan modeller için birden fazla servis üzerinden
  [çıkarım yap](./guides/inference).
