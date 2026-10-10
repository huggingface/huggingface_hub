<!--⚠️ ध्यान दें कि यह फ़ाइल Markdown में है, लेकिन इसमें हमारे डॉक्यूमेंटेशन बिल्डर (MDX के समान) के लिए विशेष सिंटैक्स शामिल है, जो आपके Markdown व्यूअर में सही तरीके से प्रदर्शित नहीं हो सकता।
-->
# Filesystem API के माध्यम से Hub के साथ इंटरैक्ट करें
 
[`HfApi`] के अलावा, `huggingface_hub` लाइब्रेरी [`HfFileSystem`] भी प्रदान करती है, जो Hugging Face Hub के लिए एक Pythonic [fsspec-संगत](https://filesystem-spec.readthedocs.io/en/latest/) फ़ाइल इंटरफ़ेस है। [`HfFileSystem`] [`HfApi`] के ऊपर निर्मित है और `cp`, `mv`, `ls`, `du`, `glob`, `get_file` तथा `put_file` जैसी सामान्य फ़ाइल सिस्टम संचालन सुविधाएँ प्रदान करता है।

>[!WARNING]
> [`HfFileSystem`] fsspec संगतता प्रदान करता है, जो उन लाइब्रेरीज़ के लिए उपयोगी है जिन्हें इसकी आवश्यकता होती है (उदाहरण के लिए, `pandas` के माध्यम से सीधे Hugging Face डेटासेट और बकेट्स पढ़ना)। हालांकि, यह संगतता लेयर अतिरिक्त ओवरहेड उत्पन्न करती है। बेहतर प्रदर्शन और विश्वसनीयता के लिए, जहाँ संभव हो, [`HfApi`] की मेथड्स का उपयोग करने की अनुशंसा की जाती है।


## उपयोग

```python
>>> from huggingface_hub import hffs

>>> # किसी डेटासेट डायरेक्टरी में सभी फ़ाइलों की सूची बनाएँ 
>>> hffs.ls("datasets/my-username/my-dataset-repo/data", detail=False)
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # किसी बकेट डायरेक्टरी में सभी फ़ाइलों की सूची बनाएँ 
>>> hffs.ls("buckets/my-username/my-bucket/experiment-data", detail=False)
['bucket/my-username/my-bucket/data/train-0000.parquet', 'bucket/my-username/my-bucket/data/train-0001.parquet', ...]

>>> # डेटासेट रिपॉज़िटरी में सभी ".csv" फ़ाइलें सूची बनाएँ
>>> hffs.glob("datasets/my-username/my-dataset-repo/**/*.csv")
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # किसी रिमोट फ़ाइल को पढ़ें 
>>> with hffs.open("datasets/my-username/my-dataset-repo/data/train.csv", "r") as f:
...     train_data = f.readlines()

>>> # किसी रिमोट फ़ाइल की सामग्री को स्ट्रिंग के रूप में पढ़ें
>>> train_data = hffs.read_text("datasets/my-username/my-dataset-repo/data/train.csv", revision="dev")

>>> # किसी रिमोट फ़ाइल में लिखें
>>> with hffs.open("datasets/my-username/my-dataset-repo/data/validation.csv", "w") as f:
...     f.write("text,label")
...     f.write("Fantastic movie!,good")
```

वैकल्पिक `revision` आर्ग्युमेंट को किसी विशेष कमिट (जैसे किसी ब्रांच, टैग नाम या कमिट हैश) से ऑपरेशन चलाने के लिए पास किया जा सकता है। ध्यान दें कि `revision` का उपयोग Buckets के साथ नहीं किया जा सकता।

Python के बिल्ट-इन `open` के विपरीत, `fsspec` का `open` डिफ़ॉल्ट रूप से बाइनरी मोड (`"rb"`) में खुलता है। इसका अर्थ है कि टेक्स्ट मोड में पढ़ने के लिए आपको स्पष्ट रूप से `"r"` और लिखने के लिए `"w"` मोड सेट करना होगा। किसी फ़ाइल में सामग्री जोड़ना (मोड `"a"` और `"ab"`) अभी समर्थित नहीं है।

## एकीकरण

[`HfFileSystem`] का उपयोग किसी भी ऐसी लाइब्रेरी के साथ किया जा सकता है जो `fsspec` को एकीकृत करती हो, बशर्ते URL निम्नलिखित प्रारूप का पालन करता हो:

```
hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/hf_urls_with_buckets.png"/>
</div>

`repo_type_prefix`, डेटासेट के लिए `datasets/`, Spaces के लिए `spaces/` होता है, जबकि मॉडल्स के लिए URL में किसी प्रीफ़िक्स की आवश्यकता नहीं होती।

रिपॉज़िटरीज़ के अलावा, [`HfFileSystem`] Hugging Face Buckets का भी समर्थन करता है, जो S3-जैसी ऑब्जेक्ट स्टोरेज सेवा है (अधिक जानकारी के लिए [इस गाइड](./buckets) को देखें):

```
hf://buckets/<bucket_id>/<path/in/bucket>
```

नीचे कुछ ऐसे रोचक एकीकरण दिए गए हैं, जहाँ [`HfFileSystem`] Hub के साथ इंटरैक्ट करना आसान बनाता है:

* किसी Hub रिपॉज़िटरी से [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files) DataFrame को पढ़ना/लिखना:


  ```python
  >>> import pandas as pd

  >>> # किसी रिमोट CSV फ़ाइल को DataFrame में पढ़ें
  >>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv")
  >>> df = pd.read_csv("hf://buckets/my-username/my-bucket/train.csv")

  >>> # किसी DataFrame को रिमोट CSV फ़ाइल में लिखें
  >>> df.to_csv("hf://datasets/my-username/my-dataset-repo/test.csv")
  >>> df.to_csv("hf://buckets/my-username/my-bucket/test.csv")
  ```>>> # किसी रिमोट फ़ाइल में लिखें

उसी वर्कफ़्लो का उपयोग [Dask](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html) और [Polars](https://pola-rs.github.io/polars/py-polars/html/reference/io.html) DataFrames के लिए भी किया जा सकता है।

* [DuckDB](https://duckdb.org/docs/guides/python/filesystems) का उपयोग करके (रिमोट) Hub फ़ाइलों पर क्वेरी करना:

  ```python
  >>> from huggingface_hub import HfFileSystem
  >>> import duckdb

  >>> fs = HfFileSystem()
  >>> duckdb.register_filesystem(fs)
  >>> # किसी रिमोट फ़ाइल पर क्वेरी चलाएँ और परिणाम को DataFrame के रूप में प्राप्त करें
  >>> fs_query_file = "hf://datasets/my-username/my-dataset-repo/data_dir/data.parquet"
  >>> df = duckdb.query(f"SELECT * FROM '{fs_query_file}' LIMIT 10").df()
  ```

* [Zarr](https://zarr.readthedocs.io/en/stable/tutorial.html#io-with-fsspec) के साथ Hub को एक Array Store के रूप में उपयोग करना:

  ```python
  >>> import numpy as np
  >>> import zarr

  >>> embeddings = np.random.randn(50000, 1000).astype("float32")

  >>> # किसी रिपॉज़िटरी में एक array लिखें
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w") as root:
  ...    foo = root.create_group("embeddings")
  ...    foobar = foo.zeros('experiment_0', shape=(50000, 1000), chunks=(10000, 1000), dtype='f4')
  ...    foobar[:] = embeddings

  >>> # किसी रिपॉज़िटरी से एक array पढ़ें
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r") as root:
  ...    first_row = root["embeddings/experiment_0"][0]
  ```
  
## प्रमाणीकरण

कई मामलों में, Hub के साथ इंटरैक्ट करने के लिए आपको अपने Hugging Face अकाउंट में लॉग इन होना आवश्यक है। Hub पर उपलब्ध प्रमाणीकरण विधियों के बारे में अधिक जानने के लिए दस्तावेज़ के [Authentication](../quick-start#authentication) अनुभाग को देखें।

अपने `token` को [`HfFileSystem`] के एक आर्ग्युमेंट के रूप में पास करके प्रोग्रामेटिक रूप से लॉग इन करना भी संभव है:

```python
>>> from huggingface_hub import HfFileSystem
>>> hffs = HfFileSystem(token=token)
```

यदि आप इस तरीके से लॉग इन करते हैं, तो अपना सोर्स कोड साझा करते समय सावधान रहें कि आपका `token` गलती से सार्वजनिक न हो जाए।