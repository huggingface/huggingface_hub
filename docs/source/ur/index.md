<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 🤗 Hub کلائنٹ لائبریری

`huggingface_hub` لائبریری آپ کو [Hugging Face Hub](https://hf.co) کے ساتھ کام کرنے دیتی ہے، جو creators اور collaborators کے لیے ایک machine learning platform ہے۔ اپنے projects کے لیے pre-trained models اور datasets دریافت کریں، یا Hub پر hosted hundreds of machine learning apps کے ساتھ تجربہ کریں۔ آپ اپنے models اور datasets بھی بنا کر community کے ساتھ share کر سکتے ہیں۔ `huggingface_hub` لائبریری Python کے ذریعے یہ سب کام کرنے کا ایک آسان طریقہ دیتی ہے۔

`huggingface_hub` لائبریری کے ساتھ جلدی شروع کرنے کے لیے [quick start guide](quick-start) پڑھیں۔ آپ سیکھیں گے کہ Hub سے files کیسے download کرنی ہیں، repository کیسے بنانی ہے، اور files کو Hub پر کیسے upload کرنا ہے۔ آگے پڑھتے رہیں تاکہ آپ جان سکیں کہ 🤗 Hub پر اپنی repositories کیسے manage کرنی ہیں، discussions میں کیسے interact کرنا ہے، یا inference کیسے چلانا ہے۔

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guides/overview">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">طریقہ کار guides</div>
      <p class="text-gray-700">عملی guides جو آپ کو ایک specific goal حاصل کرنے میں مدد دیتی ہیں۔ یہ guides دیکھیں تاکہ آپ سیکھ سکیں کہ real-world problems حل کرنے کے لیے huggingface_hub کیسے استعمال کرنا ہے۔</p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./package_reference/overview">
      <div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">حوالہ</div>
      <p class="text-gray-700">huggingface_hub classes اور methods کی مکمل اور technical description۔</p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./concepts/git_vs_http">
      <div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">تصوراتی guides</div>
      <p class="text-gray-700">huggingface_hub کی philosophy کو بہتر سمجھنے کے لیے high-level explanations۔</p>
    </a>

  </div>
</div>

<!--
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/overview"
  ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Tutorials</div>
  <p class="text-gray-700">Learn the basics and become familiar with using huggingface_hub to programmatically interact with the 🤗 Hub!</p>
</a> -->

## حصہ ڈالیں

`huggingface_hub` میں ہر contribution welcome ہے اور برابر value رکھتا ہے! 🤗 Code میں existing issues add یا fix کرنے کے علاوہ، آپ documentation کو accurate اور up-to-date رکھنے میں بھی مدد کر سکتے ہیں، issues پر questions کا جواب دے سکتے ہیں، اور ایسی new features request کر سکتے ہیں جو آپ کے خیال میں library کو بہتر بنائیں گی۔ نیا issue یا feature request submit کرنے، pull request بھیجنے، اور اپنی contributions test کرنے کا طریقہ جاننے کے لیے [contribution guide](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) دیکھیں۔

Contributors کو ہمارے [code of conduct](https://github.com/huggingface/huggingface_hub/blob/main/CODE_OF_CONDUCT.md) کا بھی احترام کرنا چاہیے تاکہ ہر ایک کے لیے ایک inclusive اور welcoming collaborative space بن سکے۔
