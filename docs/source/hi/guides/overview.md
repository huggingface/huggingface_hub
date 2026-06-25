<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# उपयोग मार्गदर्शिकाएँ

इस अनुभाग में आपको व्यावहारिक मार्गदर्शिकाएँ मिलेंगी, जो किसी विशेष लक्ष्य को प्राप्त करने में आपकी सहायता करेंगी।
इन मार्गदर्शिकाओं के माध्यम से आप सीखेंगे कि `huggingface_hub` का उपयोग करके वास्तविक समस्याओं का समाधान कैसे किया जाए:

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-y-4 md:gap-x-5">

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./repository">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Repository
      </div><p class="text-gray-700">
        Hub पर Repository कैसे बनाएं, उसे कॉन्फ़िगर कैसे करें और उसके साथ कैसे कार्य करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./download">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Download files
      </div><p class="text-gray-700">
        Hub से फ़ाइलें या पूरी Repository कैसे डाउनलोड करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./upload">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Upload files
      </div><p class="text-gray-700">
        फ़ाइल या फ़ोल्डर कैसे अपलोड करें और Hub पर मौजूद Repository में बदलाव कैसे करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./search">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Search
      </div><p class="text-gray-700">
        200k+ सार्वजनिक Models, Datasets और Spaces में प्रभावी ढंग से खोज कैसे करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./hf_file_system">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        HfFileSystem
      </div><p class="text-gray-700">
        Python के file interface के समान सुविधाजनक इंटरफ़ेस के माध्यम से Hub के साथ कैसे कार्य करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./inference">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Inference
      </div><p class="text-gray-700">
        Hugging Face Inference Providers का उपयोग करके अनुमान (predictions) कैसे प्राप्त करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./community">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Community Tab
      </div><p class="text-gray-700">
        Community Tab में Discussions और Pull Requests के साथ कैसे कार्य करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./collections">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Collections
      </div><p class="text-gray-700">
        प्रोग्रामेटिक तरीके से Collections कैसे बनाएं।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./manage-cache">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Cache
      </div><p class="text-gray-700">
        Cache प्रणाली कैसे काम करती है और इसका लाभ कैसे उठाया जाए।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./model-cards">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Model Cards
      </div><p class="text-gray-700">
        Model Cards कैसे बनाएं और उन्हें साझा कैसे करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./manage-spaces">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Manage your Space
      </div><p class="text-gray-700">
        अपने Space के hardware और configuration का प्रबंधन कैसे करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./integrations">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Integrate a library
      </div><p class="text-gray-700">
        किसी library को Hub के साथ एकीकृत करने का क्या अर्थ है और यह कैसे किया जाए।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./webhooks_server">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Webhooks server
      </div><p class="text-gray-700">
        Webhooks प्राप्त करने के लिए Server कैसे बनाएं और उसे Space के रूप में कैसे deploy करें।
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./jobs">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Jobs
      </div><p class="text-gray-700">
        Hugging Face infrastructure पर compute Jobs कैसे चलाएँ, उनका प्रबंधन कैसे करें और उपयुक्त hardware कैसे चुनें।
      </p>
    </a>

  </div>
</div>
