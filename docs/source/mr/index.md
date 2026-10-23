<!--⚠️ लक्षात ठेवा की ही फाइल Markdown मध्ये आहे, परंतु यात आपल्या doc-builder साठीची विशेष syntax (MDX प्रमाणे) वापरली आहे. त्यामुळे ती तुमच्या Markdown viewer मध्ये योग्य प्रकारे render होईलच असे नाही.
-->

# 🤗 Hub क्लायंट लायब्ररी

`huggingface_hub` लायब्ररी तुम्हाला [Hugging Face Hub](https://hf.co) सोबत संवाद साधण्याची सुविधा देते. हे निर्माते आणि सहकार्य करणाऱ्यांसाठीचे एक मशीन लर्निंग प्लॅटफॉर्म आहे. तुमच्या प्रकल्पांसाठी आधीपासून प्रशिक्षित (pre-trained) मॉडेल्स आणि डेटासेट्स शोधा किंवा Hub वर होस्ट केलेल्या शेकडो मशीन लर्निंग अॅप्सचा वापर करून पाहा. तसेच, तुम्ही स्वतःचे मॉडेल्स आणि डेटासेट्स तयार करून समुदायासोबत शेअरही करू शकता. `huggingface_hub` लायब्ररी Python मधून हे सर्व सोप्या पद्धतीने करण्याची सुविधा देते.

`huggingface_hub` लायब्ररी वापरण्यास सुरुवात करण्यासाठी [Quick Start Guide](quick-start) वाचा. यात तुम्ही Hub वरून फाइल्स डाउनलोड करणे, रिपॉझिटरी तयार करणे आणि Hub वर फाइल्स अपलोड करणे शिकाल. पुढे वाचत राहा आणि 🤗 Hub वरील तुमच्या रिपॉझिटरीजचे व्यवस्थापन कसे करायचे, चर्चांमध्ये (Discussions) कसा सहभाग घ्यायचा आणि inference कसे चालवायचे हे जाणून घ्या.

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guides/overview">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">मार्गदर्शिका</div>
      <p class="text-gray-700">एखादे विशिष्ट उद्दिष्ट साध्य करण्यासाठी उपयुक्त अशा व्यावहारिक मार्गदर्शिका. वास्तविक समस्यांचे निराकरण करण्यासाठी `huggingface_hub` कसे वापरायचे ते जाणून घेण्यासाठी या मार्गदर्शिका पाहा.</p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./package_reference/overview">
      <div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">संदर्भ</div>
      <p class="text-gray-700">`huggingface_hub` मधील क्लासेस आणि मेथड्सचे सविस्तर तांत्रिक वर्णन.</p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./concepts/git_vs_http">
      <div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">संकल्पनात्मक मार्गदर्शिका</div>
      <p class="text-gray-700">`huggingface_hub` मागील संकल्पना आणि तत्त्वज्ञान अधिक चांगल्या प्रकारे समजून घेण्यासाठी उच्च-स्तरीय स्पष्टीकरणे.</p>
    </a>

  </div>
</div>

<!--
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/overview"
  ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">ट्युटोरियल्स</div>
  <p class="text-gray-700">मूलभूत गोष्टी शिका आणि 🤗 Hub सोबत प्रोग्रामद्वारे संवाद साधण्यासाठी `huggingface_hub` कसे वापरायचे ते जाणून घ्या!</p>
</a> -->

## योगदान द्या

`huggingface_hub` मध्ये केलेल्या सर्व योगदानांचे स्वागत आहे आणि प्रत्येक योगदानाला समान महत्त्व दिले जाते! 🤗 विद्यमान कोडमधील समस्या दुरुस्त करण्याव्यतिरिक्त, दस्तऐवजीकरण अचूक आणि अद्ययावत ठेवण्यास मदत करून, issues वरील प्रश्नांची उत्तरे देऊन आणि लायब्ररी अधिक उपयुक्त बनवतील अशी नवीन वैशिष्ट्ये (features) सुचवूनही तुम्ही योगदान देऊ शकता. नवीन issue किंवा feature request कशी सबमिट करायची, pull request कशी तयार करायची आणि तुमचे योगदान अपेक्षेप्रमाणे कार्य करते याची चाचणी कशी करायची हे जाणून घेण्यासाठी [Contribution Guide](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) पहा.

सर्वांसाठी समावेशक आणि स्वागतार्ह सहकार्याचे वातावरण निर्माण करण्यासाठी योगदानकर्त्यांनी आमच्या [Code of Conduct](https://github.com/huggingface/huggingface_hub/blob/main/CODE_OF_CONDUCT.md) चेही पालन करावे.