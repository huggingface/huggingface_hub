<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 🤗 ஹப் கிளையன்ட் லைப்ரரி

`Huggingface_hub` லைப்ரரி உங்களை [ஹக்கிங் ஃபேஸ் ஹப்](https://hf.co)  உடன் தொடர்புகொள்ள அனுமதிக்கிறது, இது படைப்பாளர்கள் மற்றும் கூட்டுப்பணியாளர்களுக்கான இயந்திர கற்றல் தளமாகும். உங்கள் திட்டங்களுக்கான முன் பயிற்சி பெற்ற மாதிரிகள் மற்றும் தரவுத்தொகுப்புகளைக் கண்டறியவும் அல்லது ஹப்பில் ஹோஸ்ட் செய்யப்பட்ட நூற்றுக்கணக்கான இயந்திர கற்றல் பயன்பாடுகளுடன் விளையாடவும். உங்கள் சொந்த மாதிரிகள் மற்றும் தரவுத்தொகுப்புகளை உருவாக்கி சமூகத்துடன் பகிரலாம். `huggingface_hub` லைப்ரரி பைதான் மூலம் இவற்றைச் செய்வதற்கான எளிய வழியை வழங்குகிறது.


[இந்த துரிதத் தொடக்கக் கையேட்டை](quick-start) வாசித்தால், `huggingface_hub` நூலகத்துடன் வேலை செய்ய எவ்வாறு ஆரம்பிக்கலாம் என்பதை நீங்கள் கற்றுக்கொள்வீர்கள். இதில், 🤗 ஹப் (Hub) இலிருந்து கோப்புகளை எவ்வாறு பதிவிறக்குவது, ஒரு `repository` உருவாக்குவது மற்றும் கோப்புகளை ஹபுக்கு எவ்வாறு பதிவேற்றுவது என்பதை நீங்கள் கற்றுக்கொள்வீர்கள்.மேலும், 🤗 ஹபில் உங்கள் repositoryகளை எவ்வாறு நிர்வகிக்க வேண்டும், விவாதங்களில் எவ்வாறு ஈடுபட வேண்டும், அல்லது `Inference API`யை எப்படி அணுகுவது என்பதையும் கற்றுக்கொள்ள இந்த வழிகாட்டியை தொடர்ந்து வாசியுங்கள்.


<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">

<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guides/overview">
  <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">எப்படி செய்ய வேண்டும் கையேடுகள்</div>
  <p class="text-gray-700">ஒரு குறிப்பிட்ட இலக்கை அடைய உதவுவதற்கான நடைமுறை கையேடுகள். உண்மையான உலக பிரச்சினைகளைத் தீர்க்க hஹக்கிங் ஃபேஸ் ஹப் ஐ எவ்வாறு பயன்படுத்துவது கற்றுக்கொள்ள இந்த கையேடுகளைப் பார்க்கவும்.</p>
</a>

<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./package_reference/overview">
  <div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">குறிப்பு</div>
  <p class="text-gray-700">ஹக்கிங் ஃபேஸ் ஹப் வகுப்புகள் மற்றும் முறைகள் பற்றிய முழுமையான மற்றும் தொழில்நுட்ப விவரணம்.</p>
</a>

<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./concepts/git_vs_http">
  <div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">விளக்கக் கையேடுகள்</div>
  <p class="text-gray-700">ஹக்கிங் ஃபேஸ் ஹப் தத்துவத்தை மேலும் புரிந்துகொள்ள உயர்நிலையான விளக்கங்கள்.</p>
</a>


  </div>
</div>

## பங்களிப்பு

`huggingface_hub`-க்கு அனைத்து பங்களிப்புகளும் வரவேற்கப்படுகின்றன மற்றும் சமமாக மதிக்கப்படுகின்றன! 🤗 கோடில் உள்ள உள்ளமைவுகளையும் அல்லது பிழைகளைச் சரிசெய்வதோடு, ஆவணங்களை சரியாகவும், தற்போதைய நிலையில் இருப்பதையும் உறுதிப்படுத்துவதன் மூலம் தங்களால் உதவலாம், மேலும் இஷ்யூக்களுக்கான கேள்விகளுக்கு பதிலளிக்கவும், நூலகத்தை மேம்படுத்துமாறு நீங்கள் நினைப்பதைத் தொடர்ந்து புதிய அம்சங்களை கோரலாம். பங்களிப்பு குறித்த [வழிகாட்டலை](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) பார்க்கவும், புதிய இஷ்யூவோ அல்லது அம்சக் கோரிக்கையோ எப்படி சமர்ப்பிக்க வேண்டும், புல் ரிக்வெஸ்ட்களை (Pull Request) சமர்ப்பிப்பது எப்படி, மேலும் உங்கள் பங்களிப்புகள் அனைத்தும் எதிர்பார்த்தது போல வேலை செய்கிறதா என்பதைச் சோதிப்பது எப்படி என்பதையும் கற்றுக்கொள்ளலாம்.

பங்களிப்பாளர்கள் அனைவருக்கும் உள்ளடக்கிய மற்றும் வரவேற்கக்கூடிய ஒத்துழைப்பு நிலையை உருவாக்க, நாங்கள் உருவாக்கிய [நடத்தை விதிகளை](https://github.com/huggingface/huggingface_hub/blob/main/CODE_OF_CONDUCT.md) மதிக்க வேண்டும்.






