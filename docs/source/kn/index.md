<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 🤗 ಹಬ್ ಕ್ಲೈಂಟ್ ಲೈಬ್ರರಿ

`huggingface_hub` ಲೈಬ್ರರಿಯು [Hugging Face Hub](https://hf.co) ಜೊತೆ ಸಂವಹನ ನಡೆಸಲು ನಿಮಗೆ ಅವಕಾಶ ನೀಡುತ್ತದೆ. ಇದು
ಸೃಷ್ಟಿಕರ್ತರಿಗೆ ಮತ್ತು ಸಹಯೋಗಿಗಳಿಗೆ ಇರುವ ಒಂದು ಮಷಿನ್ ಲರ್ನಿಂಗ್ ವೇದಿಕೆ. ನಿಮ್ಮ ಪ್ರಾಜೆಕ್ಟ್‌ಗಳಿಗೆ ಬೇಕಾದ
ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದ ಮಾಡೆಲ್‌ಗಳನ್ನು ಮತ್ತು ಡೇಟಾಸೆಟ್‌ಗಳನ್ನು ಇಲ್ಲಿ ಹುಡುಕಬಹುದು, ಅಥವಾ ಹಬ್‌ನಲ್ಲಿ ಹೋಸ್ಟ್ ಆಗಿರುವ
ನೂರಾರು ಮಷಿನ್ ಲರ್ನಿಂಗ್ ಆ್ಯಪ್‌ಗಳನ್ನು ಬಳಸಿ ನೋಡಬಹುದು. ನಿಮ್ಮದೇ ಮಾಡೆಲ್‌ಗಳನ್ನು ಮತ್ತು ಡೇಟಾಸೆಟ್‌ಗಳನ್ನು ರಚಿಸಿ
ಸಮುದಾಯದೊಂದಿಗೆ ಹಂಚಿಕೊಳ್ಳಲೂಬಹುದು. ಇವೆಲ್ಲವನ್ನೂ Python ಮೂಲಕ ಸುಲಭವಾಗಿ ಮಾಡುವ ದಾರಿಯನ್ನು
`huggingface_hub` ಲೈಬ್ರರಿ ಒದಗಿಸುತ್ತದೆ.

`huggingface_hub` ಲೈಬ್ರರಿಯೊಂದಿಗೆ ಬೇಗನೆ ಕೆಲಸ ಆರಂಭಿಸಲು [ಕ್ವಿಕ್‌ಸ್ಟಾರ್ಟ್ ಮಾರ್ಗದರ್ಶಿ](quick-start) ಓದಿ.
ಹಬ್‌ನಿಂದ ಫೈಲ್‌ಗಳನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡುವುದು, ರಿಪೊಸಿಟರಿ ರಚಿಸುವುದು ಮತ್ತು ಹಬ್‌ಗೆ ಫೈಲ್‌ಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡುವುದು
ಹೇಗೆ ಎಂಬುದನ್ನು ಅಲ್ಲಿ ಕಲಿಯುವಿರಿ. 🤗 ಹಬ್‌ನಲ್ಲಿ ನಿಮ್ಮ ರಿಪೊಸಿಟರಿಗಳನ್ನು ನಿರ್ವಹಿಸುವುದು, ಚರ್ಚೆಗಳಲ್ಲಿ
ಭಾಗವಹಿಸುವುದು, ಅಥವಾ ಇನ್ಫರೆನ್ಸ್ ಚಲಾಯಿಸುವುದು ಹೇಗೆ ಎಂಬುದರ ಬಗ್ಗೆ ಇನ್ನಷ್ಟು ತಿಳಿಯಲು ಮುಂದೆ ಓದುತ್ತಾ ಹೋಗಿ.

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guides/overview">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">ಹೇಗೆ-ಮಾಡುವುದು ಮಾರ್ಗದರ್ಶಿಗಳು</div>
      <p class="text-gray-700">ನಿರ್ದಿಷ್ಟ ಗುರಿಯನ್ನು ಸಾಧಿಸಲು ನೆರವಾಗುವ ಪ್ರಾಯೋಗಿಕ ಮಾರ್ಗದರ್ಶಿಗಳು. ನಿಜ ಜೀವನದ ಸಮಸ್ಯೆಗಳನ್ನು ಬಗೆಹರಿಸಲು huggingface_hub ಅನ್ನು ಹೇಗೆ ಬಳಸುವುದು ಎಂದು ಈ ಮಾರ್ಗದರ್ಶಿಗಳಲ್ಲಿ ನೋಡಿ.</p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./package_reference/overview">
      <div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">ರೆಫರೆನ್ಸ್</div>
      <p class="text-gray-700">huggingface_hub ನ ಕ್ಲಾಸ್‌ಗಳು ಮತ್ತು ಮೆಥಡ್‌ಗಳ ಸಂಪೂರ್ಣ ಹಾಗೂ ತಾಂತ್ರಿಕ ವಿವರಣೆ.</p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./concepts/git_vs_http">
      <div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">ಪರಿಕಲ್ಪನಾ ಮಾರ್ಗದರ್ಶಿಗಳು</div>
      <p class="text-gray-700">huggingface_hub ನ ಮೂಲ ತತ್ವಗಳನ್ನು ಚೆನ್ನಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ನೆರವಾಗುವ ಉನ್ನತ ಮಟ್ಟದ ವಿವರಣೆಗಳು.</p>
    </a>

  </div>
</div>

## ಕೊಡುಗೆ ನೀಡಿ

`huggingface_hub` ಗೆ ಬರುವ ಎಲ್ಲ ಕೊಡುಗೆಗಳಿಗೂ ಸ್ವಾಗತ, ಮತ್ತು ಎಲ್ಲವಕ್ಕೂ ಸಮಾನ ಮಹತ್ವ! 🤗 ಕೋಡ್‌ನಲ್ಲಿ ಹೊಸದನ್ನು
ಸೇರಿಸುವುದು ಅಥವಾ ಈಗಿರುವ ಸಮಸ್ಯೆಗಳನ್ನು ಸರಿಪಡಿಸುವುದರ ಜೊತೆಗೆ, ಡಾಕ್ಯುಮೆಂಟೇಶನ್ ನಿಖರವಾಗಿಯೂ ಇತ್ತೀಚಿನದಾಗಿಯೂ
ಇದೆಯೇ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳುವ ಮೂಲಕ, ಇಶ್ಯೂಗಳಲ್ಲಿನ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸುವ ಮೂಲಕ, ಮತ್ತು ಲೈಬ್ರರಿಯನ್ನು
ಉತ್ತಮಗೊಳಿಸುತ್ತವೆ ಎಂದು ನಿಮಗೆ ಅನಿಸುವ ಹೊಸ ಫೀಚರ್‌ಗಳನ್ನು ಕೇಳುವ ಮೂಲಕವೂ ನೀವು ನೆರವಾಗಬಹುದು. ಹೊಸ ಇಶ್ಯೂ ಅಥವಾ
ಫೀಚರ್ ಮನವಿಯನ್ನು ಸಲ್ಲಿಸುವುದು ಹೇಗೆ, ಪುಲ್ ರಿಕ್ವೆಸ್ಟ್ ಸಲ್ಲಿಸುವುದು ಹೇಗೆ, ಮತ್ತು ಎಲ್ಲವೂ ನಿರೀಕ್ಷಿಸಿದಂತೆ ಕೆಲಸ
ಮಾಡುತ್ತಿದೆ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು ನಿಮ್ಮ ಕೊಡುಗೆಗಳನ್ನು ಪರೀಕ್ಷಿಸುವುದು ಹೇಗೆ ಎಂಬುದನ್ನು ತಿಳಿಯಲು
[ಕೊಡುಗೆ ಮಾರ್ಗದರ್ಶಿ](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md)
ನೋಡಿ.

ಎಲ್ಲರಿಗೂ ಸೇರಿದ, ಸ್ವಾಗತಾರ್ಹ ಸಹಯೋಗದ ವಾತಾವರಣವನ್ನು ನಿರ್ಮಿಸಲು ಕೊಡುಗೆದಾರರು ನಮ್ಮ
[ನೀತಿ ಸಂಹಿತೆಯನ್ನು](https://github.com/huggingface/huggingface_hub/blob/main/CODE_OF_CONDUCT.md)
ಗೌರವಿಸಬೇಕು.
