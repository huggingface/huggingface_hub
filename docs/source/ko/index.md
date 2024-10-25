<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 🤗 Hub 클라이언트 라이브러리 [[hub-client-library]]

`huggingface_hub` 라이브러리는 [Hugging Face Hub](https://hf.co)와 상호작용할 수 있게 해줍니다. Hugging Face Hub는 창작자와 협업자를 위한 머신러닝 플랫폼입니다. 여러분의 프로젝트에 적합한 사전 훈련된 모델과 데이터셋을 발견하거나, Hub에 호스팅된 수백 개의 머신러닝 앱들을 사용해보세요. 또한, 여러분이 만든 모델과 데이터셋을 커뮤니티와 공유할 수도 있습니다. `huggingface_hub` 라이브러리는 파이썬으로 이 모든 것을 간단하게 할 수 있는 방법을 제공합니다.

`huggingface_hub` 라이브러리를 사용하기 위한 [빠른 시작 가이드](quick-start)를 읽어보세요. Hub에서 파일을 다운로드하거나, 레포지토리를 생성하거나, 파일을 업로드하는 방법을 배울 수 있습니다. 계속 읽어보면, 🤗 Hub에서 여러분의 레포지토리를 어떻게 관리하고, 토론에 어떻게 참여하고, 추론 API에 어떻게 접근하는지 알아볼 수 있습니다.


<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guides/overview">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">How-to 가이드</div>
      <p class="text-gray-700">특정 목표를 달성하는 데 도움이 되는 실용적인 가이드입니다. huggingface_hub로 실제 문제를 해결하는 방법을 배우려면 이 가이드들을 살펴보세요.</p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./package_reference/overview">
      <div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">라이브러리 레퍼런스</div>
      <p class="text-gray-700">huggingface_hub의 클래스와 메소드에 대한 완전하고 기술적인 설명입니다.</p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./concepts/git_vs_http">
      <div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">개념 가이드</div>
      <p class="text-gray-700">huggingface_hub의 철학을 더 잘 이해하기 위한 고수준의 설명입니다.</p>
    </a>

  </div>
</div>

<!--
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/overview"
  ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Tutorials</div>
  <p class="text-gray-700">Learn the basics and become familiar with using huggingface_hub to programmatically interact with the 🤗 Hub!</p>
</a> -->

## 기여하기 [[contribute]]

`huggingface_hub`에 대한 모든 기여를 환영하며, 소중히 생각합니다! 🤗 코드에서 기존의 이슈를 추가하거나 수정하는 것 외에도, 문서를 정확하고 최신으로 유지하도록 개선하거나, 이슈에 대한 질문에 답하거나, 라이브러리를 개선할 수 있다고 생각하는 새로운 기능을 요청하는 것도 커뮤니티에 도움이 됩니다. 새로운 이슈나 기능 요청을 제출하는 방법, PR을 제출하는 방법, 기여한 내용을 테스트하여 모든 것이 예상대로 작동하는지 확인하는 방법 등에 대해 더 알아보려면 [기여
가이드](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md)를 살펴보세요.

기여자들은 또한 모든 사람들을 위해 포괄적이고 환영받는 협업 공간을 만들기 위해 우리의 [행동
강령](https://github.com/huggingface/huggingface_hub/blob/main/CODE_OF_CONDUCT.md)을 준수해야 합니다.
