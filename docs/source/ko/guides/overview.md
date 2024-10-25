<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# How-to 가이드 [[howto-guides]]

특정 목표를 달성하는 데 도움이 되는 실용적인 가이드들입니다. huggingface_hub로 실제 문제를 해결하는 방법을 배우려면 다음 문서들을 살펴보세요.

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-y-4 md:gap-x-5">

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./repository">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        리포지토리
      </div><p class="text-gray-700">
        Hub에서 리포지토리를 만드는 방법은 무엇인가요? 구성하는 방법은요? 리포지토리와 상호 작용하려면 어떻게 해야하나요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./download">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        파일 다운로드
      </div><p class="text-gray-700">
        Hub에서 파일을 다운로드하려면 어떻게 하나요? 리포지토리는요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./upload">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        파일 업로드
      </div><p class="text-gray-700">
        파일이나 폴더를 어떻게 업로드하나요? Hub의 기존 리포지토리를 변경하려면 어떻게 해야 하나요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./search">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        검색
      </div><p class="text-gray-700">
        20만 개가 넘게 공개된 모델, 데이터 세트 및 Space를 효율적으로 검색하는 방법은 무엇인가요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./hf_file_system">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        HfFileSystem
      </div><p class="text-gray-700">
        Python의 파일 인터페이스를 모방한 편리한 인터페이스를 통해 Hub와 상호 작용하는 방법은 무엇인가요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./inference">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Inference
      </div><p class="text-gray-700">
        가속화된 Inference API로 추론하려면 어떻게 하나요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./community">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        커뮤니티 탭
      </div><p class="text-gray-700">
        커뮤니티 탭에서 PR과 댓글을 통해 어떻게 소통할 수 있나요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./manage-cache">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        캐시
      </div><p class="text-gray-700">
        캐시 시스템은 어떻게 작동하나요? 이점은 무엇인가요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./model-cards">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        모델 카드
      </div><p class="text-gray-700">
        모델 카드는 어떻게 만들고 공유하나요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./manage-spaces">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Space 관리
      </div><p class="text-gray-700">
        Space 하드웨어와 구성은 어떻게 관리하나요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./integrations">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        라이브러리 통합
      </div><p class="text-gray-700">
        라이브러리를 Hub와 통합한다는 것은 무엇을 의미하나요? 그리고 어떻게 할 수 있을까요?
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./webhooks_server">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        웹훅 서버
      </div><p class="text-gray-700">
        웹훅을 수신할 서버를 만들고 Space로 배포하는 방법은 무엇인가요?
      </p>
    </a>

  </div>
</div>
