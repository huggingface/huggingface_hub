<!--⚠️ 请注意，此文件为 Markdown 格式，但包含我们文档生成器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确渲染。
-->

# 操作指南

在本节中，您将找到帮助您实现特定目标的实用指南。
查看这些指南，了解如何使用 huggingface_hub 解决实际问题：

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-y-4 md:gap-x-5">

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./repository">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        仓库
      </div><p class="text-gray-700">
        如何在 Hub 上创建仓库？如何配置它？如何与之交互？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./download">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        下载文件
      </div><p class="text-gray-700">
        如何从 Hub 下载文件？如何下载仓库？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./upload">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        上传文件
      </div><p class="text-gray-700">
        如何上传文件或文件夹？如何对 Hub 上的现有仓库进行更改？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./search">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        搜索
      </div><p class="text-gray-700">
        如何高效地搜索超过 200k+ 个公共模型、数据集和Space？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./hf_file_system">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        HfFileSystem
      </div><p class="text-gray-700">
        如何通过一个模仿 Python 文件接口的便捷接口与 Hub 交互？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./inference">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        推理
      </div><p class="text-gray-700">
        如何使用加速推理 API 进行预测？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./community">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        社区
      </div><p class="text-gray-700">
        如何与社区（讨论和拉取请求）互动？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./collections">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        集合
      </div><p class="text-gray-700">
        如何以编程方式构建集合？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./manage-cache">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        缓存
      </div><p class="text-gray-700">
        缓存系统如何工作？如何从中受益？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./model-cards">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        模型卡片
      </div><p class="text-gray-700">
        如何创建和分享模型卡片？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./manage-spaces">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        管理您的Space
      </div><p class="text-gray-700">
        如何管理您的Space的硬件和配置？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./integrations">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        集成库
      </div><p class="text-gray-700">
        将库集成到 Hub 中意味着什么？如何实现？
      </p>
    </a>

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./webhooks_server">
      <div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">
        Webhooks 服务器
      </div><p class="text-gray-700">
        如何创建一个接收 Webhooks 的服务器并将其部署为一个Space？
      </p>
    </a>

  </div>
</div>
