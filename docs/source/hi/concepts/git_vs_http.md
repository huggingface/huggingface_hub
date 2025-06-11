# Git vs HTTP पैराडाइम

`huggingface_hub` लाइब्रेरी Hugging Face Hub के साथ आदान-प्रदान करने के लिए एक लाइब्रेरी है, जो git-आधारित repositories (models, datasets या Spaces) का एक संग्रह है। `huggingface_hub` का उपयोग करके Hub तक पहुंचने के दो मुख्य तरीके हैं।

पहला तरीका, जिसे "git-आधारित" तरीका कहा जाता है, [`Repository`] क्लास द्वारा संचालित है। यह विधि `git` कमांड के चारों ओर एक आवरण का उपयोग करती है जिसमें Hub के साथ आदान-प्रदान करने के लिए विशेष रूप से डिज़ाइन किए गए अतिरिक्त functions हैं। दूसरा विकल्प, जिसे "HTTP-आधारित" तरीका कहा जाता है, [`HfApi`] client का उपयोग करके HTTP requests बनाने में शामिल है। आइए प्रत्येक तरीका के फायदे और नुकसान की जांच करते हैं।

## Repository: ऐतिहासिक git-आधारित तरीका

शुरुआत में, `huggingface_hub` मुख्य रूप से [`Repository`] क्लास के चारों ओर बनाया गया था। यह सामान्य `git` कमांड जैसे `"git add"`, `"git commit"`, `"git push"`, `"git tag"`, `"git checkout"`, आदि के लिए Python wrappers प्रदान करता है।

लाइब्रेरी विवरण सेट करने और बड़ी फाइलों को track करने में भी मदद करती है, जो अक्सर machine learning repositories में उपयोग की जाती हैं। इसके अतिरिक्त, लाइब्रेरी आपको अपनी विधियों को पृष्ठभूमि में कार्यान्वित करने की अनुमति देती है, जो training के दौरान डेटा अपलोड करने के लिए उपयोगी है।

[`Repository`] का उपयोग करने का मुख्य फायदा यह है कि यह आपको अपनी मशीन पर संपूर्ण repository की एक local copy बनाए रखने की अनुमति देता है। यह एक नुकसान भी हो सकता है क्योंकि इसके लिए आपको इस local copy को लगातार update और maintain करना होता है। यह पारंपरिक software development के समान है जहां प्रत्येक developer अपनी स्वयं की local copy maintain करता है और feature पर काम करते समय changes push करता है। हालांकि, machine learning के संदर्भ में, यह हमेशा आवश्यक नहीं हो सकता क्योंकि users को केवल inference के लिए weights download करने या weights को एक format से दूसरे में convert करने की आवश्यकता हो सकती है, बिना पूरी repository को clone करने की आवश्यकता के।

<Tip warning={true}>

[`Repository`] अब http-आधारित विकल्पों के पक्ष में deprecated है। legacy code में इसकी बड़ी अपनाई जाने के कारण, [`Repository`] का पूर्ण removal केवल `v1.0` release में होगा।

</Tip>

## HfApi: एक लचीला और सुविधाजनक HTTP client

[`HfApi`] क्लास को local git repositories का एक विकल्प प्रदान करने के लिए विकसित किया गया था, जो maintain करना मुश्किल हो सकता है, विशेष रूप से बड़े models या datasets के साथ व्यवहार करते समय। [`HfApi`] क्लास git-आधारित तरीकाों की समान functionality प्रदान करती है, जैसे files download और push करना और branches तथा tags बनाना, लेकिन एक local folder की आवश्यकता के बिना जिसे sync में रखना पड़ता है।

`git` द्वारा पहले से प्रदान की गई functionalities के अलावा, [`HfApi`] क्लास अतिरिक्त features प्रदान करती है, जैसे repos manage करने की क्षमता, efficient reuse के लिए caching का उपयोग करके files download करना, repos और metadata के लिए Hub को search करना, discussions, PRs, और comments जैसी community features तक पहुंच, और Spaces hardware और secrets को configure करना।

## मुझे क्या उपयोग करना चाहिए? और कब?

कुल मिलाकर, **HTTP-आधारित तरीका सभी cases में** `huggingface_hub` का उपयोग करने का **अनुशंसित तरीका है**। [`HfApi`] changes को pull और push करने, PRs, tags और branches के साथ काम करने, discussions के साथ interact करने और बहुत कुछ करने की अनुमति देता है। `0.16` release के बाद से, http-आधारित methods भी पृष्ठभूमि में चल सकती हैं, जो [`Repository`] क्लास का अंतिम प्रमुख फायदा था।

हालांकि, सभी git commands [`HfApi`] के माध्यम से उपलब्ध नहीं हैं। कुछ को कभी भी implement नहीं किया जा सकता है, लेकिन हम हमेशा सुधार करने और gap को बंद करने की कोशिश कर रहे हैं। यदि आपको अपना use case covered नहीं दिखता है, तो कृपया [Github पर एक issue खोलें](https://github.com/huggingface/huggingface_hub)! हम अपने users के साथ और उनके लिए 🤗 ecosystem बनाने में मदद करने के लिए feedback का स्वागत करते हैं।

git-आधारित [`Repository`] पर http-आधारित [`HfApi`] की यह प्राथमिकता का मतलब यह नहीं है कि git versioning Hugging Face Hub से जल्द ही गायब हो जाएगी। workflows में जहां यह समझ में आता है, वहां `git` commands का locally उपयोग करना हमेशा संभव होगा।