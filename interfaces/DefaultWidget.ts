import { PipelineType } from './Types';

type LanguageCode = string;

type PerLanguageMapping = Map<
	keyof typeof PipelineType,
	(Record<string, any> | string)[]
>;


/// NOTE TO CONTRIBUTORS:
///
/// When adding sample inputs for a new language, you don't
/// necessarily have to translate the inputs from existing languages.
/// (which were quite random to begin with)
///
/// i.e. Feel free to be creative and provide better samples.
//

/// The <mask> placeholder will be replaced by the correct mask token
/// in the following examples, depending on the model type
///
/// see [INTERNAL] github.com/huggingface/moon-landing/blob/c5c3d45fe0ab27347b3ab27bdad646ef20732351/server/lib/App.ts#L254
//

const MAPPING_EN: PerLanguageMapping = new Map([
	[ "text-classification", [
		`I like you. I love you`,
	] ],
	[ "token-classification", [
		`My name is Wolfgang and I live in Berlin`,
		`My name is Sarah and I live in London`,
		`My name is Clara and I live in Berkeley, California.`,
	] ],
	[ "question-answering", [
		{
			text: `Where do I live?`,
			context: `My name is Wolfgang and I live in Berlin`,
		},
		{
			text: `Where do I live?`,
			context: `My name is Sarah and I live in London`,
		},
		{
			text: `What's my name?`,
			context: `My name is Clara and I live in Berkeley.`,
		},
		{
			text: `Which name is also used to describe the Amazon rainforest in English?`,
			context: `The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.`,
		}
	] ],
	[ "table-question-answering", [
		{
			text: `How many stars does the transformers repository have?`,
			table: {
				Repository: [
					"Transformers",
					"Datasets",
					"Tokenizers",
				],
				Stars: [
					36542,
					4512,
					3934,
				],
				Contributors: [
					651,
					77,
					34,
				],
				"Programming language": [
					"Python",
					"Python",
					"Rust, Python and NodeJS"
				]
			},
		},
	] ],
	[ "translation", [
		`My name is Wolfgang and I live in Berlin`,
		`My name is Sarah and I live in London`,
	] ],
	[ "summarization", [
		`The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.`,
	] ],
	[ "text-generation", [
		`My name is Julien and I like to`,
		`My name is Thomas and my main`,
		`My name is Mariama, my favorite`,
		`My name is Clara and I am`,
		`Once upon a time,`,
	] ],
	[ "fill-mask", [
		`Paris is the <mask> of France.`,
		`The goal of life is <mask>.`,
	] ],
	[ "zero-shot-classification", [
		{
			text: "I have a problem with my iphone that needs to be resolved asap!!",
			candidate_labels: "urgent, not urgent, phone, tablet, computer",
			multi_class: true,
		},
		{
			text: "Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.",
			candidate_labels: "mobile, website, billing, account access",
			multi_class: false,
		},
		{
			text: "A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world. Konstantin Batygin did not set out to solve one of the solar system’s most puzzling mysteries when he went for a run up a hill in Nice, France. Dr. Batygin, a Caltech researcher, best known for his contributions to the search for the solar system’s missing “Planet Nine,” spotted a beer bottle. At a steep, 20 degree grade, he wondered why it wasn’t rolling down the hill. He realized there was a breeze at his back holding the bottle in place. Then he had a thought that would only pop into the mind of a theoretical astrophysicist: “Oh! This is how Europa formed.” Europa is one of Jupiter’s four large Galilean moons. And in a paper published Monday in the Astrophysical Journal, Dr. Batygin and a co-author, Alessandro Morbidelli, a planetary scientist at the Côte d’Azur Observatory in France, present a theory explaining how some moons form around gas giants like Jupiter and Saturn, suggesting that millimeter-sized grains of hail produced during the solar system’s formation became trapped around these massive worlds, taking shape one at a time into the potentially habitable moons we know today.",
			candidate_labels: "space & cosmos, scientific discovery, microbiology, robots, archeology",
			multi_class: true,
		}
	] ],
]);

const MAPPING_ZH: PerLanguageMapping = new Map([
	[ "text-classification", [
		`我喜欢你。 我爱你`,
	] ],
	[ "token-classification", [
		`我叫沃尔夫冈，我住在柏林。`,
		`我叫萨拉，我住在伦敦。`,
		`我叫克拉拉，我住在加州伯克利。`,
	] ],
	[ "question-answering", [
		{
			text: `我住在哪里？`,
			context: `我叫沃尔夫冈，我住在柏林。`,
		},
		{
			text: `我住在哪里？`,
			context: `我叫萨拉，我住在伦敦。`,
		},
		{
			text: `我的名字是什么？`,
			context: `我叫克拉拉，我住在伯克利。`,
		},
	] ],
	[ "translation", [
		`我叫沃尔夫冈，我住在柏林。`,
		`我叫萨拉，我住在伦敦。`,
	] ],
	[ "summarization", [
		`该塔高324米（1063英尺），与一幢81层的建筑物一样高，是巴黎最高的建筑物。 它的底座是方形的，每边长125米（410英尺）。 在建造过程中，艾菲尔铁塔超过了华盛顿纪念碑，成为世界上最高的人造结构，它保持了41年的头衔，直到1930年纽约市的克莱斯勒大楼竣工。这是第一个到达300米高度的结构。 由于1957年在塔顶增加了广播天线，因此它现在比克莱斯勒大厦高5.2米（17英尺）。 除发射器外，艾菲尔铁塔是法国第二高的独立式建筑，仅次于米劳高架桥。`,
	] ],
	[ "text-generation", [
		`我叫朱利安，我喜欢`,
		`我叫托马斯，我的主要`,
		`我叫玛丽亚，我最喜欢的`,
		`我叫克拉拉，我是`,
		`从前，`,
	] ],
	[ "fill-mask", [
		`巴黎是<mask>国的首都。`,
		`生活的真谛是<mask>。`
	] ],
]);

const MAPPING_FR: PerLanguageMapping = new Map([
	[ "text-classification", [
		`Je t'apprécie beaucoup. Je t'aime.`,
	] ],
	[ "token-classification", [
		`Mon nom est Wolfgang et je vis à Berlin`,
	] ],
	[ "question-answering", [
		{
			text: `Où est-ce que je vis?`,
			context: `Mon nom est Wolfgang et je vis à Berlin`,
		}
	] ],
	[ "translation", [
		`Mon nom est Wolfgang et je vis à Berlin`,
	] ],
	[ "summarization", [
		`La tour fait 324 mètres (1,063 pieds) de haut, environ la même hauteur qu'un immeuble de 81 étages, et est la plus haute structure de Paris. Sa base est carrée, mesurant 125 mètres (410 pieds) sur chaque côté. Durant sa construction, la tour Eiffel surpassa le Washington Monument pour devenir la plus haute structure construite par l'homme dans le monde, un titre qu'elle conserva pendant 41 ans jusqu'à l'achèvement du Chrysler Building à New-York City en 1930. Ce fut la première structure à atteindre une hauteur de 300 mètres. Avec l'ajout d'une antenne de radiodiffusion au sommet de la tour Eiffel en 1957, celle-ci redevint plus haute que le Chrysler Building de 5,2 mètres (17 pieds). En excluant les transmetteurs, elle est la seconde plus haute stucture autoportante de France après le viaduc de Millau.`,
	] ],
	[ "text-generation", [
		`Mon nom est Julien et j'aime`,
		`Mon nom est Thomas et mon principal`,
		`Il était une fois`,
	] ],
	[ "fill-mask", [
		`Paris est la <mask> de la France.`,
	] ],
]);

const MAPPING_ES: PerLanguageMapping = new Map([
	[ "text-classification", [
		`Te quiero. Te amo.`,
	] ],
	[ "token-classification", [
		`Me llamo Wolfgang y vivo en Berlin`,
	] ],
	[ "question-answering", [
		{
			text: `¿Dónde vivo?`,
			context: `Me llamo Wolfgang y vivo en Berlin`,
		},
		{
			text: `¿Quién inventó el submarino?`,
			context: `Isaac Peral fue un murciano que inventó el submarino`,
		},
		{
			text: `¿Cuántas personas hablan español?`,
			context: `El español es el segundo idioma más hablado del mundo con más de 442 millones de hablantes`,
		}
	] ],
	[ "translation", [
		`Me llamo Wolfgang y vivo en Berlin`,
		`Los ingredientes de una tortilla de patatas son: huevos, patatas y cebolla`,
	] ],
	[ "summarization", [
		`La torre tiene 324 metros (1.063 pies) de altura, aproximadamente la misma altura que un edificio de 81 pisos y la estructura más alta de París. Su base es cuadrada, mide 125 metros (410 pies) a cada lado. Durante su construcción, la Torre Eiffel superó al Washington Monument para convertirse en la estructura artificial más alta del mundo, un título que mantuvo durante 41 años hasta que el Chrysler Building en la ciudad de Nueva York se terminó en 1930. Fue la primera estructura en llegar Una altura de 300 metros. Debido a la adición de una antena de transmisión en la parte superior de la torre en 1957, ahora es más alta que el Chrysler Building en 5,2 metros (17 pies). Excluyendo los transmisores, la Torre Eiffel es la segunda estructura independiente más alta de Francia después del Viaducto de Millau.`,
	] ],
	[ "text-generation", [
		`Me llamo Julien y me gusta`,
		`Me llamo Thomas y mi principal`,
		`Me llamo Manuel y trabajo en`,
		`Érase una vez,`,
		`Si tú me dices ven, `
	] ],
	[ "fill-mask", [
		`Mi nombre es <mask> y vivo en Nueva York.`,
		`El español es un idioma muy <mask> en el mundo.`,
	] ],
]);

const MAPPING_RU: PerLanguageMapping = new Map([
	[ "text-classification", [
		`Ты мне нравишься. Я тебя люблю`,
	] ],
	[ "token-classification", [
		`Меня зовут Вольфганг и я живу в Берлине`,
	] ],
	[ "question-answering", [
		{
			text: `Где живу?`,
			context: `Меня зовут Вольфганг и я живу в Берлине`,
		}
	] ],
	[ "translation", [
		`Меня зовут Вольфганг и я живу в Берлине`,
	] ],
	[ "summarization", [
		`Высота башни составляет 324 метра (1063 фута), примерно такая же высота, как у 81-этажного здания, и самое высокое сооружение в Париже. Его основание квадратно, размером 125 метров (410 футов) с любой стороны. Во время строительства Эйфелева башня превзошла монумент Вашингтона, став самым высоким искусственным сооружением в мире, и этот титул она удерживала в течение 41 года до завершения строительство здания Крайслер в Нью-Йорке в 1930 году. Это первое сооружение которое достигло высоты 300 метров. Из-за добавления вещательной антенны на вершине башни в 1957 году она сейчас выше здания Крайслер на 5,2 метра (17 футов). За исключением передатчиков, Эйфелева башня является второй самой высокой отдельно стоящей структурой во Франции после виадука Мийо.`,
	] ],
	[ "text-generation", [
		`Меня зовут Жюльен и`,
		`Меня зовут Томас и мой основной`,
		`Однажды`,
	] ],
	[ "fill-mask", [
		`Меня зовут <mask> и я инженер живущий в Нью-Йорке.`,
	] ],
]);

const MAPPING_UK: PerLanguageMapping = new Map([
	[ "translation", [
		`Мене звати Вольфґанґ і я живу в Берліні.`,
	] ],
	[ "fill-mask", [
		`Мене звати <mask>.`,
	] ],
]);

const MAPPING_IT: PerLanguageMapping = new Map([
	[ "text-classification", [
		`Mi piaci. Ti amo`,
	] ],
	[ "token-classification", [
		`Mi chiamo Wolfgang e vivo a Berlino`,
		`Mi chiamo Sarah e vivo a Londra`,
		`Mi chiamo Clara e vivo a Berkeley in California.`,
	] ],
	[ "question-answering", [
		{
			text: `Dove vivo?`,
			context: `Mi chiamo Wolfgang e vivo a Berlino`,
		},
		{
			text: `Dove vivo?`,
			context: `Mi chiamo Sarah e vivo a Londra`,
		},
		{
			text: `Come mio chiamo?`,
			context: `Mi chiamo Clara e vivo a Berkeley.`,
		},
	] ],
	[ "translation", [
		`Mi chiamo Wolfgang e vivo a Berlino`,
		`Mi chiamo Sarah e vivo a Londra`,
	] ],
	[ "summarization", [
		`La torre degli Asinelli è una delle cosiddette due torri di Bologna, simbolo della città, situate in piazza di porta Ravegnana, all'incrocio tra le antiche strade San Donato (ora via Zamboni), San Vitale, Maggiore e Castiglione. Eretta, secondo la tradizione, fra il 1109 e il 1119 dal nobile Gherardo Asinelli, la torre è alta 97,20 metri, pende verso ovest per 2,23 metri e presenta all'interno una scalinata composta da 498 gradini. Ancora non si può dire con certezza quando e da chi fu costruita la torre degli Asinelli. Si presume che la torre debba il proprio nome a Gherardo Asinelli, il nobile cavaliere di fazione ghibellina al quale se ne attribuisce la costruzione, iniziata secondo una consolidata tradizione l'11 ottobre 1109 e terminata dieci anni dopo, nel 1119.`,
	] ],
	[ "text-generation", [
		`Mi chiamo Loreto e mi piace`,
		`Mi chiamo Thomas e il mio principale`,
		`Mi chiamo Marianna, la mia cosa preferita`,
		`Mi chiamo Clara e sono`,
		`C'era una volta`,
	] ],
	[ "fill-mask", [
		`Roma è la <mask> d'Italia.`,
		`Lo scopo della vita è <mask>.`,
	] ],
]);

const MAPPING_FA: PerLanguageMapping = new Map([
	[ "text-classification", [
		`به موقع تحویل شد و همه چیز خوب بود.`,
		`سیب زمینی بی کیفیت بود.`,
		`قیمت و کیفیت عالی`,
		`خوب نبود اصلا`,
	] ],
	[ "token-classification", [
		`این سریال به صورت رسمی در تاریخ دهم می ۲۰۱۱ توسط شبکه فاکس برای پخش رزرو شد.`,
		`دفتر مرکزی شرکت کامیکو در شهر ساسکاتون ساسکاچوان قرار دارد.`,
		`در سال ۲۰۱۳ درگذشت و آندرتیکر و کین برای او مراسم یادبود گرفتند.`,
	] ],
	[ "question-answering", [] ],
	[ "translation", [] ],
	[ "summarization", [] ],
	[ "text-generation", [] ],
	[ "fill-mask", [
		`زندگی یک سوال است و این که چگونه <mask> کنیم پاسخ این سوال!`,
		`زندگی از مرگ پرسید: چرا همه من را <mask> دارند اما از تو متنفرند؟`,
	] ],
]);

const MAPPING_AR: PerLanguageMapping = new Map([
	[ "text-classification", [
		`أحبك. أهواك`,
	] ],
	[ "token-classification", [
		`إسمي محمد وأسكن في برلين`,
		`إسمي ساره وأسكن في لندن`,
		`إسمي سامي وأسكن في القدس في فلسطين.`,
	] ],
	[ "question-answering", [
		{
			text: `أين أسكن؟`,
			context: `إسمي محمد وأسكن في بيروت`,
		},
		{
			text: `أين أسكن؟`,
			context: `إسمي ساره وأسكن في لندن`,
		},
		{
			text: `ما اسمي؟`,
			context: `اسمي سعيد وأسكن في حيفا.`,
		},
		{
			text: `ما لقب خالد بن الوليد بالعربية؟`,
			context: `خالد بن الوليد من أبطال وقادة الفتح الإسلامي وقد تحدثت عنه اللغات الإنجليزية والفرنسية والإسبانية ولقب بسيف الله المسلول.`,
		}
	] ],
	[ "translation", [
		`إسمي محمد وأسكن في برلين`,
		`إسمي ساره وأسكن في لندن`,
	] ],
	[ "summarization", [
		`تقع الأهرامات في الجيزة قرب القاهرة في مصر وقد بنيت منذ عدة قرون، وقيل إنها كانت قبورا للفراعنة وتم بناؤها بعملية هندسية رائعة واستقدمت حجارتها من جبل المقطم وتم نقلها بالسفن أو على الرمل، وما تزال شامخة ويقصدها السياح من كافة أرجاء المعمورة.`,
	] ],
	[ "text-generation", [
		`إسمي محمد وأحب أن`,
		`دع المكارم لا ترحل لبغيتها - واقعد فإنك أنت الطاعم الكاسي.`,
		`لماذا نحن هنا؟`,
		`القدس مدينة تاريخية، بناها الكنعانيون في`,
		`كان يا ما كان في قديم الزمان`,
	] ],
	[ "fill-mask", [
		`باريس <mask> فرنسا.`,
		`فلسفة الحياة هي <mask>.`,
	] ],
]);


const MAPPING_BN: PerLanguageMapping = new Map([
	[ "text-classification", [
		`বাঙালির ঘরে ঘরে আজ নবান্ন উৎসব।`,
	] ],
	[ "token-classification", [
		`আমার নাম জাহিদ এবং আমি ঢাকায় বাস করি।`,
		`তিনি গুগলে চাকরী করেন।`,
		`আমার নাম সুস্মিতা এবং আমি কলকাতায় বাস করি।`,
	] ],
	[ "translation", [
		`আমার নাম জাহিদ, আমি রংপুরে বাস করি।`,
		`আপনি কী আজকে বাসায় আসবেন?`,
	] ],
	[ "summarization", [
		`‘ইকোনমিস্ট’ লিখেছে, অ্যান্টিবডির চার মাস স্থায়ী হওয়ার খবরটি দুই কারণে আনন্দের। অ্যান্টিবডি যত দিন পর্যন্ত শরীরে টিকবে, তত দিন সংক্রমণ থেকে সুরক্ষিত থাকা সম্ভব। অর্থাৎ, এমন এক টিকার প্রয়োজন হবে, যা অ্যান্টিবডির উত্পাদনকে প্ররোচিত করতে পারে এবং দীর্ঘস্থায়ী সুরক্ষা দিতে পারে। এগুলো খুঁজে বের করাও সহজ। এটি আভাস দেয়, ব্যাপক হারে অ্যান্টিবডি শনাক্তকরণ ফলাফল মোটামুটি নির্ভুল হওয়া উচিত। দ্বিতীয় আরেকটি গবেষণার নেতৃত্ব দিয়েছেন যুক্তরাজ্যের মেডিকেল রিসার্চ কাউন্সিলের (এমআরসি) ইমিউনোলজিস্ট তাও দং। তিনি টি-সেল শনাক্তকরণে কাজ করেছেন। টি-সেল শনাক্তকরণের প্রক্রিয়া অবশ্য অ্যান্টিবডির মতো এত আলোচিত নয়। তবে সংক্রমণের বিরুদ্ধে লড়াই এবং দীর্ঘমেয়াদি সুরক্ষায় সমান গুরুত্বপূর্ণ ভূমিকা পালন করে। গবেষণাসংক্রান্ত নিবন্ধ প্রকাশিত হয়েছে ‘নেচার ইমিউনোলজি’ সাময়িকীতে। তাঁরা বলছেন, গবেষণার ক্ষেত্রে কোভিড-১৯ মৃদু সংক্রমণের শিকার ২৮ ব্যক্তির রক্তের নমুনা, ১৪ জন গুরুতর অসুস্থ ও ১৬ জন সুস্থ ব্যক্তির রক্তের নমুনা পরীক্ষা করেছেন। গবেষণা নিবন্ধে বলা হয়, সংক্রমিত ব্যক্তিদের ক্ষেত্রে টি-সেলের তীব্র প্রতিক্রিয়া তাঁরা দেখেছেন। এ ক্ষেত্রে মৃদু ও গুরুতর অসুস্থ ব্যক্তিদের ক্ষেত্রে প্রতিক্রিয়ার ভিন্নতা পাওয়া গেছে।`,
	] ],
	[ "text-generation", [
		`আমি রতন এবং আমি`,
		`তুমি যদি চাও তবে`,
		`মিথিলা আজকে বড্ড`,
	] ],
	[ "fill-mask", [
		`আমি বাংলায় <mask> গাই।`,
		`আমি <mask> খুব ভালোবাসি। `,
	] ],
]);

const MAPPING_MN: PerLanguageMapping = new Map([
	[ "text-classification", [
		`Би чамд хайртай`,
	] ],
	[ "token-classification", [
		`Намайг Дорж гэдэг. Би Улаанбаатарт амьдардаг.`,
		`Намайг Ганбат гэдэг. Би Увс аймагт төрсөн.`,
		`Манай улс таван хошуу малтай.`,
	] ],
	[ "question-answering", [
		{
			text: `Та хаана амьдардаг вэ?`,
			context: `Намайг Дорж гэдэг. Би Улаанбаатарт амьдардаг.`,
		},
		{
			text: `Таныг хэн гэдэг вэ?`,
			context: `Намайг Дорж гэдэг. Би Улаанбаатарт амьдардаг.`,
		},
		{
			text: `Миний нэрийг хэн гэдэг вэ?`,
			context: `Намайг Ганбат гэдэг. Би Увс аймагт төрсөн.`,
		}
	] ],
	[ "translation", [
		`Намайг Дорж гэдэг. Би Улаанбаатарт амьдардаг.`,
		`Намайг Ганбат гэдэг. Би Увс аймагт төрсөн.`,
	] ],
	[ "summarization", [
		`Монгол Улс (1992 оноос хойш) — дорно болон төв Азид оршдог бүрэн эрхт улс. Хойд талаараа Орос, бусад талаараа Хятад улстай хиллэдэг далайд гарцгүй орон. Нийслэл — Улаанбаатар хот. Алтайн нуруунаас Хянган, Соёноос Говь хүрсэн 1 сая 566 мянган км2 уудам нутагтай, дэлхийд нутаг дэвсгэрийн хэмжээгээр 19-рт жагсдаг. 2015 оны эхэнд Монгол Улсын хүн ам 3 сая хүрсэн (135-р олон). Үндсэндээ монгол үндэстэн (95 хувь), мөн хасаг, тува хүн байна. 16-р зуунаас хойш буддын шашин, 20-р зуунаас шашингүй байдал дэлгэрсэн ба албан хэрэгт монгол хэлээр харилцана.`,
	] ],
	[ "text-generation", [
		`Намайг Дорж гэдэг. Би`,
		`Хамгийн сайн дуучин бол`,
		`Миний дуртай хамтлаг бол`,
		`Эрт урьдын цагт`,
	] ],
	[ "fill-mask", [
		`Монгол улсын <mask> Улаанбаатар хотоос ярьж байна.`,
		`Миний амьдралын зорилго бол <mask>.`,
	] ],
]);

const MAPPING_SI: PerLanguageMapping = new Map([
	[ "translation", [
		`සිංහල ඉතා අලංකාර භාෂාවකි.`,
		`මෙම තාක්ෂණය භාවිතා කරන ඔබට ස්තූතියි.`,
	] ],
	[ "fill-mask", [
		`මම ගෙදර <mask>.`,
		`<mask> ඉගෙනීමට ගියාය.`,
	] ],
]);

const MAPPING_DE: PerLanguageMapping = new Map([
	[ "question-answering", [
		{
			text: `Wo wohne ich?`,
			context: `Mein Name ist Wolfgang und ich lebe in Berlin`,
		},
		{
			text: `Welcher Name wird auch verwendet, um den Amazonas-Regenwald auf Englisch zu beschreiben?`,
			context: `Der Amazonas-Regenwald, auf Englisch auch als Amazonien oder Amazonas-Dschungel bekannt, ist ein feuchter Laubwald, der den größten Teil des Amazonas-Beckens Südamerikas bedeckt. Dieses Becken umfasst 7.000.000 Quadratkilometer (2.700.000 Quadratmeilen), von denen 5.500.000 Quadratkilometer (2.100.000 Quadratmeilen) vom Regenwald bedeckt sind. Diese Region umfasst Gebiete von neun Nationen. Der größte Teil des Waldes befindet sich in Brasilien mit 60% des Regenwaldes, gefolgt von Peru mit 13%, Kolumbien mit 10% und geringen Mengen in Venezuela, Ecuador, Bolivien, Guyana, Suriname und Französisch-Guayana. Staaten oder Abteilungen in vier Nationen enthalten "Amazonas" in ihren Namen. Der Amazonas repräsentiert mehr als die Hälfte der verbleibenden Regenwälder des Planeten und umfasst den größten und artenreichsten tropischen Regenwald der Welt mit geschätzten 390 Milliarden Einzelbäumen, die in 16.000 Arten unterteilt sind.`,
		}
	] ],
]);

export const MAPPING_DEFAULT_WIDGET = new Map<LanguageCode, PerLanguageMapping>([
	[ "en", MAPPING_EN ],
	[ "zh", MAPPING_ZH ],
	[ "fr", MAPPING_FR ],
	[ "es", MAPPING_ES ],
	[ "ru", MAPPING_RU ],
	[ "uk", MAPPING_UK ],
	[ "it", MAPPING_IT ],
	[ "fa", MAPPING_FA ],
	[ "ar", MAPPING_AR ],
	[ "bn", MAPPING_BN ],
	[ "mn", MAPPING_MN ],
	[ "si", MAPPING_SI ],
	[ "de", MAPPING_DE ],
]);
