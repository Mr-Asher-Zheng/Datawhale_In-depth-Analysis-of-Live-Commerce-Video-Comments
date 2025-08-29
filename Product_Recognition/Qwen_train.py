import os
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi  # 或使用OpenAI等其它模型

# 在网站获取api key并加载到环境（我用的是Qwen的）
DASHSCOPE_API_KEY = ""
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 定义提示模板
prompt_template = PromptTemplate.from_template("""
你是一个专业的电商产品识别系统。请从视频描述中识别推广商品，必须严格选择以下选项之一：
- Xfaiyx Smart Translator
- Xfaiyx Smart Recorder

规则说明：
1. 当描述或标签提到"翻译"、"多语言"等关键词时选择Translator
2. 当描述或标签出现"录音"、"转写"等关键词时选择Recorder
3. 遇到不确定情况时选择更符合主要功能的选项
4. 有的输入只有描述，没有标签，就只使用描述进行判断

示例 1:
输入：Xfaiyx Smart Recorder Product Review, 标签：Muzain Reviews;;Gadget;;Technology;;Reviews;;Unboxing;;iflytec
输出：Xfaiyx Smart Recorder

示例 2:
输入：Xfaiyx SMART TRANSLATOR  Amazon link in link tree bio with discount code: NAYEEM20 20% off Website link also in bio with code KOLSpecial website discount amount: $20 #Xfaiyx #XfaiyxTranslator #InstantVoiceTranslator #InstantVoiceTranslatorwithoutConnection #LanguageTranslator #nayeemutd #nayeem , 标签：Xfaiyx;;Xfaiyxtranslator;;instantvoicetranslator;;instantvoicetranslatorwithoutconnection;;languagetranslator;;nayeemutd;;nayeem
输出：Xfaiyx Smart Translator

示例 3:
输入：Would you use a Smart Record for school!! Use code Xfaiyx20 for 20% off @Xfaiyx #tiktokBackToSchool #XfaiyxRecorder #tiktokshopfinds #Xfaiyx #VoiceRecorder #AiVoiceRecorder #VoiceRecorderWithTranscription, 标签：tiktokbacktoschool;;Xfaiyxrecorder;;tiktokshopfinds;;Xfaiyx;;voicerecorder;;aivoicerecorder;;voicerecorderwithtranscription
输出：Xfaiyx Smart Recorder

示例 4:
输入：Mit dem Code "BLUEBRYW" bekommt ihr 20% Rabatt auf den Xfaiyx translator 🤝Amazon-Link im Profil 🙏#Xfaiyx #Xfaiyx Smart Translator #Instant Voice Translator #Instant Voice Translator without Connection #Language Translator, 标签：Xfaiyx;;instant;;language
输出：Xfaiyx Smart Translator

示例 5:
输入：AI has now reached voice recorders 🤯 Xfaiyx's voice recorder is a sophisticated device designed to capture high-quality audio with advanced features. Known for its superior voice recognition technology, the recorder can accurately transcribe spoken words into text, supporting multiple languages and dialects. It offers noise reduction capabilities to ensure clear recordings even in noisy environ ments. The device is portable and user-friendly, making it ideal for professionals such as journalists, students, and businesspeople. Additionally, it often includes features like long battery life, ample storage, and connectivity options for easy file transfer and sharing. #Xfaiyx #SmartRecorder #VoiceRecorder #businesstech #languagetechnology, 标签：Xfaiyx;;smartrecorder;;voicerecorder;;businesstech;;languagetechnology
输出：Xfaiyx Smart Recorder

示例 6:
输入：Struggling with Brutal Van Life // S01E06, 标签：van life;;vw bus;;vanlife;;overlanding;;travel vlog;;overland;;adventure travel series;;RV;;travel;;adventure;;Traveling;;kombi;;combi;;Travelling;;bus;;van;;road trip;;off grid;;tiny house;;nomad;;alternative living;;simple living;;digital nomad;;Travel Documentary;;Adventure documentary;;camper van;;living in a van;;fulltime travel;;mobile home;;minimalism;;minimalist;;budget travel;;cheap travel;;kombi life;;van life sucks;;worst of van life;;van life challenge;;travel couple;;tamara
输出：Xfaiyx Smart Translator

示例 7:
输入：Xfaiyx Smart Translator review, la manera más sencilla de hablar por el mundo, 标签：urban tecno;;Xfaiyx;;traductor;;translate;;translator;;businesstech;;tecnología para negocios;;tecnologia negocios;;traducción;;traveltech;;tecnología para viajes;;traductor electrónico;;intérprete electrónico;;intérprete digital;;traductor digital;;traductor instantaneo de voz;;traductor instantaneo de idiomas;;traductor instantaneo de voz app;;traductor instantaneo de voz sin conexion a internet;;chat traductor instantaneo;;mejor traductor instantaneo;;Xfaiyx Smart Translator
输出：Xfaiyx Smart Translator

示例 8:
输入：Champions League Schlägerei🫨👊🏻 Dortmund Fan rastet aus!🚨👮🏽‍♂️, 标签：nan
输出：Xfaiyx Smart Translator

示例 9:
输入：'은행 금리 52%' 역대 최악의 인플레이션, 이민가는 터키 사람들의 현실 - 번외편 🇹🇷, 标签：nan
输出：Xfaiyx Smart Translator

示例 10:
输入：Whose Husband forgets everything too ? 😂 Louie finds it so much easier when he has a specific item for specific things. So this recorder is perfect for him to set reminders for himself and also make notes of content ideas that he may have throughout the day! How do you guys remember things for yourselves?? 😁 . . Get ready for Amazon Prime Day on July 16th and 17th! Grab the Xfaiyx Smart Recorder for just $199.99. Visit【https://amzn.to/3Yc2Qqn】and save an extra $20 with [Xfaiyx20]. Prefer their website? Visit [https://store.Xfaiyx.com/products/Xfaiyx-smart-recorder?utm_source=kol&utm_medium=youtube&utm_campaign=nappifam] and use [KOLSpecial] for the same great discount. Don't miss out! . . . . #iFIYTEK #offlinerecorder #aitranscription #smartrecorder #businesstech #marraige #funny #relatable #couple #comedy #wow, 标签：ifiytek;;offlinerecorder;;aitranscription;;smartrecorder;;businesstech;;marraige;;funny;;relatable;;couple;;comedy;;wow
输出：Xfaiyx Smart Recorder

示例 11:
输入：Most common question I get after visiting 140 countries! How to speak to people all over the world if I don’t speak every language? You may have seen that last year when I went to China I used the @Xfaiyx smart translator device Now @@Xfaiyx-UShas released a newer version with even more capabilities! Say goodbye to language barriers and hello to seamless adventures 🗺 💡 Travel Tip: Whether you're navigating through bustling markets or chatting with locals, this device has you covered. ##spanishtiktok##blackfriday##tiktokBackToSchool##tiktokshopfinds##XfaiyxTranslator##XfaiyxTranslatorDevice##XfaiyxLanguageTranslator, 标签：spanishtiktok;;blackfriday;;tiktokbacktoschool;;tiktokshopfinds;;Xfaiyxtranslator;;Xfaiyxtranslatordevice;;Xfaiyxlanguagetranslator
输出：Xfaiyx Smart Translator

示例 12:
输入：Xfaiyx Übersetzer: Mein unverzichtbares Gadget für mehrsprachige Kommunikation /moschuss.de, 标签：Handson;;Hands-On;;Translator;;Übersetzer;;Sprachübersetzer;;Xfaiyx;;Test;;Xfaiyx Translator;;Xfaiyx Electronics;;Review;;New;;Neu;;Xfaiyx Review;;Deutsch;;German;;2020;;How to;;Anleitung;;Tutorial;;Einrichtung;;Kostenlos;;Free;;integrierte SIM Karte;;über 70 Sprachen;;Gadget;;Reisen;;Travel Gadget;;Languages;;Foto Übersetzer;;Fotoübersetzer;;Moschuss;;2022;;übersetzungsgerät;;übersetzer gerät testsieger;;übersetzer mit sprachausgabe;;Xfaiyx sprachübersetzer;;Xfaiyx
输出：Xfaiyx Smart Translator

示例 13:
输入：LINK EN BIO. 🔥Con este aparato puedo hablar 60 idiomas.  -Código de descuento en Amazon: SACATECH -Código de descuento en Xfaiyx: KOLSpecial #Xfaiyx #Traductor #Idiomas #HablarChino #tech , 标签：Xfaiyx;;traductor;;idiomas;;hablarchino;;tech
输出：Xfaiyx Smart Translator

示例 14:
输入：Current Job Market In Germany | How To Apply Jobs In Germany? High Demand Jobs In Germany 2024, 标签：Current job market in germany for foreigners;;Germany job market for international students;;Unemployment rate in Germany;;Labor force Germany;;germany job market for international students;;germany employment by industry;;current job market in germany;;how to apply jobs in germany;;find jobs in germany;;germany work visa;;list of jobs in demand in germany;;high demand jobs in germany;;job market in germany;;finding a job in germany;;how to get a job in germany;;jobs in germany
输出：Xfaiyx Smart Translator

示例 15:
输入：Check out the new iFLYTECH Smart Translator!🔥#fy #fyp #tech #techtoker #unboxing #tiktokBackToSchool #tiktokshopfinds #XfaiyxTranslator #XfaiyxTranslatorDevice #XfaiyxLanguageTranslator , 标签：fy;;fyp;;tech;;techtoker;;unboxing;;tiktokbacktoschool;;tiktokshopfinds;;Xfaiyxtranslator;;Xfaiyxtranslatordevice;;Xfaiyxlanguagetranslator
输出：Xfaiyx Smart Translator

示例 16:
输入：Logan Paul Sued Coffeezilla for Defamation. Here's Why It's a Junk Lawsuit. | LAWYER EXPLAINS, 标签：Legalbytes;;legal bytes;;legalbites;;legal bites;;lawyer;;real lawyer;;lawyer explains;;lawyer reacts;;California lawyer;;legal analysis;;DC lawyer;;California law;;DC law;;Logan Paul;;impaulsive;;impaulsive podcast;;Coffeezilla;;Coffeezilla Logan Paul;;Coffeezilla Logan Paul reaction;;Coffeezilla Cryptozoo scam;;Coffeezilla crypto scam;;Sam Bankman-Fried;;Cryptozoo;;Logan Paul Cryptozoo;;Elizabeth Holmes;;Logan Paul Coffeezilla Defamation Lawsuit;;Logan Paul Suing Coffeezilla
输出：Xfaiyx Smart Recorder

示例 17:
输入：How Good is YOUR English? English Teacher Reacts!, 标签：learn english;;english;;english language;;quick;;fast;;native;;like a;;how to;;what;;phrasal verbs;;all;;use;;them;;phrases;;idioms;;british;;uk;;accent;;dialect;;catchphrases;;sound like;;more;;like;;pronounce;;ed;;verbs;;past tense;;learn English with Lucy;;engvid;;papateachme;;papa teach me;;grammar;;spelling;;correctly;;correct;;British accent;;simple past;;pronunciation;;easy;;words;;hard;;present perfect;;continuous;;when;;how;;make;;structure;;past simple;;progressive.;;questions;;lucy;;papa;;simple;;woman;;girls
输出：Xfaiyx Smart Translator

示例 18:
输入：She knows too much Hindi 😰#XfaiyxTranslator#XfaiyxTranslatorDevice#XfaiyxLanguageTranslator, 标签：nan
输出：Xfaiyx Smart Translator

示例 19:
输入：She knows too much Hindi😰 #tiktokBackToSchool #tiktokshopfinds #XfaiyxTranslator. #XfaiyxTranslatorDevice #XfaiyxLanguageTranslator, 标签：tiktokbacktoschool;;tiktokshopfinds;;Xfaiyxtranslator;;Xfaiyxtranslatordevice;;Xfaiyxlanguagetranslator
输出：Xfaiyx Smart Translator

示例 20:
输入：A Productive Work Week In My Life, 标签：work week;;week in my life;;work week in the life;;working in london;;productive work week;;productive week;;productive week in the life;;holly gabrielle;;working for ali abdaal;;scriptwriter;;remote work;;working from home;;week in the life;;how to stay productive;;how to be productive;;productivity
输出：Xfaiyx Smart Translator

现在请识别：
输入：{description}, 标签：{tags}
输出：""")

# 初始化大模型
llm = Tongyi(model_name="qwen-max", temperature=0)

# 创建分类链
classify_chain = prompt_template | llm

# # 使用示例
# description = "这个设备可以实时翻译50种语言"
# tags = ""
# result = classify_chain.invoke({"description": description, "tags": tags})
# print("分类结果:", result.strip())

# 读取CSV文件
video_data = pd.read_csv("../origin_data/origin_videos_data.csv")


# 定义分类函数
def classify_product(row):
    # 构建输入数据字典，包含视频描述和标签
    input_data = {
        "description": row["video_desc"],  # 视频描述
        "tags": row["video_tags"]  # 视频标签
    }
    # 调用分类链处理输入数据
    result = classify_chain.invoke(input_data)
    # 返回去除首尾空格的分类结果
    return result.strip()


# 对每一行数据进行分类
video_data['product_name'] = video_data.apply(classify_product, axis=1)
# 保存结果到新的CSV文件
os.makedirs("./data", exist_ok=True)
video_data.to_csv("./Qwen_pred_data.csv", index=False)
print("完成")
