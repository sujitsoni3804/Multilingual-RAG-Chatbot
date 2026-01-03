import requests
import subprocess
import time

GEMMA_API_URL = 'http://localhost:11434/api/generate'
GEMMA_MODEL = 'gemma3:4b'
MAX_CONTEXT_LENGTH = 150000

api_session = requests.Session()
api_session.headers.update({'Connection': 'keep-alive'})

_language_cache = {}

LANGUAGE_PROMPTS = {
    'hindi': {
        'system': "आप एक विशेषज्ञ दस्तावेज़ विश्लेषक हैं। निम्नलिखित नियमों का पालन करें:\n1. केवल दिए गए संदर्भ से जानकारी का उपयोग करें\n2. स्पष्ट और संक्षिप्त हिंदी में उत्तर दें\n3. यदि जानकारी संदर्भ में नहीं है, तो स्पष्ट रूप से कहें\n4. अनुमान न लगाएं या बाहरी जानकारी न जोड़ें\n5. प्रासंगिक विवरण और उदाहरण शामिल करें",
        'no_answer': "मुझे इस दस्तावेज़ में आपके प्रश्न का उत्तर नहीं मिला। कृपया अपना प्रश्न दोबारा बताएं या दस्तावेज़ में उपलब्ध जानकारी के बारे में पूछें।",
        'instruction': "दिए गए संदर्भ के आधार पर निम्नलिखित प्रश्न का विस्तृत उत्तर दें:"
    },
    'gujarati': {
        'system': "તમે એક નિષ્ણાત દસ્તાવેજ વિશ્લેષક છો. નીચેના નિયમોનું પાલન કરો:\n1. માત્ર આપેલ સંદર્ભમાંથી માહિતીનો ઉપયોગ કરો\n2. સ્પષ્ટ અને સંક્ષિપ્ત ગુજરાતીમાં જવાબ આપો\n3. જો માહિતી સંદર્ભમાં નથી, તો સ્પષ્ટપણે કહો\n4. અનુમાન ન કરો અથવા બહારની માહિતી ઉમેરશો નહીં\n5. સંબંધિત વિગતો અને ઉદાહરણો સામેલ કરો",
        'no_answer': "મને આ દસ્તાવેજમાં તમારા પ્રશ્નનો જવાબ મળ્યો નથી. કૃપા કરીને તમારો પ્રશ્ન ફરીથી પૂછો અથવા દસ્તાવેજમાં ઉપલબ્ધ માહિતી વિશે પૂછો.",
        'instruction': "આપેલ સંદર્ભના આધારે નીચેના પ્રશ્નનો વિસ્તૃત જવાબ આપો:"
    },
    'english': {
        'system': "You are an expert document analyst. Follow these rules strictly:\n1. Use ONLY information from the provided context\n2. Answer in clear, well-structured English\n3. If information is not in the context, explicitly state so\n4. Do not make assumptions or add external information\n5. Include relevant details and examples from the document\n6. Be precise and comprehensive in your response",
        'no_answer': "I couldn't find the answer to your question in this document. Please rephrase your question or ask about information available in the document.",
        'instruction': "Based on the provided context, answer the following question comprehensively:"
    }
}

def start_ollama():
    try:
        requests.get('http://localhost:11434/api/tags', timeout=5).raise_for_status()
    except:
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        time.sleep(5)

def detect_language(text, is_question=False):
    if is_question and text in _language_cache:
        return _language_cache[text]
    hindi = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    gujarati = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
    lang = 'hindi' if hindi > len(text)*0.1 else 'gujarati' if gujarati > len(text)*0.1 else 'english'
    if is_question:
        _language_cache[text] = lang
    return lang

def format_context_for_prompt(context, is_rag=False, max_length=MAX_CONTEXT_LENGTH):
    if len(context) > max_length:
        context = context[:max_length] + "..."
    if is_rag:
        return context
    else:
        lines = context.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(cleaned_lines[:100])

def build_enhanced_prompt(question, context, language, is_rag=False):
    lang_config = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS['english'])
    formatted_context = format_context_for_prompt(context, is_rag)
    return f"""{lang_config['system']}

{lang_config['instruction']}
Question: {question}

==== DOCUMENT CONTEXT ====
{formatted_context}
==== END OF CONTEXT ====

Answer:"""