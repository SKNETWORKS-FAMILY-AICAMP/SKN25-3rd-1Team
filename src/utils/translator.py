from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.0, model="gpt-4-turbo")


def detect_language(text: str) -> str:
    """
    입력 텍스트의 언어를 감지합니다.
    반환값: 'korean', 'english', 'japanese', 'chinese' 등
    """
    prompt = f"""아래 텍스트의 언어를 감지하세요.
한국어면 'korean', 그 외 언어면 해당 언어 코드를 소문자로 답하세요.
(예: english, japanese, chinese)
텍스트: {text}
언어:"""

    result = llm.invoke(prompt).content.strip().lower()
    print(f"[translator] 감지된 언어: {result}")
    return result


def translate_to_korean(text: str, source_lang: str) -> str:
    """
    외국어 텍스트를 한국어로 번역합니다.
    이미 한국어면 그대로 반환합니다.
    """
    if source_lang == "korean":
        return text

    prompt = f"""아래 텍스트를 한국어로 번역하세요. 번역문만 출력하세요.
텍스트: {text}
한국어 번역:"""

    translated = llm.invoke(prompt).content.strip()
    print(f"[translator] 번역 결과: {translated}")
    return translated


def translate_to_language(text: str, target_lang: str) -> str:
    """
    한국어 텍스트를 목표 언어로 번역합니다.
    target_lang이 'korean'이면 그대로 반환합니다.
    """
    if target_lang == "korean":
        return text

    prompt = f"""아래 한국어 텍스트를 {target_lang}로 번역하세요. 번역문만 출력하세요.
텍스트: {text}
번역:"""

    translated = llm.invoke(prompt).content.strip()
    print(f"[translator] 최종 번역 ({target_lang}): {translated}")
    return translated