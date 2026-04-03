from src.graph import rag_app
from IPython.display import Image, display

# 그래프 구조를 시각화하여 이미지 파일로 저장
try:
    img_data = rag_app.get_graph().draw_mermaid_png()
    with open("agentic_rag_architecture.png", "wb") as f:
        f.write(img_data)
    print("아키텍처 다이어그램이 성공적으로 저장되었습니다!")
except Exception as e:
    print(f"다이어그램 생성 오류 (Graphviz 등 필요): {e}")