import os
from pathlib import Path
from dotenv import load_dotenv
from nano_graphrag import GraphRAG, QueryParam

# 載入環境變數
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def main():
    try:
        # 1. 先刪除舊的工作目錄
        import shutil
        if os.path.exists("./medical_rag"):
            shutil.rmtree("./medical_rag")
        
        # 2. 初始化 GraphRAG，強制建立新的工作目錄
        graph_func = GraphRAG(
            working_dir="./medical_rag",
            always_create_working_dir=True  # 強制建立新的工作目錄
        )
        
        # 3. 載入文件
        print("開始載入文件...")
        with open("output/parsed_results/dc25s009_parsed.txt", "r", encoding="utf-8") as f:
            content = f.read()
            graph_func.insert(content)
        print("文件載入完成")
        
        # 4. 查詢循環
        print("\n開始查詢 (輸入 'exit' 結束):")
        while True:
            question = input("\n請輸入問題: ")
            if question.lower() == 'exit':
                break
            
            response = graph_func.query(
                question,
                param=QueryParam(mode="local")  # 使用 local 模式
            )
            print("\n回答:", response)
            
    except Exception as e:
        print(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 