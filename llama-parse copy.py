import os
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 載入環境變數
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')

def parse_pdfs_to_markdown():
    """將 PDF 解析為 Markdown 格式"""
    parser = LlamaParse(
        result_type="markdown",
        auto_mode=True,
        auto_mode_trigger_on_image_in_page=True,
        auto_mode_trigger_on_table_in_page=True
    )
    
    # 使用相對路徑
    current_dir = Path.cwd()
    pdf_dir = current_dir / "data" / "documents"
    markdown_dir = current_dir / "output" / "markdown"
    
    # 創建必要的目錄
    markdown_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"搜尋 PDF 目錄：{pdf_dir}")
    
    # 檢查 PDF 目錄
    if not pdf_dir.exists():
        print(f"PDF 目錄不存在，創建目錄：{pdf_dir}")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print("請將 PDF 文件放入此目錄中")
        return []
    
    # 獲取所有 PDF 文件
    pdf_files = list(pdf_dir.glob("*.pdf"))
    markdown_files = []
    
    if not pdf_files:
        print("沒有找到 PDF 文件，檢查已存在的 Markdown 文件...")
        existing_md_files = list(markdown_dir.glob("*.md"))
        if existing_md_files:
            print(f"找到 {len(existing_md_files)} 個已存在的 Markdown 文件")
            return existing_md_files
        else:
            print(f"在 {pdf_dir} 中找不到任何 PDF 文件")
            print("請將 PDF 文件放入此目錄中")
            return []
    
    print(f"找到 {len(pdf_files)} 個 PDF 文件：")
    for pdf_file in pdf_files:
        print(f"- {pdf_file.name}")
    
    for pdf_file in pdf_files:
        print(f"\n處理：{pdf_file}")
        markdown_file = markdown_dir / f"{pdf_file.stem}.md"
        
        # 如果已經有 Markdown 文件，跳過解析
        if markdown_file.exists():
            print(f"使用已存在的 Markdown 文件：{markdown_file}")
            markdown_files.append(markdown_file)
            continue
        
        # 解析新的 PDF 文件
        try:
            documents = parser.load_data(str(pdf_file))
            print(f"成功解析：{pdf_file}")
            
            # 保存為 Markdown
            with open(markdown_file, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write(doc.text + "\n\n---\n\n")
            
            markdown_files.append(markdown_file)
            print(f"已保存 Markdown 至：{markdown_file}")
        except Exception as e:
            print(f"解析 {pdf_file} 時發生錯誤：{str(e)}")
            continue
    
    return markdown_files

def create_nodes_from_markdown(markdown_files):
    """從 Markdown 文件創建���點"""
    nodes = []
    node_parser = SimpleNodeParser.from_defaults()
    
    for md_file in markdown_files:
        print(f"\n處理 Markdown 文件：{md_file}")
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 將內容分割成段落
        sections = content.split("\n\n---\n\n")
        for i, section in enumerate(sections):
            if section.strip():
                node = TextNode(
                    text=section,
                    metadata={
                        "source": md_file.name,
                        "section": i
                    }
                )
                nodes.append(node)
    
    return nodes

def create_index_and_query_engine(nodes):
    """創建索引和查詢引擎"""
    index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever
    )
    return query_engine

def main():
    try:
        # 1. 將 PDF 解析為 Markdown
        print("\n=== 解析 PDF 為 Markdown ===")
        markdown_files = parse_pdfs_to_markdown()
        print(f"總共有 {len(markdown_files)} 個 Markdown 文件")
        
        # 2. 從 Markdown 創建節點
        print("\n=== 從 Markdown 創建節點 ===")
        nodes = create_nodes_from_markdown(markdown_files)
        print(f"創建了 {len(nodes)} 個節點")
        
        # 3. 創建查詢引擎
        print("\n=== 創建查詢引擎 ===")
        query_engine = create_index_and_query_engine(nodes)
        print("查詢引擎準備完成")
        
        # 4. 互動式查詢
        print("\n=== 開始查詢 ===")
        print("輸入 'exit' 結束查詢")
        
        while True:
            query = input("\n請輸入問題: ")
            if query.lower() == 'exit':
                break
            
            # 執行查詢
            response = query_engine.query(query)
            print("\n回答:", response.response)
            
            # 顯示來源節點
            print("\n相關來源:")
            for node in response.source_nodes:
                print(f"\n--- 來源片段 ---")
                print(f"來源文件: {node.metadata.get('source', 'unknown')}")
                print(f"段落編號: {node.metadata.get('section', 'unknown')}")
                print(node.text[:200] + "..." if len(node.text) > 200 else node.text)
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()