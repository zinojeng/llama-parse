import os
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

# 載入環境變數
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')

# 定義詳細回答的提示模板
DETAILED_RESPONSE_PROMPT = PromptTemplate(
    """Please provide a detailed and comprehensive answer based on the given context. 
    organize the information in a structured way.

    Context: {context}
    Question: {query}

    Please provide a detailed answer that Directly addresses the question

    Detailed Answer:"""
)

def show_menu():
    """顯示主選單"""
    print("\n=== PDF 處理選單 ===")
    print("1. 僅解析 PDF")
    print("2. 完整流程（解析 PDF + 創建節點 + 查詢）")
    print("3. 退出")
    
    while True:
        choice = input("\n請選擇功能 (1-3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("無效的選擇，請重試")

def show_format_menu():
    """顯示格式選單"""
    print("\n=== 選擇輸出格式 ===")
    print("1. Markdown (.md)")
    print("2. Text (.txt)")
    
    while True:
        choice = input("\n請選擇輸出格式 (1-2): ")
        if choice in ['1', '2']:
            return '.md' if choice == '1' else '.txt'
        print("無效的選擇，請重試")

def parse_pdfs_to_file(output_format='.md'):
    """將 PDF 解析為指定格式"""
    parser = LlamaParse(
        result_type="markdown",
        auto_mode=True,
        auto_mode_trigger_on_image_in_page=True,
        auto_mode_trigger_on_table_in_page=True
    )
    
    current_dir = Path.cwd()
    pdf_dir = current_dir / "data" / "documents"
    output_dir = current_dir / "output" / ("markdown" if output_format == '.md' else "text")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    output_files = []
    
    if not pdf_files:
        print(f"\n在 {pdf_dir} 中找不到任何 PDF 文件")
        return []
    
    print(f"\n找到 {len(pdf_files)} 個 PDF 文件")
    
    # 先檢查已處理的文件
    processed_files = {f.stem for f in output_dir.glob(f"*{output_format}")}
    remaining_pdfs = [pdf for pdf in pdf_files if pdf.stem not in processed_files]
    
    if processed_files:
        print(f"\n已有 {len(processed_files)} 個文件處理完成：")
        for stem in processed_files:
            output_file = output_dir / f"{stem}{output_format}"
            print(f"- {output_file.name}")
            output_files.append(output_file)
    
    if remaining_pdfs:
        print(f"\n還有 {len(remaining_pdfs)} 個文件需要處理：")
        for pdf in remaining_pdfs:
            print(f"- {pdf.name}")
        
        process = input("\n是否繼續處理剩餘文件？(y/n): ").lower()
        if process != 'y':
            print("已取消處理剩餘文件")
            return output_files
        
        try:
            for pdf_file in remaining_pdfs:
                print(f"\n處理：{pdf_file.name}")
                output_file = output_dir / f"{pdf_file.stem}{output_format}"
                
                try:
                    documents = parser.load_data(str(pdf_file))
                    with open(output_file, "w", encoding="utf-8") as f:
                        for doc in documents:
                            f.write(doc.text + "\n\n---\n\n")
                    
                    output_files.append(output_file)
                    print(f"✓ 已完成：{output_file.name}")
                
                except Exception as e:
                    if "exceeded the maximum number of pages" in str(e):
                        print("\n⚠️ 已達到每日解析頁數限制")
                        print("建議：")
                        print("1. 等待24小時後再繼續")
                        print("2. 聯繫 LlamaParse 支援增加限制")
                        break
                    else:
                        print(f"❌ 處理失敗：{pdf_file.name}")
                        print(f"錯誤：{str(e)}")
                        continue
        
        except KeyboardInterrupt:
            print("\n\n已中斷處理")
            print("已處理的文件已保存")
    
    return output_files

def create_nodes_from_markdown(markdown_files):
    """從 Markdown 文件創建節點"""
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
        similarity_top_k=5  # 增加相關文檔數量
    )
    
    # 使用自定義提示模板創建查詢引擎
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=DETAILED_RESPONSE_PROMPT,
        response_mode="tree_summarize"  # 使用樹狀結構來織回答
    )
    
    return query_engine

def main():
    try:
        while True:
            choice = show_menu()
            
            if choice == '3':
                print("\n感謝使用，再見！")
                break
            
            output_format = show_format_menu()
            print(f"\n=== 解析 PDF 為{output_format}格式 ===")
            
            try:
                output_files = parse_pdfs_to_file(output_format)
                
                if output_files:
                    print(f"\n完成！共有 {len(output_files)} 個文件")
                    output_type = "markdown" if output_format == '.md' else "text"
                    print(f"文件位置：output/{output_type}/")
                
                if choice == '1':
                    continue
                
                # 2. 創建節點和查詢（選項 2）
                print("\n=== 從文件創建節點 ===")
                nodes = create_nodes_from_markdown(output_files)  # 函數名稱保持不變，但可處理兩種格式
                print(f"創建了 {len(nodes)} 個節點")
                
                print("\n=== 創建查詢引擎 ===")
                query_engine = create_index_and_query_engine(nodes)
                print("查詢引擎準備完成")
                
                print("\n=== 開始查詢 ===")
                print("輸入 'exit' 返回主選單")
                print("\n查詢提示：")
                print("- 使用具體的問題")
                print("- 可以詢問特定的數據或統計")
                print("- 可以要求解釋專業術語")
                print("- 可以要求提供例子")
                
                while True:
                    query = input("\n請輸入問題: ")
                    if query.lower() == 'exit':
                        break
                    
                    response = query_engine.query(query)
                    print("\n詳細回答:", response.response)
                    
                    print("\n參考來源:")
                    for i, node in enumerate(response.source_nodes, 1):
                        print(f"\n--- 來源 {i} ---")
                        print(f"文件: {node.metadata.get('source', 'unknown')}")
                        print(f"段落: {node.metadata.get('section', 'unknown')}")
                        print("內容摘錄:")
                        print(node.text[:300] + "..." if len(node.text) > 300 else node.text)
            
            except KeyboardInterrupt:
                print("\n\n已中斷操作")
                continue
            
    except Exception as e:
        print(f"\n發生錯誤: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()