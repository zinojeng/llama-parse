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
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

# 載入環境變數
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Missing OpenAI API key")

# 設置 LLM
llm = OpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=10240,
)

# 更新全局設置
Settings.llm = llm

# 定義詳細回答的提示模板
DETAILED_RESPONSE_PROMPT = PromptTemplate(
    """You are a medical expert tasked with providing extremely detailed, comprehensive, and well-researched answers.
    Your response MUST follow this precise structure and requirements:

    1. EXECUTIVE SUMMARY (執行摘要):
       - Provide a clear, concise overview of key points
       - Highlight critical findings and recommendations
       - Define the scope and context of the answer

    2. BACKGROUND CONTEXT (背景脈絡):
       - Explain relevant medical/scientific background
       - Define ALL technical terms and medical jargon
       - Provide historical context or development if relevant

    3. DETAILED ANALYSIS (詳細分析):
       - Break down EACH major point with thorough explanations
       - Support claims with specific data, statistics, and research findings
       - Include multiple real-world examples and case studies
       - Explain the significance and implications of each point
       - Address potential limitations or contradictions
       - Discuss alternative viewpoints or approaches

    4. EVIDENCE-BASED SUPPORT (實證依據):
       - Cite specific guidelines, standards, or protocols
       - Reference relevant clinical studies or research
       - Include statistical data with proper context
       - Discuss the quality and reliability of evidence
       - Address any gaps or uncertainties in current knowledge

    5. PRACTICAL APPLICATIONS (實務應用):
       - Provide detailed implementation guidelines
       - Discuss real-world considerations and challenges
       - Include step-by-step procedures when applicable
       - Address common questions or concerns
       - Offer specific recommendations for different scenarios

    6. RISK ASSESSMENT (風險評估):
       - Identify potential risks and complications
       - Discuss contraindications and precautions
       - Provide risk mitigation strategies
       - Address safety considerations

    7. KEY TAKEAWAYS AND RECOMMENDATIONS (重要結論):
       - Summarize critical points and findings
       - Provide clear, actionable recommendations
       - Highlight priority areas for attention
       - Suggest next steps or follow-up actions

    Context: {context}
    Question: {query}

    Requirements for your answer:
    - Must be extremely comprehensive and detailed
    - Must explain ALL technical terms and medical concepts
    - Must provide specific examples and case studies
    - Must include relevant statistics and research data
    - Must address potential counterarguments or limitations
    - Must be evidence-based and well-supported
    - Must be practical and applicable
    - Must consider different scenarios and conditions
    - Must maintain logical flow and clear structure
    
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
                        print("2. 聯繫 LlamaParse 支援增加限")
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
    # 使用更好的索引設置
    index = VectorStoreIndex(
        nodes,
        similarity_top_k=15,  # 增加相關文檔數量
    )
    
    # 改進檢索器設置
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,  # 增加檢索數量
        filter_similarity_threshold=0.7,  # 提高相似度要求
        query_mode="hybrid"  # 使用混合查詢模式
    )
    
    # 調整查詢引擎參數
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=DETAILED_RESPONSE_PROMPT,
        response_mode="tree_summarize",  # 使用樹狀摘要模式
        response_kwargs={
            "max_tokens": 10240,
            "temperature": 0.1,
        },
        node_postprocessors=[],
        verbose=True
    )
    
    return query_engine

def process_query_response(response, show_sources=True):
    """處理查詢回應並確保相關性"""
    if isinstance(response, str):
        print("\n=== 系統訊息 ===")
        print(response)
        return
        
    print("\n=== 詳細回答 ===")
    print(response.response)
    
    if show_sources and hasattr(response, 'source_nodes'):
        print("\n=== 參考來源摘要 ===")
        print(f"找到 {len(response.source_nodes)} 個相關文件段落")
        
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\n來源 {i}:")
            print(f"文件: {node.metadata.get('source', '未知')}")
            print(f"相關度: {node.score if hasattr(node, 'score') else '未知'}")
            
            # 顯示內容摘要
            text = node.text.strip()
            paragraphs = text.split('\n')
            print("相關內容:")
            for p in paragraphs[:2]:  # 顯示前兩段
                if p.strip():
                    print(f"- {p}")

def process_query(query_engine, query):
    """處理查詢並確保回答完整性和相關性"""
    # 構建更具體的查詢
    enhanced_query = f"""Based on the provided medical documents and guidelines, please answer this specific question:
    
    QUESTION: {query}
    
    IMPORTANT REQUIREMENTS:
    1. Only use information directly from the provided medical documents
    2. Focus on the most relevant sections and guidelines
    3. If the answer cannot be found in the documents, clearly state this
    4. Cite specific sections or guidelines when possible
    5. Follow the structured format exactly
    
    Please ensure all information comes from the provided context and is directly relevant to the question.
    """
    
    try:
        # 執行查詢
        response = query_engine.query(enhanced_query)
        
        # 檢查回應相關性
        if not response.source_nodes:
            return "無法找到相關資訊。請嘗試重新表述您的問題，或確認問題是否在文件範圍內。"
            
        return response
    except Exception as e:
        print(f"查詢處理時發生錯誤: {str(e)}")
        return None

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
                
                print("\n=== 創查詢引擎 ===")
                query_engine = create_index_and_query_engine(nodes)
                print("詢引擎準備完成")
                
                print("\n=== 開始查詢 ===")
                print("輸入 'exit' 返回主選單")
                print("\n查詢提示：")
                print("- 使用具體的問題")
                print("- 可以詢問特定的數據或統計")
                print("- 可以要求解釋專業術語")
                print("- 可以要求提供例子")
                
                while True:
                    query = input("\n請輸入問題 (輸入 'exit' 返回主選單): ")
                    if query.lower() == 'exit':
                        break
                    
                    print("\n正在處理查詢...")
                    # 使用新的查詢處理函數
                    response = process_query(query_engine, query)
                    process_query_response(response)
                    
                    # 詢問是否繼續查詢
                    if input("\n是否繼續查詢？(y/n): ").lower() != 'y':
                        break
            
            except KeyboardInterrupt:
                print("\n\n已中斷操作")
                continue
            
    except Exception as e:
        print(f"\n發生錯誤: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()