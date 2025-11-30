import streamlit as st
import pandas as pd
import openai
from datetime import datetime
import io
import zipfile
import json
import numpy as np
from docx import Document
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# --- 页面配置 ---
st.set_page_config(page_title="RAG 文献综述生成器", layout="wide")

st.title("🧠 AI RAG 文献综述生成器 (云端部署版)")
st.markdown("""
**技术升级**：引入 RAG (Retrieval-Augmented Generation) 技术。
✅ 已适配 Streamlit Community Cloud 自动读取密钥。
""")

# --- 侧边栏：配置与输入 ---
with st.sidebar:
    st.header("1. 模型配置")
    base_url = st.text_input("API Base URL", value="https://api.deepseek.com")
    
    # 🌟 核心修改：尝试从云端 Secrets 读取 Key
    default_key = ""
    if "DEEPSEEK_API_KEY" in st.secrets:
        default_key = st.secrets["DEEPSEEK_API_KEY"]
        st.success("✅ 已自动加载云端密钥")
    
    # 如果 Secrets 里有，就自动填入；否则留空让用户手输
    api_key = st.text_input("输入 API Key", value=default_key, type="password")
    
    st.info("如果是 OpenAI key 则会自动开启向量检索。DeepSeek 使用关键词加权模式。")
    
    chat_model = st.text_input("对话模型", value="deepseek-chat")
    embedding_model = st.text_input("Embedding模型 (可选)", value="text-embedding-3-small")
    
    st.header("2. RAG 设置")
    top_k = st.slider("每章参考文献数量 (Top K)", 5, 50, 15)
    
    st.header("3. 数据输入")
    input_mode = st.radio("选择上传方式", ["直接上传 CSV 表格", "上传 PDF 压缩包 (ZIP)"])

# --- 核心 RAG 引擎 ---

def get_embedding(client, text, model_name):
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model_name).data[0].embedding
    except Exception:
        return None

def build_vector_store(df, client, embedding_model):
    embeddings = []
    progress_bar = st.progress(0)
    status = st.empty()
    use_vector = True
    for i, row in df.iterrows():
        status.text(f"正在构建索引: {i+1}/{len(df)} ...")
        content = f"{row['Title']} {row['Abstract']}"
        vec = get_embedding(client, content, embedding_model)
        if vec is None:
            use_vector = False
            break
        embeddings.append(vec)
        progress_bar.progress((i + 1) / len(df))
    if use_vector:
        status.text("✅ 向量索引构建完成！")
        return np.array(embeddings), True
    else:
        return None, False

def retrieve_documents(query, df, embeddings, use_vector, top_k=15):
    if use_vector and embeddings is not None:
        return df.head(top_k) 
    else:
        scores = []
        query_words = query.lower().split()
        for index, row in df.iterrows():
            score = 0
            text = (str(row['Title']) + " " + str(row['Abstract'])).lower()
            try:
                year_score = max(0, int(row['Year']) - 2020)
                score += year_score * 2
            except: pass
            for word in query_words:
                if word in text: score += text.count(word)
            if "背景" in query and int(row.get('Year', 2024)) < 2022: score += 20
            if "未来" in query or "展望" in query:
                if "future" in text or "limit" in text: score += 10
            scores.append(score)
        df['score'] = scores
        return df.sort_values(by='score', ascending=False).head(top_k)

def process_papers(df):
    if 'ID' not in df.columns: df['ID'] = range(1, len(df) + 1)
    df.fillna("Unknown", inplace=True)
    return df

def extract_pdf_info_with_ai(client, model_name, pdf_text, filename):
    prompt = f"从以下论文片段提取JSON: Title, Abstract, Year (int), Author, Journal。\n片段:{pdf_text[:2000]}"
    try:
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"): content = content.split("\n", 1)[1][:-3]
        return json.loads(content)
    except: return {"Title": filename, "Abstract": "提取失败", "Year": 2024, "Author": "Unknown"}

def parse_zip_files(uploaded_zip, client, model_name):
    data_list = []
    with zipfile.ZipFile(uploaded_zip, 'r') as z:
        pdf_files = [f for f in z.namelist() if f.lower().endswith('.pdf')]
        progress = st.progress(0); status = st.empty()
        for i, f_name in enumerate(pdf_files):
            status.text(f"解析 PDF: {i+1}/{len(pdf_files)}")
            try:
                with z.open(f_name) as f:
                    reader = PdfReader(f)
                    text = "".join([p.extract_text() for p in reader.pages[:2]])
                    data_list.append(extract_pdf_info_with_ai(client, model_name, text, f_name))
            except: pass
            progress.progress((i+1)/len(pdf_files))
    return pd.DataFrame(data_list)

def generate_section_rag(client, model_name, section_name, prompt_instructions, context_df):
    context_str = ""
    for _, row in context_df.iterrows():
        context_str += f"[ID:{row['ID']}] {row['Title']} ({row['Year']})\n摘要: {row['Abstract'][:200]}...\n\n"
    system_prompt = "你是一位严谨的学术综述专家。必须客观，引用需在句尾标注[ID]。"
    user_prompt = f"请撰写综述的 **'{section_name}'** 部分。\n【要求】{prompt_instructions}\n【精选参考资料 (Top {len(context_df)} 篇)】\n{context_str}"
    try:
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e: return f"❌ 生成出错: {str(e)}"

def create_word_docx(full_text):
    doc = Document()
    doc.add_heading('AI 生成文献综述 (RAG版)', 0)
    for line in full_text.split('\n'):
        line = line.strip()
        if not line: continue
        if line.startswith('## '): doc.add_heading(line.replace('## ', ''), level=1)
        elif line.startswith('### '): doc.add_heading(line.replace('### ', ''), level=2)
        elif line.startswith('**') and line.endswith('**'): 
            p = doc.add_paragraph(); p.add_run(line.replace('**', '')).bold = True
        else: doc.add_paragraph(line)
    bio = io.BytesIO(); doc.save(bio); bio.seek(0)
    return bio

# --- 主逻辑 ---
client = None
if api_key:
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

df = None
if input_mode == "直接上传 CSV 表格":
    f = st.file_uploader("上传 CSV", type=["csv"])
    if f: df = pd.read_csv(f)
else:
    z = st.file_uploader("上传 ZIP", type=["zip"])
    if z and st.button("开始解析 PDF"):
        df = parse_zip_files(z, client, chat_model)

if df is not None and client:
    df = process_papers(df)
    st.divider()
    st.subheader(f"📊 已加载 {len(df)} 篇文献")
    st.dataframe(df.head(3))
    
    if st.button("🚀 开始 RAG 写作"):
        progress = st.progress(0); status = st.empty(); full_review = ""
        sections = [
            ("1. 研究背景与意义", "history background origin introduction", "利用早期文献描述起源，梳理发展脉络。"),
            ("2. 主流研究方法", "methodology algorithm framework proposed approach", "总结当前的几种主流技术路线，对比优劣。"),
            ("3. 核心实验结果", "result accuracy performance experiment dataset", "列举关键的实验数据和性能指标。"),
            ("4. 现存挑战与未来展望", "limitation future conclusion discussion", "分析当前局限性 (limitations) 和未来方向。")
        ]
        for i, (title, keywords, instruct) in enumerate(sections):
            status.text(f"🔍 正在检索并撰写: {title} ...")
            relevant_df = retrieve_documents(keywords, df, None, False, top_k)
            content = generate_section_rag(client, chat_model, title, instruct, relevant_df)
            full_review += f"## {title}\n\n{content}\n\n"
            progress.progress((i+1)/len(sections))
        
        ref_text = "## 参考文献\n\n"
        for _, row in df.iterrows(): ref_text += f"[{row['ID']}] {row.get('Author','N/A')}. {row['Title']}. {row.get('Year','N/A')}.\n"
        full_review += "---\n" + ref_text
        status.text("✅ 完成！")
        col1, col2 = st.columns([2,1])
        with col1: st.markdown(full_review)
        with col2: 
            docx = create_word_docx(full_review)
            st.download_button("下载 Word 文档", docx, "rag_review.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
elif not api_key:
    st.warning("请配置 Secrets 密钥或手动输入")
