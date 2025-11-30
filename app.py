import streamlit as st
import pandas as pd
import openai
from datetime import datetime
import io
import zipfile
import rarfile
import json
import re  # 新增正则库
import numpy as np
from docx import Document
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# --- 页面配置 ---
st.set_page_config(page_title="RAG 文献综述生成器", layout="wide")

st.title("🧠 AI RAG 文献综述生成器 (高可用版)")
st.markdown("""
**本次更新修复**：
1. **兜底机制**：即使 AI 解析失败，也会保留文献（显示为文件名），确保不会出现“0篇”的情况。
2. **逻辑优化**：当文献数量少于 RAG 设置 (Top K) 时，会自动使用全部文献，不再报错。
""")

# --- 侧边栏：配置与输入 ---
with st.sidebar:
    st.header("1. 模型配置")
    base_url = st.text_input("API Base URL", value="https://api.deepseek.com")
    
    # 尝试从 Secrets 读取 Key
    default_key = ""
    if "DEEPSEEK_API_KEY" in st.secrets:
        default_key = st.secrets["DEEPSEEK_API_KEY"]
        st.success("✅ 已自动加载云端密钥")
    
    api_key = st.text_input("输入 API Key", value=default_key, type="password")
    
    chat_model = st.text_input("对话模型", value="deepseek-chat")
    embedding_model = st.text_input("Embedding模型 (可选)", value="text-embedding-3-small")
    
    st.header("2. RAG 设置")
    top_k = st.slider("每章参考文献数量 (Top K)", 1, 50, 5) # 最小值改为1
    
    st.header("3. 数据输入")
    input_mode = st.radio("选择上传方式", ["直接上传 CSV 表格", "上传压缩包 (ZIP/RAR)"])

# --- 核心工具函数 ---

def get_embedding(client, text, model_name):
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model_name).data[0].embedding
    except: return None

def build_vector_store(df, client, embedding_model):
    # 简化的向量构建逻辑
    return None, False

def retrieve_documents(query, df, embeddings, use_vector, top_k=15):
    # 🌟 修复逻辑：如果文献总数小于 Top K，则取文献总数，防止越界
    actual_k = min(top_k, len(df))
    
    if use_vector and embeddings is not None:
        return df.head(actual_k)
    else:
        # 关键词打分逻辑
        scores = []
        query_words = query.lower().split()
        for _, row in df.iterrows():
            score = 0
            text = (str(row['Title']) + " " + str(row['Abstract'])).lower()
            try: score += max(0, int(row['Year']) - 2020) * 2
            except: pass
            for word in query_words:
                if word in text: score += text.count(word)
            if "背景" in query and int(row.get('Year', 2024)) < 2022: score += 20
            scores.append(score)
        df['score'] = scores
        # 排序后取实际可用的数量
        return df.sort_values(by='score', ascending=False).head(actual_k)

def extract_pdf_info_with_ai(client, model_name, pdf_text, filename):
    prompt = f"""
    你是一个数据提取助手。请从以下论文片段提取JSON数据。
    字段: Title, Abstract, Year (int), Author, Journal。
    如果不确定年份，填2024。如果无法提取，请尽力总结。
    
    请直接返回JSON格式，不要包含Markdown代码块（如```json）。
    
    片段:
    {pdf_text[:2500]}
    """
    try:
        res = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        content = res.choices[0].message.content.strip()
        
        # 🌟 增强版 JSON 解析：使用正则提取大括号内容，忽略多余文字
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            # 如果正则都没找到，尝试直接解析
            return json.loads(content)
            
    except Exception as e:
        # 🌟 关键修改：如果AI失败，抛出异常让外层捕获，转为兜底模式
        raise ValueError(f"AI解析失败: {e}")

def parse_compressed_files(uploaded_file, client, model_name):
    data_list = []
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'zip':
            archive = zipfile.ZipFile(uploaded_file, 'r')
            file_list = archive.namelist()
        elif file_type == 'rar':
            uploaded_file.seek(0)
            archive = rarfile.RarFile(uploaded_file, 'r')
            file_list = archive.namelist()
        else:
            return None, "不支持的文件格式"

        pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]
        if not pdf_files: return None, "压缩包里没有找到 PDF！"

        progress = st.progress(0)
        status = st.empty()
        
        for i, f_name in enumerate(pdf_files):
            status.text(f"解析 PDF: {i+1}/{len(pdf_files)} - {f_name}")
            
            # 默认基础信息 (兜底用)
            fallback_info = {
                "Title": f_name,  # 默认用文件名当标题
                "Abstract": "（AI自动提取失败，仅保留文件名）",
                "Year": 2024,
                "Author": "Unknown"
            }
            
            try:
                # 读取文本
                with archive.open(f_name) as f:
                    if file_type == 'zip':
                        pdf_bytes = io.BytesIO(f.read())
                    else:
                        pdf_bytes = io.BytesIO(f.read())
                        
                    reader = PdfReader(pdf_bytes)
                    # 尝试读取前2页，如果读不到也别报错
                    text = ""
                    for page in reader.pages[:2]:
                        extracted = page.extract_text()
                        if extracted: text += extracted
                    
                    if len(text) < 50:
                        raise ValueError("PDF 文字太少或无法识别")
                    
                    # AI 提取
                    ai_info = extract_pdf_info_with_ai(client, model_name, text, f_name)
                    data_list.append(ai_info) # 成功！
                    
            except Exception as e:
                # 🌟 核心修复：如果出错了，不要跳过！把兜底信息加进去！
                # print(f"Error: {e}") 
                data_list.append(fallback_info)
            
            progress.progress((i+1)/len(pdf_files))
            
        return pd.DataFrame(data_list), None

    except rarfile.RarCannotExec:
        return None, "服务器缺少 unrar 工具，请检查 packages.txt。"
    except Exception as e:
        return None, f"解压失败: {str(e)}"

def generate_section_rag(client, model_name, sec_name, instruct, context_df):
    ctx_str = "".join([f"[ID:{r['ID']}] {r['Title']}\n摘要:{r['Abstract'][:200]}...\n\n" for _,r in context_df.iterrows()])
    sys = "你是一位严谨的学术综述专家。必须客观，引用需在句尾标注[ID]。"
    user = f"请撰写 **'{sec_name}'**。\n要求:{instruct}\n资料:\n{ctx_str}"
    try: return client.chat.completions.create(model=model_name, messages=[{"role":"system","content":sys},{"role":"user","content":user}]).choices[0].message.content
    except Exception as e: return f"Error: {e}"

def create_word_docx(text):
    doc = Document(); doc.add_heading('AI 综述 (RAG版)', 0)
    for line in text.split('\n'):
        if line.startswith('## '): doc.add_heading(line[3:], 1)
        elif line.startswith('### '): doc.add_heading(line[4:], 2)
        else: doc.add_paragraph(line)
    b = io.BytesIO(); doc.save(b); b.seek(0); return b

# --- 主逻辑 ---
client = None
if api_key: client = openai.OpenAI(api_key=api_key, base_url=base_url)

df = None
if input_mode == "直接上传 CSV 表格":
    f = st.file_uploader("上传 CSV", type=["csv"])
    if f: df = pd.read_csv(f)
else:
    z = st.file_uploader("上传压缩包", type=["zip", "rar"])
    if z and st.button("开始解析压缩包"):
        df, err = parse_compressed_files(z, client, chat_model)
        if err: st.error(err)

if df is not None and client:
    if 'ID' not in df.columns: df['ID'] = range(1, len(df)+1)
    df.fillna("Unknown", inplace=True)
    
    st.divider()
    # 🌟 显示当前文献数量
    st.subheader(f"📊 已成功加载 {len(df)} 篇文献")
    
    if len(df) == 0:
        st.error("没有提取到任何文献，请检查压缩包是否包含 PDF。")
    else:
        st.dataframe(df.head(3))
        
        # 动态调整 RAG 提示
        actual_k = min(top_k, len(df))
        if len(df) < top_k:
            st.info(f"💡 提示：上传文献数 ({len(df)}) 少于 RAG 设置 ({top_k})，将使用全部文献进行生成。")
    
        if st.button("🚀 开始 RAG 写作"):
            progress = st.progress(0); status = st.empty(); full_review = ""
            sections = [
                ("1. 研究背景", "history background", "梳理发展脉络。"),
                ("2. 核心方法", "methodology approach", "对比主流技术路线。"),
                ("3. 实验结果", "experiment result", "列举关键性能指标。"),
                ("4. 总结与展望", "conclusion future", "分析局限与未来。")
            ]
            for i, (t, k, ins) in enumerate(sections):
                status.text(f"撰写: {t} ..."); rel_df = retrieve_documents(k, df, None, False, top_k)
                full_review += f"## {t}\n\n{generate_section_rag(client, chat_model, t, ins, rel_df)}\n\n"
                progress.progress((i+1)/len(sections))
            
            full_review += "---\n## 参考文献\n" + "\n".join([f"[{r['ID']}] {r['Title']}." for _,r in df.iterrows()])
            st.download_button("下载 Word", create_word_docx(full_review), "review.docx")

elif not api_key: st.warning("请配置 Secrets 密钥")
