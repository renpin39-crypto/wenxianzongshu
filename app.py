import streamlit as st
import pandas as pd
import openai
from datetime import datetime
import io
import zipfile
import rarfile
import json
import re
import numpy as np
from docx import Document
from pypdf import PdfReader

# --- 页面配置 ---
st.set_page_config(page_title="双引擎文献综述", layout="wide")

st.title("🚀 双引擎 AI 综述生成器 (Kimi读 + DeepSeek写)")
st.markdown("""
**核心架构**：
1. **阅读引擎 (Kimi)**：利用长窗口优势，精准解析 PDF 提取摘要。
2. **写作引擎 (DeepSeek)**：利用强推理能力，基于 RAG 逻辑撰写综述。
""")

# --- 侧边栏：双模型配置 ---
with st.sidebar:
    st.header("1. 阅读引擎 (解析PDF)")
    st.caption("推荐使用 Kimi (Moonshot)")
    
    # 尝试读取 Kimi Secrets
    default_kimi = st.secrets.get("MOONSHOT_API_KEY", "")
    kimi_key = st.text_input("Kimi API Key", value=default_kimi, type="password", key="k_key")
    kimi_base = st.text_input("Kimi Base URL", value="https://api.moonshot.cn/v1", key="k_base")
    # Kimi 模型通常用 moonshot-v1-8k 或 moonshot-v1-32k
    kimi_model = st.text_input("Kimi 模型名", value="moonshot-v1-8k", key="k_model")

    st.divider()

    st.header("2. 写作引擎 (生成正文)")
    st.caption("推荐使用 DeepSeek")
    
    # 尝试读取 DeepSeek Secrets
    default_ds = st.secrets.get("DEEPSEEK_API_KEY", "")
    ds_key = st.text_input("DeepSeek API Key", value=default_ds, type="password", key="d_key")
    ds_base = st.text_input("DeepSeek Base URL", value="https://api.deepseek.com", key="d_base")
    ds_model = st.text_input("DeepSeek 模型名", value="deepseek-chat", key="d_model")
    
    st.divider()
    
    st.header("3. RAG 设置")
    top_k = st.slider("每章参考数量", 1, 50, 5)
    
    st.header("4. 数据输入")
    input_mode = st.radio("选择方式", ["直接上传 CSV", "上传压缩包 (ZIP/RAR)"])

# --- 核心逻辑 ---

def get_client(api_key, base_url):
    if not api_key: return None
    return openai.OpenAI(api_key=api_key, base_url=base_url)

def extract_pdf_info_with_kimi(client, model, pdf_text, filename):
    """专门用 Kimi 提取信息"""
    prompt = f"""
    你是一个专业的数据提取助手。请从以下论文片段提取JSON数据。
    字段: Title, Abstract, Year (int), Author, Journal。
    如果不确定年份，填2024。
    
    请直接返回JSON格式，不要包含Markdown代码块。
    片段:
    {pdf_text[:10000]} 
    """ # Kimi 可以处理更长的文本，这里放宽到 10000 字符
    try:
        res = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        content = res.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match: return json.loads(match.group(0))
        else: return json.loads(content)
    except Exception as e:
        raise ValueError(f"Kimi解析失败: {e}")

def parse_compressed_files(uploaded_file, client, model):
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
        else: return None, "不支持的格式"

        pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]
        if not pdf_files: return None, "没有找到 PDF"

        progress = st.progress(0); status = st.empty()
        
        for i, f_name in enumerate(pdf_files):
            status.text(f"Kimi 正在阅读: {i+1}/{len(pdf_files)} - {f_name}")
            fallback = {"Title": f_name, "Abstract": "解析失败", "Year": 2024, "Author": "Unknown"}
            
            try:
                with archive.open(f_name) as f:
                    bytes_io = io.BytesIO(f.read())
                    reader = PdfReader(bytes_io)
                    text = "".join([p.extract_text() for p in reader.pages[:3]]) # Kimi可以多读一页
                    
                    if len(text) < 50: raise ValueError("文字过少")
                    
                    # 使用 Kimi 客户端
                    info = extract_pdf_info_with_kimi(client, model, text, f_name)
                    data_list.append(info)
            except Exception:
                data_list.append(fallback)
            
            progress.progress((i+1)/len(pdf_files))
            
        return pd.DataFrame(data_list), None
    except rarfile.RarCannotExec: return None, "缺少 unrar 工具"
    except Exception as e: return None, str(e)

def retrieve_documents(query, df, top_k):
    """关键词检索逻辑"""
    actual_k = min(top_k, len(df))
    scores = []
    q_words = query.lower().split()
    for _, row in df.iterrows():
        s = 0
        txt = (str(row['Title']) + " " + str(row['Abstract'])).lower()
        try: s += max(0, int(row['Year']) - 2020) * 2
        except: pass
        for w in q_words: 
            if w in txt: s += txt.count(w)
        if "背景" in query and int(row.get('Year', 2024)) < 2022: s += 20
        scores.append(s)
    df['score'] = scores
    return df.sort_values(by='score', ascending=False).head(actual_k)

def generate_section_deepseek(client, model, sec_name, instruct, context_df):
    """专门用 DeepSeek 写作"""
    ctx = "".join([f"[ID:{r['ID']}] {r['Title']}\n摘要:{r['Abstract'][:300]}\n\n" for _,r in context_df.iterrows()])
    sys = "你是一位严谨的学术综述专家。必须客观，引用需在句尾标注[ID]。"
    user = f"请撰写 **'{sec_name}'**。\n要求:{instruct}\n资料:\n{ctx}"
    try: return client.chat.completions.create(model=model, messages=[{"role":"system","content":sys},{"role":"user","content":user}]).choices[0].message.content
    except Exception as e: return f"Error: {e}"

def create_word_docx(text):
    doc = Document(); doc.add_heading('AI 综述 (双引擎版)', 0)
    for line in text.split('\n'):
        if line.startswith('## '): doc.add_heading(line[3:], 1)
        elif line.startswith('### '): doc.add_heading(line[4:], 2)
        else: doc.add_paragraph(line)
    b = io.BytesIO(); doc.save(b); b.seek(0); return b

# --- 主程序 ---

# 初始化两个客户端
client_kimi = get_client(kimi_key, kimi_base)
client_ds = get_client(ds_key, ds_base)

df = None

# 1. 解析阶段 (用 Kimi)
if input_mode == "直接上传 CSV":
    f = st.file_uploader("上传 CSV", type=["csv"])
    if f: df = pd.read_csv(f)
else:
    z = st.file_uploader("上传 ZIP/RAR", type=["zip", "rar"])
    if z and st.button("开始解析 (调用 Kimi)"):
        if not client_kimi:
            st.error("❌ 请先配置 Kimi API Key")
        else:
            df, err = parse_compressed_files(z, client_kimi, kimi_model)
            if err: st.error(err)

# 2. 写作阶段 (用 DeepSeek)
if df is not None:
    if 'ID' not in df.columns: df['ID'] = range(1, len(df)+1)
    df.fillna("Unknown", inplace=True)
    
    st.divider()
    st.subheader(f"📊 已加载 {len(df)} 篇文献")
    st.dataframe(df.head(3))
    
    if len(df) > 0:
        if st.button("🚀 开始写作 (调用 DeepSeek)"):
            if not client_ds:
                st.error("❌ 请先配置 DeepSeek API Key")
            else:
                progress = st.progress(0); status = st.empty(); full_review = ""
                sections = [
                    ("1. 研究背景", "history background", "梳理脉络。"),
                    ("2. 核心方法", "methodology approach", "对比技术路线。"),
                    ("3. 实验结果", "experiment result", "列举数据。"),
                    ("4. 总结与展望", "conclusion future", "分析未来方向。")
                ]
                for i, (t, k, ins) in enumerate(sections):
                    status.text(f"DeepSeek 正在撰写: {t} ...")
                    rel_df = retrieve_documents(k, df, top_k)
                    content = generate_section_deepseek(client_ds, ds_model, t, ins, rel_df)
                    full_review += f"## {t}\n\n{content}\n\n"
                    progress.progress((i+1)/len(sections))
                
                full_review += "---\n## 参考文献\n" + "\n".join([f"[{r['ID']}] {r['Title']}." for _,r in df.iterrows()])
                st.download_button("下载 Word", create_word_docx(full_review), "review.docx")
