import streamlit as st
import pandas as pd
import openai
from datetime import datetime
import io
import zipfile
import json
import re
import numpy as np
from docx import Document
from pypdf import PdfReader

# --- 页面配置 ---
st.set_page_config(page_title="双引擎文献综述", layout="wide")

st.title("🚀 双引擎 AI 综述生成器 (防重置版)")
st.markdown("""
**注意**：由于服务器限制，**请使用 ZIP 格式**上传压缩包。
""")

# --- 初始化记忆 (Session State) ---
# 这是解决“闪退”的关键：如果内存里没有 df，先创建一个空的
if 'df' not in st.session_state:
    st.session_state.df = None

# --- 侧边栏 ---
with st.sidebar:
    st.header("1. 阅读引擎 (Kimi)")
    default_kimi = st.secrets.get("MOONSHOT_API_KEY", "")
    kimi_key = st.text_input("Kimi API Key", value=default_kimi, type="password")
    kimi_base = st.text_input("Kimi Base URL", value="https://api.moonshot.cn/v1")
    kimi_model = st.text_input("Kimi 模型名", value="moonshot-v1-8k")

    st.divider()

    st.header("2. 写作引擎 (DeepSeek)")
    default_ds = st.secrets.get("DEEPSEEK_API_KEY", "")
    ds_key = st.text_input("DeepSeek API Key", value=default_ds, type="password")
    ds_base = st.text_input("DeepSeek Base URL", value="https://api.deepseek.com")
    ds_model = st.text_input("DeepSeek 模型名", value="deepseek-chat")
    
    st.divider()
    st.header("3. 设置")
    top_k = st.slider("每章参考数量", 1, 50, 5)
    input_mode = st.radio("选择方式", ["直接上传 CSV", "上传 ZIP 压缩包"])

# --- 核心逻辑 ---

def get_client(api_key, base_url):
    if not api_key: return None
    return openai.OpenAI(api_key=api_key, base_url=base_url)

def extract_pdf_info_with_kimi(client, model, pdf_text, filename):
    prompt = f"""
    请从以下论文片段提取JSON: Title, Abstract, Year (int), Author, Journal。
    如果不确定，请尽力提取。直接返回JSON。
    片段:
    {pdf_text[:8000]}
    """
    try:
        res = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        content = res.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match: return json.loads(match.group(0))
        try: return json.loads(content)
        except: return {"Title": filename, "Abstract": f"【非结构化】{content[:300]}", "Year": 2024, "Author": "Unknown"}
    except Exception as e: raise ValueError(f"API错误: {e}")

def parse_zip_files(uploaded_file, client, model):
    data_list = []
    try:
        archive = zipfile.ZipFile(uploaded_file, 'r')
        pdf_files = [f for f in archive.namelist() if f.lower().endswith('.pdf')]
        
        if not pdf_files: return None, "ZIP包里没有找到PDF"

        progress = st.progress(0); status = st.empty()
        
        for i, f_name in enumerate(pdf_files):
            status.text(f"Kimi 正在分析: {i+1}/{len(pdf_files)} - {f_name}")
            try:
                with archive.open(f_name) as f:
                    bytes_io = io.BytesIO(f.read())
                    reader = PdfReader(bytes_io)
                    text = "".join([p.extract_text() for p in reader.pages[:3]])
                    if len(text.strip()) < 20: raise ValueError("无法读取文字(可能是扫描件)")
                    info = extract_pdf_info_with_kimi(client, model, text, f_name)
                    data_list.append(info)
            except Exception as e:
                data_list.append({"Title": f_name, "Abstract": f"❌ {str(e)}", "Year": 2024, "Author": "Unknown"})
            progress.progress((i+1)/len(pdf_files))
            
        return pd.DataFrame(data_list), None
    except Exception as e: return None, str(e)

def retrieve_documents(query, df, top_k):
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
        scores.append(s)
    df['score'] = scores
    return df.sort_values(by='score', ascending=False).head(actual_k)

def generate_section_deepseek(client, model, sec_name, instruct, context_df):
    ctx = "".join([f"[ID:{r['ID']}] {r['Title']}\n摘要:{r['Abstract'][:300]}\n\n" for _,r in context_df.iterrows()])
    sys = "你是一位严谨的学术综述专家。"
    user = f"请撰写 **'{sec_name}'**。\n要求:{instruct}\n资料:\n{ctx}"
    try: return client.chat.completions.create(model=model, messages=[{"role":"system","content":sys},{"role":"user","content":user}]).choices[0].message.content
    except Exception as e: return f"Error: {e}"

def create_word_docx(text):
    doc = Document(); doc.add_heading('AI 综述', 0)
    for line in text.split('\n'):
        if line.startswith('## '): doc.add_heading(line[3:], 1)
        elif line.startswith('### '): doc.add_heading(line[4:], 2)
        else: doc.add_paragraph(line)
    b = io.BytesIO(); doc.save(b); b.seek(0); return b

# --- 主程序 ---
client_kimi = get_client(kimi_key, kimi_base)
client_ds = get_client(ds_key, ds_base)

# 1. 解析逻辑
if input_mode == "直接上传 CSV":
    f = st.file_uploader("上传 CSV", type=["csv"])
    if f: 
        st.session_state.df = pd.read_csv(f) # 存入记忆
else:
    z = st.file_uploader("上传 ZIP 压缩包", type=["zip"])
    # 只有当点击解析按钮时，才进行繁重的解析工作
    if z and st.button("开始解析 (调用 Kimi)"):
        if not client_kimi: st.error("请填入 Kimi API Key")
        else:
            df_result, err = parse_zip_files(z, client_kimi, kimi_model)
            if err: st.error(err)
            else:
                st.session_state.df = df_result # 关键：解析成功后，存入记忆！

# 2. 写作逻辑 (只要记忆里有数据，就显示)
if st.session_state.df is not None:
    df = st.session_state.df
    if 'ID' not in df.columns: df['ID'] = range(1, len(df)+1)
    df.fillna("Unknown", inplace=True)
    
    st.divider()
    st.subheader(f"📊 已加载 {len(df)} 篇文献")
    st.dataframe(df.head(3))
    
    # 这里的按钮点击后，虽然页面刷新，但 st.session_state.df 还在，所以不会闪退
    if len(df) > 0 and st.button("🚀 开始写作 (调用 DeepSeek)"):
        if not client_ds: st.error("请填入 DeepSeek API Key")
        else:
            progress = st.progress(0); status = st.empty(); full_review = ""
            sections = [("1. 研究背景", "background", "梳理脉络"), ("2. 核心方法", "methodology", "对比技术"), ("3. 实验结果", "result", "列举数据"), ("4. 总结", "conclusion", "分析未来")]
            for i, (t, k, ins) in enumerate(sections):
                status.text(f"撰写: {t} ..."); rel = retrieve_documents(k, df, top_k)
                full_review += f"## {t}\n\n{generate_section_deepseek(client_ds, ds_model, t, ins, rel)}\n\n"
                progress.progress((i+1)/len(sections))
            st.download_button("下载 Word", create_word_docx(full_review), "review.docx")
