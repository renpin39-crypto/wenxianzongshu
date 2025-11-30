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
st.set_page_config(page_title="双引擎文献综述(调试版)", layout="wide")

st.title("🛠️ 双引擎综述生成器 (含错误诊断)")
st.markdown("""
**调试模式已开启**：
如果出现解析失败，摘要栏会显示具体的**错误原因**，而不是简单的“失败”。
同时增强了对非 JSON 格式返回的兼容性。
""")

# --- 侧边栏：双模型配置 ---
with st.sidebar:
    st.header("1. 阅读引擎 (Kimi)")
    # 尝试读取 Kimi Secrets
    default_kimi = st.secrets.get("MOONSHOT_API_KEY", "")
    kimi_key = st.text_input("Kimi API Key", value=default_kimi, type="password", key="k_key")
    kimi_base = st.text_input("Kimi Base URL", value="https://api.moonshot.cn/v1", key="k_base")
    kimi_model = st.text_input("Kimi 模型名", value="moonshot-v1-8k", key="k_model")

    st.divider()

    st.header("2. 写作引擎 (DeepSeek)")
    default_ds = st.secrets.get("DEEPSEEK_API_KEY", "")
    ds_key = st.text_input("DeepSeek API Key", value=default_ds, type="password", key="d_key")
    ds_base = st.text_input("DeepSeek Base URL", value="https://api.deepseek.com", key="d_base")
    ds_model = st.text_input("DeepSeek 模型名", value="deepseek-chat", key="d_model")
    
    st.divider()
    st.header("3. 设置")
    top_k = st.slider("每章参考数量", 1, 50, 5)
    input_mode = st.radio("选择方式", ["直接上传 CSV", "上传压缩包 (ZIP/RAR)"])

# --- 核心逻辑 ---

def get_client(api_key, base_url):
    if not api_key: return None
    return openai.OpenAI(api_key=api_key, base_url=base_url)

def extract_pdf_info_with_kimi(client, model, pdf_text, filename):
    """Kimi 提取逻辑 (增强容错)"""
    # 提示词优化：明确告诉它可能是中文
    prompt = f"""
    你是一个数据提取助手。请阅读以下论文片段（可能包含中文或英文）。
    
    【任务】
    提取以下字段并返回 JSON 格式：
    - Title: 论文标题 (如果找不到，用文件名 "{filename}")
    - Abstract: 摘要 (如果找不到摘要，请总结前两页内容，300字以内)
    - Year: 发表年份 (int类型, 找不到填2024)
    - Author: 第一作者 (找不到填 Unknown)
    
    【重要】
    请直接返回 JSON 数据，不要包含 ```json 或其他废话。
    如果不确定，请尽力提取，不要报错。
    
    【论文片段】:
    {pdf_text[:8000]}
    """
    try:
        res = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        content = res.choices[0].message.content.strip()
        
        # 1. 尝试正则提取 JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        
        # 2. 尝试直接解析
        try:
            return json.loads(content)
        except:
            # 3. 🌟 最后的挽救：如果不是 JSON，直接把 Kimi 的回复当成摘要！
            # 这样至少不会报错，内容还在
            return {
                "Title": filename,
                "Abstract": f"【非结构化提取】{content[:300]}...", # 保留它的回复
                "Year": 2024,
                "Author": "Unknown"
            }
            
    except Exception as e:
        raise ValueError(f"API调用错误: {str(e)}")

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
            status.text(f"Kimi 正在分析: {i+1}/{len(pdf_files)} - {f_name}")
            
            # 默认错误信息
            err_msg = "未知错误"
            
            try:
                with archive.open(f_name) as f:
                    bytes_io = io.BytesIO(f.read())
                    reader = PdfReader(bytes_io)
                    
                    # 尝试读取文本
                    text = ""
                    for page in reader.pages[:3]:
                        extracted = page.extract_text()
                        if extracted: text += extracted
                    
                    # 🌟 诊断1：PDF 是否为空（扫描件）
                    if len(text.strip()) < 20: 
                        err_msg = "PDF为扫描件或纯图片，无法读取文字"
                        raise ValueError(err_msg)
                    
                    # 调用 API
                    info = extract_pdf_info_with_kimi(client, model, text, f_name)
                    data_list.append(info)
                    
            except Exception as e:
                # 🌟 诊断2：捕获具体错误并显示在表格里
                clean_err = str(e).replace("ValueError: ", "")
                data_list.append({
                    "Title": f_name, 
                    "Abstract": f"❌ 解析失败: {clean_err}", 
                    "Year": 2024, 
                    "Author": "Unknown"
                })
            
            progress.progress((i+1)/len(pdf_files))
            
        return pd.DataFrame(data_list), None
    except rarfile.RarCannotExec: return None, "服务器缺少 unrar"
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

df = None
if input_mode == "直接上传 CSV":
    f = st.file_uploader("上传 CSV", type=["csv"])
    if f: df = pd.read_csv(f)
else:
    z = st.file_uploader("上传 ZIP/RAR", type=["zip", "rar"])
    if z and st.button("开始解析 (调用 Kimi)"):
        if not client_kimi: st.error("请填入 Kimi API Key")
        else:
            df, err = parse_compressed_files(z, client_kimi, kimi_model)
            if err: st.error(err)

if df is not None:
    if 'ID' not in df.columns: df['ID'] = range(1, len(df)+1)
    df.fillna("Unknown", inplace=True)
    st.divider(); st.subheader(f"📊 已加载 {len(df)} 篇文献"); st.dataframe(df)
    
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
