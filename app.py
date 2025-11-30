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

st.title("🎓 AI 综述生成器 (强力清洗 + 格式修正版)")
st.markdown("""
**本次修复**：
1. **清理乱码**：强力清除 `-`、`*`、`**` 等所有 Markdown 列表符号。
2. **找回参考文献**：强制将参考文献列表写入 Word 文档末尾。
3. **修正引用**：自动将 `(资料 1)` 统一修正为 `[1]`。
""")

# --- 初始化记忆 ---
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
                    if len(text.strip()) < 20: raise ValueError("无法读取文字")
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
    
    # 🌟 提示词再次强化：要求纯文本
    system_prompt = """
    你是一位严谨的学术综述作者。
    【重要格式要求】
    1. **纯文本段落**：严禁使用任何列表符号（如 -、*、1.）。所有观点必须用连贯的句子写在段落里。
    2. **去格式化**：严禁使用 markdown 加粗（**text**）或标题（###）。
    3. **引用格式**：必须严格使用数字引用 [1]、[2]，严禁写成 (资料 1) 或 [ID:1]。
    """
    
    user_prompt = f"""
    请撰写章节：**'{sec_name}'**。
    写作指引：{instruct}
    资料库：
    {ctx}
    """
    try: 
        return client.chat.completions.create(
            model=model, 
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=0.3
        ).choices[0].message.content
    except Exception as e: return f"Error: {e}"

def clean_text_content(text):
    """🔥 强力清洗函数：去除所有 Markdown 和 奇怪的引用"""
    # 1. 去除 Markdown 加粗 (**text** -> text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # 2. 去除行首的列表符号 (- , * , 1. )
    # 匹配规则：行首 + (减号或星号或数字点) + 空格
    text = re.sub(r'^\s*[\-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # 3. 修正引用格式
    # 把 (资料 1) 或 (ID:1) 或 [ID:1] 统一变成 [1]
    text = re.sub(r'[\[\(]?(?:资料|ID|Ref|Reference)[:\s]?(\d+)[\]\)]?', r'[\1]', text)
    
    return text

def create_word_docx(full_text_list, df_refs):
    """
    full_text_list: 正文列表 [(标题, 内容), (标题, 内容)...]
    df_refs: 参考文献 DataFrame (用于强制生成文末列表)
    """
    doc = Document()
    doc.add_heading('AI 文献综述 (纯净排版)', 0)
    
    # 1. 写入正文
    for title, content in full_text_list:
        doc.add_heading(title, level=1)
        
        # 清洗内容
        clean_content = clean_text_content(content)
        
        # 按行写入，避免一大坨
        for line in clean_content.split('\n'):
            line = line.strip()
            if line:
                # 剔除掉模型可能自己生成的标题行（以 # 开头的）
                if not line.startswith('#'):
                    doc.add_paragraph(line)
    
    # 2. 🔥 强制写入参考文献 (保证绝对不丢失)
    doc.add_page_break() # 另起一页
    doc.add_heading('参考文献', level=1)
    
    for _, r in df_refs.iterrows():
        # 格式：[1] 作者. 标题. 年份.
        ref_str = f"[{r['ID']}] {r['Author']}. {r['Title']}. {r['Year']}."
        doc.add_paragraph(ref_str)
            
    b = io.BytesIO(); doc.save(b); b.seek(0); return b

# --- 主程序 ---
client_kimi = get_client(kimi_key, kimi_base)
client_ds = get_client(ds_key, ds_base)

if input_mode == "直接上传 CSV":
    f = st.file_uploader("上传 CSV", type=["csv"])
    if f: st.session_state.df = pd.read_csv(f)
else:
    z = st.file_uploader("上传 ZIP 压缩包", type=["zip"])
    if z and st.button("开始解析 (调用 Kimi)"):
        if not client_kimi: st.error("请填入 Kimi API Key")
        else:
            df_result, err = parse_zip_files(z, client_kimi, kimi_model)
            if err: st.error(err)
            else: st.session_state.df = df_result

if st.session_state.df is not None:
    df = st.session_state.df
    if 'ID' not in df.columns: df['ID'] = range(1, len(df)+1)
    df.fillna("Unknown", inplace=True)
    
    st.divider()
    st.subheader(f"📊 已加载 {len(df)} 篇文献")
    st.dataframe(df)
    
    if len(df) > 0 and st.button("🚀 开始写作 (纯净模式)"):
        if not client_ds: st.error("请填入 DeepSeek API Key")
        else:
            progress = st.progress(0); status = st.empty()
            
            # 🌟 改动：用列表存储每一章，而不是拼字符串
            # 这样方便后面单独清洗每一章，且不会丢失数据
            generated_chapters = []
            
            sections = [
                ("1. 研究背景", "background", "以叙述的方式梳理研究脉络，严禁列点。"), 
                ("2. 核心方法对比", "methodology", "将不同研究的方法进行综合对比，写成连贯的段落。"), 
                ("3. 关键结果分析", "result", "归纳实验结论，避免简单罗列数据。"), 
                ("4. 总结与展望", "conclusion", "基于现状分析未来的局限性与方向。")
            ]
            
            for i, (t, k, ins) in enumerate(sections):
                status.text(f"DeepSeek 正在撰写: {t} ...")
                rel = retrieve_documents(k, df, top_k)
                content = generate_section_deepseek(client_ds, ds_model, t, ins, rel)
                generated_chapters.append((t, content)) # 存入列表
                progress.progress((i+1)/len(sections))
            
            # 调用新的 Word 生成函数，把 df 传进去专门生成参考文献
            docx_file = create_word_docx(generated_chapters, df)
            
            st.success("✅ 生成完成！")
            st.download_button("下载 Word (纯净版)", docx_file, "review_clean.docx")
