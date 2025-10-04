# ===============================
# MVP 魔法鏡牆 - 完整 PDF 下載版 (整合排版建議)
# ===============================


import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fpdf import FPDF
import os
import time
import tempfile

    
# --------------------------


st.set_page_config(page_title="🕹️ Mario 互動魔法鏡", layout="wide")
st.title("🕹️ Mario 互動魔法鏡 — 旅遊評論儀表板")
st.write(
    "歡迎來到 Mario 互動魔法鏡，這裡以尼泊爾旅遊景點原始評論資料經過數據探勘、視覺化處理後的圖像作為 Demo 示範，"
    "請透過側邊欄選擇景點，即時看到情緒地圖分布與關鍵字。資料來源：Kaggle - Tourist Review Sentiment Analysis"
)

# --------------------------
# 載入資料
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("merged_data_place.csv")
    if df['review_tokens'].dtype == 'O':
        df['review_tokens'] = df['review_tokens'].apply(eval)
    return df

df = load_data()

# --------------------------
# 側邊欄篩選
# --------------------------
places = df['place'].unique().tolist()
selected_place = st.sidebar.multiselect("各點綜覽", options=places, default=places[:10])

# 過濾資料
df_filtered = df[df['place'].isin(selected_place)]

# --------------------------
# 計算情緒分布與關鍵字
# --------------------------
emotion_counts = df_filtered.groupby(['place', 'sentiment']).size().unstack(fill_value=0)
emotion_counts['total_reviews'] = emotion_counts.sum(axis=1)
emotion_counts.reset_index(inplace=True)

place_info = df_filtered[['place','lat','lng']].drop_duplicates(subset=['place'])
emotion_map = pd.merge(emotion_counts, place_info, on='place', how='left')

top_keywords = {}
for place, group_place in df_filtered.groupby("place"):
    top_keywords[place] = {}
    for sentiment, group_sent in group_place.groupby("sentiment"):
        tokens = sum(group_sent['review_tokens'], [])
        count = Counter(tokens)
        top = [word for word, freq in count.most_common(5)]
        top_keywords[place][sentiment] = ", ".join(top)

keywords_df = pd.DataFrame([
    {"place": place, "sentiment": sentiment, "keywords": kws}
    for place, s_dict in top_keywords.items()
    for sentiment, kws in s_dict.items()
])
keyword_pivot = keywords_df.pivot(index='place', columns='sentiment', values='keywords').reset_index()
emotion_map = pd.merge(emotion_map, keyword_pivot, on='place', how='left')

emotion_map = emotion_map.rename(columns={
    'positive_x':'positive',
    'neutral_x':'neutral',
    'negative_x':'negative',
    'positive_y':'positive_keywords',
    'neutral_y':'neutral_keywords',
    'negative_y':'negative_keywords'
})
emotion_map['positive_ratio'] = emotion_map.get('positive', 0) / emotion_map['total_reviews']

# --------------------------
# 智慧摘要函式
# --------------------------
def generate_recommendation(df, place):
    places = place if isinstance(place, list) else [place]
    results = {}
    for p in places:
        subset = df[df['place'] == p]
        if subset.empty:
            results[p] = f"{p}：無資料可用。"
            continue
        pos_ratio = (subset['sentiment'] == 'positive').mean()
        kw_candidates = None
        for col in ['keywords_cn','keywords','review_tokens']:
            if col in subset.columns:
                exploded = subset[col].explode().dropna()
                if not exploded.empty:
                    flat = []
                    for v in exploded:
                        if isinstance(v,list):
                            flat.extend(v)
                        else:
                            flat.append(v)
                    if len(flat)>0:
                        from collections import Counter
                        topk = [x for x,_ in Counter(flat).most_common(3)]
                        kw_candidates = topk
                        break
        if not kw_candidates:
            kw_candidates = []
        if pos_ratio > 0.91:
            mood = "強烈推薦 👍"
        elif pos_ratio > 0.88:
            mood = "值得一遊 😉"
        elif pos_ratio > 0.85:
            mood = "可安排短暫造訪（視偏好）🤏"
        else:
            mood = "口碑普通，建議斟酌或查更多資訊 🤔"
        kw_str = ", ".join(kw_candidates) if kw_candidates else "無顯著關鍵字"
        results[p] = f"評論正向比例為 {pos_ratio:.0%}；熱門關鍵字：{kw_str}。建議：{mood}"
    return results[places[0]] if isinstance(place,str) else results

# ===========================
# 使用 Tabs 區分 Overview 與 Detail
# ===========================
tab_overview, tab_detail = st.tabs([" 👉  多景點綜覽", " 👉  單景點詳情"])

# --------------------------
# Overview - 多選景點
# --------------------------
with tab_overview:
    st.write("#### 🗺️ 多景點綜覽")
    st.write("統計摘要：")
    positive_reviews = df_filtered[df_filtered['sentiment']=='positive'].shape[0]
    st.write(
    f"選擇景點 {df_filtered['place'].nunique()} 筆，"
    f"合計評論 {df_filtered.shape[0]} 筆，"
    f"好評比例: {positive_reviews / df_filtered.shape[0]:.2%}"
    )
    
    st.write("##### 📊 情緒統計表")
    st.dataframe(emotion_map[[
        'place','total_reviews','positive','neutral','negative',
        'positive_keywords','neutral_keywords','negative_keywords'
    ]].sort_values('total_reviews', ascending=False))
    
    st.write("##### 🌍 情緒氣泡圖")
    fig_map = px.scatter_mapbox(
        emotion_map,
        lat="lat",
        lon="lng",
        size="total_reviews",
        color="positive_ratio",
        hover_name="place",
        hover_data=[
            "positive","neutral","negative",
            "positive_keywords","neutral_keywords","negative_keywords"
        ],
        color_continuous_scale=px.colors.diverging.RdYlGn,
        size_max=40,
        zoom=5,
        mapbox_style="carto-positron"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # 多選景點智慧摘要
    if selected_place:
        suggestion = generate_recommendation(df, selected_place)
        st.write("##### 🎯 智慧摘要")
        if isinstance(suggestion, dict):
            for place, text in suggestion.items():
                st.markdown(f"**{place}**: {text}")
        else:
            st.markdown(suggestion)

# --------------------------
# Detail - 單景點深度分析
# --------------------------
with tab_detail:
    st.write("#### 🗺️ 單景點詳情")
    selected_detail_place = st.selectbox("選擇景點查看詳細資訊", df_filtered['place'].unique())

    # --------------------------
    # 上半部：三張圖水平排列
    # --------------------------
    col1, col2, col3 = st.columns([1,1,1])
    CHART_HEIGHT = 400  # 統一高度

    # 📡 特色雷達圖
    with col1:
        st.write("##### 📡 特色雷達圖")
        aspects = {
            "自然景觀": ["lake","lakeside","boating","view","pokhara","annapurna","everest"],
            "宗教文化": ["temple","buddha","lord","shiva","pashupatinath","gautam","stupa","heritage"],
            "歷史建築": ["square","durbar","historical","bhaktapur","kathmandu","valley"],
            "野生動物與自然公園": ["park","national","safari","animals","chitwan","bardiya","jungle"],
            "戶外探險": ["trekking","trek","camp","experience","base","langtang","icefall","ebc"]
        }

        def aspect_scores(subset):
            scores = {}
            for aspect, kws in aspects.items():
                mask = subset["review_tokens"].apply(lambda tokens: any(kw in tokens for kw in kws))
                scores[aspect] = (subset[mask]["sentiment"]=="positive").mean() if mask.sum()>0 else 0
            return scores

        subset = df[df["place"]==selected_detail_place]
        scores = aspect_scores(subset)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=list(scores.values()), theta=list(scores.keys()),
                                            fill='toself', name=selected_detail_place))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False,
        height=CHART_HEIGHT)
        st.plotly_chart(fig_radar, use_container_width=True)

    # 📖 熱門關鍵字
    with col2:
        st.write("##### 📖 熱門關鍵字")
        tokens = [token for token in df[df['place']==selected_detail_place]['review_tokens'].sum()]
        counter = Counter(tokens)
        top_words = counter.most_common(10)
        words, counts = zip(*top_words)

        fig_keywords = px.bar(
            x=counts,
            y=words,
            orientation='h',
            text=counts,
            labels={'x':'counts','y':'熱門關鍵字'},
            title=f"{selected_detail_place} - Top 10 Keywords",
            color=counts,
            color_continuous_scale=px.colors.diverging.RdYlGn,
            height=CHART_HEIGHT
        )
        fig_keywords.update_layout(yaxis={'categoryorder':'total ascending'},
                                coloraxis_colorbar=dict(title='counts'))
        st.plotly_chart(fig_keywords, use_container_width=True)

    # ☁️ 關鍵文字雲
    with col3:
        st.write("##### ☁️ 關鍵文字雲")
        wordcloud = WordCloud(width=400, height=400, background_color='white').generate(" ".join(tokens))
        fig_wc, ax_wc = plt.subplots(figsize=(6, CHART_HEIGHT/100*6))  # 轉成 inches
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # --------------------------
    # 下半部：智慧摘要 + PDF下載
    # --------------------------
    st.write("##### 🎯 智慧摘要")
    suggestion = generate_recommendation(df, selected_detail_place)
    st.markdown(suggestion)

    def generate_pdf(fig_radar, fig_keywords, tokens, fig_map, suggestion, selected_detail_place):
        import tempfile, os, time
        from fpdf import FPDF
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        from plotly.io import to_image

        with tempfile.TemporaryDirectory() as tmpdir:
            # ------------------ 儲存圖表 ------------------
            # radar_path = os.path.join(tmpdir, "radar.png")
            # fig_radar.write_image(radar_path)
            radar_path = os.path.join(tmpdir, "radar.png")
            save_plotly_figure(fig_radar, radar_path)

            # bar_path = os.path.join(tmpdir, "bar.png")
            # fig_keywords.write_image(bar_path)
            bar_path = os.path.join(tmpdir, "bar.png")
            save_plotly_figure(fig_keywords, bar_path)

            # map_path = os.path.join(tmpdir, "map.png")
            # fig_map.write_image(map_path)
            map_path = os.path.join(tmpdir, "map.png")
            save_plotly_figure(fig_map, map_path)


            # 文字雲
            wc_path = os.path.join(tmpdir, "wc.png")
            wordcloud = WordCloud(width=400, height=400, background_color='white').generate(" ".join(tokens))
            fig_wc, ax_wc = plt.subplots(figsize=(6, 6))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            fig_wc.savefig(wc_path, bbox_inches="tight")
            plt.close(fig_wc)  # ✅ 釋放檔案

            # ------------------ 建立 PDF ------------------
            pdf = FPDF(orientation='L', format='A4')
            pdf.add_font('NotoSans', '', r'NotoSansTC-Regular.otf', uni=True)
            pdf.add_page()

            # 標題
            pdf.set_font("NotoSans", size=15)
            pdf.multi_cell(0, 10, f"★ Mario 互動魔法鏡：一頁式旅遊評論快照報告 ({selected_detail_place})", align="C")
            pdf.ln(5)

            # 智慧摘要
            pdf.set_font("NotoSans", size=12)
            pdf.multi_cell(0, 8, f"智慧摘要：{suggestion}")
            pdf.ln(5)

            # ---------- 四圖 2x2（指定高度，高度固定，寬度自動） ----------
            img_h = 70  # mm，高度固定
            margin_x, start_y = 15, pdf.get_y() + 5
            gap_x, gap_y = 15, 12

            pdf.set_font("NotoSans", size=11)

            # 上排
            for i, (title, path) in enumerate([("★ 特色雷達圖", radar_path), ("★ 熱門關鍵字", bar_path)]):
                with Image.open(path) as img:
                    target_w = img.width / img.height * img_h  # 寬度按比例
                x = margin_x + i * (target_w + gap_x)
                pdf.set_xy(x, start_y - 6)  # 標題稍上方
                pdf.multi_cell(target_w, 6, title)
                pdf.image(path, x=x, y=start_y, w=target_w, h=img_h)

            # 下排
            second_row_y = start_y + img_h + gap_y
            for i, (title, path) in enumerate([("★ 關鍵文字雲", wc_path), ("★ 情緒氣泡圖", map_path)]):
                with Image.open(path) as img:
                    target_w = img.width / img.height * img_h
                x = margin_x + i * (target_w + gap_x)
                pdf.set_xy(x, second_row_y - 6)
                pdf.multi_cell(target_w, 6, title)
                pdf.image(path, x=x, y=second_row_y, w=target_w, h=img_h)

            # ✅ 將 PDF 輸出到記憶體，避免臨時檔案權限問題
            pdf_bytes = io.BytesIO()
            pdf.output(pdf_bytes)
            pdf_bytes.seek(0)
            return pdf_bytes.read()


    # ------------------ Streamlit 按鈕 ------------------
    if st.button("📑 下載 PDF"):
        pdf_bytes = generate_pdf(
            fig_radar=fig_radar,
            fig_keywords=fig_keywords,
            tokens=tokens,                  
            fig_map=fig_map,
            suggestion=suggestion,          
            selected_detail_place=selected_detail_place
        )

        st.download_button(
            "點此下載完整 PDF 報告",
            data=pdf_bytes,
            file_name="report.pdf",
            mime="application/pdf"
        )


