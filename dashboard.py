import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random

# ======================
# STREAMLIT CONFIGURATION
# ======================
st.set_page_config(
    layout="wide", 
    page_title="Emotion Intelligence Dashboard",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f2f5);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 
            20px 20px 60px #d9d9d9,
            -20px -20px 60px #ffffff;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .neon-text {
        color: #00ff88;
        text-shadow: 0 0 5px #00ff88, 0 0 10px #00ff88, 0 0 15px #00ff88;
        font-weight: 600;
    }
    
    .gradient-text {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .highlight {
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        padding: 3px 8px;
        border-radius: 5px;
        color: white;
        font-weight: 600;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    
    .stSlider > div > div > div {
        color: white;
    }
    
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Modern color palettes
VIBRANT_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
    '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
    '#EE5A24', '#009432', '#0652DD', '#9980FA', '#D63031'
]

GRADIENT_COLORS = [
    'rgb(255,107,107)', 'rgb(78,205,196)', 'rgb(69,183,209)', 
    'rgb(150,206,180)', 'rgb(254,202,87)', 'rgb(255,159,243)',
    'rgb(84,160,255)', 'rgb(95,39,205)', 'rgb(0,210,211)', 'rgb(255,159,67)'
]

# ======================
# MOCK DATA GENERATION
# ======================
@st.cache_data(show_spinner=False)
def generate_comprehensive_data(n=2000):
    """Generate comprehensive mock emotion data"""
    emotions = [
        'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 
        'neutral', 'love', 'excitement', 'confusion', 'gratitude', 
        'disappointment', 'hope', 'pride', 'anxiety'
    ]
    
    sample_texts = [
        "I absolutely love spending quality time with my family during weekends",
        "This endless traffic jam is making me incredibly frustrated and angry",
        "What an absolutely amazing and unexpected surprise party that was",
        "I'm genuinely worried and anxious about the upcoming important exam",
        "That comedy movie was absolutely hilarious and made my day",
        "I honestly can't believe this incredible thing happened to me today",
        "Today was just another ordinary and routine day at the office",
        "I'm so incredibly excited and thrilled for our vacation next week",
        "This restaurant's food tastes absolutely terrible and disappointing",
        "I'm completely confused and puzzled about the new company policy"
    ]
    
    data = []
    for i in range(n):
        emotion = random.choices(emotions, weights=[15,8,10,7,12,5,20,9,11,6,8,7,9,10,8])[0]
        text = random.choice(sample_texts) + f" (analysis #{i+1})"
        confidence = random.uniform(0.55, 0.98)
        timestamp = datetime.now() - timedelta(hours=random.randint(0, 168))  # Past week
        intensity = random.uniform(0.3, 1.0)
        sentiment_score = random.uniform(-1, 1)
        
        data.append({
            'text': text,
            'pred_top': emotion,
            'score': confidence,
            'timestamp': timestamp,
            'intensity': intensity,
            'sentiment': sentiment_score,
            'text_length': len(text)
        })
    
    return pd.DataFrame(data)

# ======================
# DASHBOARD HEADER
# ======================
st.markdown('<h1 class="gradient-text">🧠 Emotion Intelligence Dashboard</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <h3 style="color: #667eea; margin-top: 0;">🚀 Advanced Emotional Analytics Platform</h3>
    <p style="font-size: 16px; line-height: 1.6;">
        Dive deep into emotional patterns with our <span class="highlight">AI-powered</span> sentiment analysis engine. 
        This comprehensive dashboard provides <span class="highlight">real-time insights</span> into emotional trends, 
        patterns, and correlations using advanced natural language processing.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================
# DATA LOADING
# ======================
@st.cache_data(show_spinner=False)
def load_and_process_data():
    """Load and enrich the emotion data"""
    df = generate_comprehensive_data(2000)
    
    # Additional processing
    df['emotion_category'] = df['pred_top'].map({
        'joy': 'Positive', 'love': 'Positive', 'excitement': 'Positive', 
        'gratitude': 'Positive', 'hope': 'Positive', 'pride': 'Positive',
        'sadness': 'Negative', 'anger': 'Negative', 'fear': 'Negative', 
        'disgust': 'Negative', 'disappointment': 'Negative', 'anxiety': 'Negative',
        'neutral': 'Neutral', 'surprise': 'Mixed', 'confusion': 'Mixed'
    })
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    return df

# Load data with spinner
with st.spinner("🔮 Analyzing emotional patterns..."):
    df = load_and_process_data()

# ======================
# SIDEBAR CONTROLS
# ======================
with st.sidebar:
    st.markdown('<h2 style="color: white;">🎛️ Dashboard Controls</h2>', unsafe_allow_html=True)
    
    # Emotion selection with better styling
    st.markdown('<h4 style="color: #feca57;">Select Emotions</h4>', unsafe_allow_html=True)
    all_emotions = df["pred_top"].value_counts().index.tolist()
    selected_emotions = st.multiselect(
        "", 
        all_emotions, 
        default=all_emotions[:8],
        help="Choose emotions to analyze"
    )
    
    # Confidence threshold
    st.markdown('<h4 style="color: #feca57;">Confidence Filter</h4>', unsafe_allow_html=True)
    min_confidence = st.slider("", 0.5, 1.0, 0.65, step=0.05)
    
    # Time range
    st.markdown('<h4 style="color: #feca57;">Analysis Period</h4>', unsafe_allow_html=True)
    hours_back = st.selectbox(
        "", 
        [6, 12, 24, 48, 72, 168], 
        index=5,
        format_func=lambda x: f"Last {x} hours"
    )
    
    # Advanced options
    st.markdown('<h4 style="color: #feca57;">Display Options</h4>', unsafe_allow_html=True)
    show_advanced = st.checkbox("🔬 Advanced Analytics", True)
    show_correlations = st.checkbox("📊 Correlation Matrix", True)
    show_word_clouds = st.checkbox("☁️ Word Clouds", False)
    show_raw_data = st.checkbox("📋 Raw Data Table", False)

# Apply filters
cutoff_time = datetime.now() - timedelta(hours=hours_back)
filtered_df = df[
    (df["pred_top"].isin(selected_emotions)) & 
    (df["score"] >= min_confidence) &
    (df["timestamp"] >= cutoff_time)
]

if filtered_df.empty:
    st.error("🚫 No data matches your filters. Please adjust your selections.")
    st.stop()

# ======================
# ENHANCED KEY METRICS
# ======================
st.markdown('<h2 style="color: white; margin-top: 30px;">📊 Real-Time Emotional Insights</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

total_samples = len(filtered_df)
unique_emotions = len(filtered_df["pred_top"].unique())
avg_confidence = filtered_df["score"].mean()
avg_intensity = filtered_df["intensity"].mean()
dominant_emotion = filtered_df["pred_top"].mode()[0]

metrics = [
    ("Total Analyzed", f"{total_samples:,}", "🎯", "#FF6B6B"),
    ("Emotion Types", str(unique_emotions), "🎭", "#4ECDC4"),
    ("Avg Confidence", f"{avg_confidence:.2f}", "🎖️", "#45B7D1"),
    ("Avg Intensity", f"{avg_intensity:.2f}", "⚡", "#FECA57"),
    ("Dominant", dominant_emotion.title(), "👑", "#96CEB4")
]

for i, (col, (title, value, icon, color)) in enumerate(zip([col1, col2, col3, col4, col5], metrics)):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">{icon}</div>
                <h4 style="color: {color}; margin: 5px 0; font-size: 0.9rem;">{title}</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 0;">{value}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ======================
# MAIN VISUALIZATIONS GRID
# ======================
st.markdown('<h2 style="color: white; margin-top: 40px;">📈 Comprehensive Analytics Dashboard</h2>', unsafe_allow_html=True)

# Row 1: Distribution Analysis
col1, col2 = st.columns([7, 5])

with col1:
    st.markdown('<h3 style="color: white;">🎨 Emotion Distribution Analysis</h3>', unsafe_allow_html=True)
    
    emotion_counts = filtered_df["pred_top"].value_counts().reset_index()
    emotion_counts.columns = ["emotion", "count"]
    
    # Enhanced bar chart with gradients
    fig = go.Figure(data=[
        go.Bar(
            x=emotion_counts["emotion"],
            y=emotion_counts["count"],
            marker=dict(
                color=VIBRANT_COLORS[:len(emotion_counts)],
                line=dict(color='rgba(255,255,255,0.8)', width=2)
            ),
            text=emotion_counts["count"],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=[(count/total_samples)*100 for count in emotion_counts["count"]]
        )
    ])
    
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title=dict(text="Emotion Frequency Distribution", font=dict(size=18, color="white")),
        xaxis=dict(title="Emotions", gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<h3 style="color: white;">🥧 Emotional Composition</h3>', unsafe_allow_html=True)
    
    # Enhanced donut chart
    fig = go.Figure(data=[go.Pie(
        labels=emotion_counts["emotion"],
        values=emotion_counts["count"],
        hole=.4,
        marker=dict(
            colors=VIBRANT_COLORS[:len(emotion_counts)],
            line=dict(color='rgba(255,255,255,0.8)', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=12, color="white"),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title=dict(text="Emotion Distribution", font=dict(size=18, color="white")),
        annotations=[dict(text=f'Total<br>{total_samples}', x=0.5, y=0.5, font_size=16, showarrow=False, font_color="white")],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# Row 2: Advanced Analytics
col1, col2 = st.columns([6, 6])

with col1:
    st.markdown('<h3 style="color: white;">📊 Confidence vs Intensity Analysis</h3>', unsafe_allow_html=True)
    
    # Scatter plot with size encoding
    fig = px.scatter(
        filtered_df, 
        x="score", 
        y="intensity",
        color="pred_top",
        size="text_length",
        hover_data=["pred_top", "sentiment"],
        color_discrete_sequence=VIBRANT_COLORS,
        template="plotly_dark"
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Confidence Score",
        yaxis_title="Emotion Intensity",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<h3 style="color: white;">📈 Temporal Emotion Patterns</h3>', unsafe_allow_html=True)
    
    # Time series heatmap
    hourly_emotions = filtered_df.groupby(['hour', 'pred_top']).size().unstack(fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=hourly_emotions.values.T,
        x=hourly_emotions.index,
        y=hourly_emotions.columns,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Hour: %{x}<br>Emotion: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        title="Emotion Intensity by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Emotions",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# Row 3: Category Analysis and Trends
col1, col2 = st.columns([5, 7])

with col1:
    st.markdown('<h3 style="color: white;">🎭 Emotional Categories</h3>', unsafe_allow_html=True)
    
    category_counts = filtered_df["emotion_category"].value_counts()
    
    # Sunburst chart for categories
    fig = go.Figure(go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        marker=dict(colors=['#FF6B6B', '#4ECDC4', '#FECA57', '#96CEB4']),
        textinfo='label+percent',
        textfont=dict(size=14, color="white")
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<h3 style="color: white;">📅 Weekly Emotion Trends</h3>', unsafe_allow_html=True)
    
    # Multi-line time series
    daily_trends = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'emotion_category']).size().unstack(fill_value=0).reset_index()
    
    fig = go.Figure()
    
    colors = {'Positive': '#4ECDC4', 'Negative': '#FF6B6B', 'Neutral': '#96CEB4', 'Mixed': '#FECA57'}
    
    for category in daily_trends.columns[1:]:
        if category in daily_trends.columns:
            fig.add_trace(go.Scatter(
                x=daily_trends['timestamp'],
                y=daily_trends[category],
                mode='lines+markers',
                name=category,
                line=dict(color=colors.get(category, '#45B7D1'), width=3),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================
# ADVANCED ANALYTICS SECTION
# ======================
if show_advanced:
    st.markdown('<h2 style="color: white; margin-top: 40px;">🔬 Advanced Analytics Suite</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<h4 style="color: white;">📊 Confidence Distribution</h4>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=filtered_df["score"],
            nbinsx=20,
            marker_color='#4ECDC4',
            opacity=0.8,
            name="Confidence"
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h4 style="color: white;">⚡ Intensity Levels</h4>', unsafe_allow_html=True)
        
        fig = px.violin(
            filtered_df, 
            y="intensity", 
            x="emotion_category",
            color="emotion_category",
            color_discrete_sequence=VIBRANT_COLORS,
            template="plotly_dark"
        )
        
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown('<h4 style="color: white;">📏 Text Length Analysis</h4>', unsafe_allow_html=True)
        
        fig = px.box(
            filtered_df, 
            y="text_length", 
            x="emotion_category",
            color="emotion_category",
            color_discrete_sequence=VIBRANT_COLORS,
            template="plotly_dark"
        )
        
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================
# CORRELATION MATRIX
# ======================
if show_correlations:
    st.markdown('<h3 style="color: white; margin-top: 30px;">🔗 Feature Correlation Matrix</h3>', unsafe_allow_html=True)
    
    # Create correlation matrix
    numeric_cols = ['score', 'intensity', 'sentiment', 'text_length']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 14},
        hoverongaps=False
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        title="Feature Correlation Analysis",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================
# WORD CLOUDS
# ======================
if show_word_clouds:
    st.markdown('<h2 style="color: white; margin-top: 40px;">☁️ Emotion Word Clouds</h2>', unsafe_allow_html=True)
    
    top_emotions = filtered_df["pred_top"].value_counts().head(4).index.tolist()
    cols = st.columns(2)
    
    for i, emotion in enumerate(top_emotions):
        with cols[i % 2]:
            st.markdown(f'<h4 style="color: white;">{emotion.capitalize()} 💭</h4>', unsafe_allow_html=True)
            
            emotion_texts = filtered_df[filtered_df["pred_top"] == emotion]["text"]
            
            if len(emotion_texts) > 0:
                try:
                    text = " ".join(emotion_texts)
                    
                    wc = WordCloud(
                        width=500, 
                        height=300, 
                        background_color="rgba(255,255,255,0)",
                        mode="RGBA",
                        colormap="plasma",
                        max_words=50,
                        relative_scaling=0.5
                    ).generate(text)
                    
                    fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_facecolor('none')
                    st.pyplot(fig, transparent=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")

# ======================
# INSIGHTS SECTION
# ======================
st.markdown('<h2 style="color: white; margin-top: 40px;">🤖 AI-Powered Insights</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="insight-box">
        <h4>🎯 Top Performers</h4>
        <ul>
    """ + "".join([f"<li><strong>{emotion}:</strong> {count} samples ({(count/total_samples)*100:.1f}%)</li>" 
                   for emotion, count in filtered_df["pred_top"].value_counts().head(3).items()]) + """
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    high_conf_emotions = filtered_df[filtered_df["score"] > 0.9]["pred_top"].value_counts().head(3)
    st.markdown("""
    <div class="insight-box">
        <h4>🎖️ High Confidence Emotions</h4>
        <ul>
    """ + "".join([f"<li><strong>{emotion}:</strong> {count} high-confidence predictions</li>" 
                   for emotion, count in high_conf_emotions.items()]) + """
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_sentiment_by_category = filtered_df.groupby("emotion_category")["sentiment"].mean().round(2)
    st.markdown("""
    <div class="insight-box">
        <h4>📊 Sentiment Analysis</h4>
        <ul>
    """ + "".join([f"<li><strong>{cat}:</strong> {score} avg sentiment</li>" 
                   for cat, score in avg_sentiment_by_category.items()]) + """
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================
# RAW DATA TABLE
# ======================
if show_raw_data:
    st.markdown('<h2 style="color: white; margin-top: 40px;">📋 Detailed Analysis Table</h2>', unsafe_allow_html=True)
    
    display_df = filtered_df[["text", "pred_top", "score", "intensity", "sentiment", "timestamp"]].sort_values("score", ascending=False).head(50)
    
    st.dataframe(
        display_df,
        column_config={
            "text": st.column_config.TextColumn("📝 Text Sample", width="large"),
            "pred_top": st.column_config.TextColumn("🎭 Emotion", width="medium"),
            "score": st.column_config.NumberColumn("🎯 Confidence", format="%.3f", help="Model confidence (0-1)"),
            "intensity": st.column_config.NumberColumn("⚡ Intensity", format="%.3f", help="Emotion intensity (0-1)"),
            "sentiment": st.column_config.NumberColumn("💭 Sentiment", format="%.3f", help="Sentiment score (-1 to 1)"),
            "timestamp": st.column_config.DatetimeColumn("⏰ Timestamp")
        },
        use_container_width=True,
        hide_index=True
    )

# ======================
# ENHANCED FOOTER
# ======================
st.markdown("""
<div style="margin-top: 50px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; text-align: center;">
    <h3 style="color: white; margin-bottom: 20px;">🚀 Emotion Intelligence Dashboard</h3>
    <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
        <div style="color: rgba(255,255,255,0.9);">
            <strong>📊 Analytics Engine:</strong> Advanced ML Models
        </div>
        <div style="color: rgba(255,255,255,0.9);">
            <strong>🎨 Visualization:</strong> Plotly & Streamlit
        </div>
        <div style="color: rgba(255,255,255,0.9);">
            <strong>⚡ Performance:</strong> Real-time Processing
        </div>
    </div>
    <p style="color: rgba(255,255,255,0.8); margin-top: 15px; font-style: italic;">
        Powered by cutting-edge AI technology for comprehensive emotional intelligence analysis
    </p>
</div>
""", unsafe_allow_html=True)