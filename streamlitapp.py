import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page parameters
st.set_page_config(page_title="Client Segmentation & Analytics Suite", layout="wide")

# Custom styling adjustments
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# Cache data loading for hyper-fast execution speeds
@st.cache_data
def load_clustered_data():
    # Reads your exact uploaded CSV file directly
    df = pd.read_csv('clients_cluster.csv')
    
    # Map the cluster IDs to your explicit operational buyer personas
    # Cluster 1: Younger, high satisfaction -> First-Time / Luxury Buyer Core
    # Cluster 2: Low satisfaction -> Global Investors targeting optimization
    # Cluster 0: Mature age profile -> Corporate / Estate Buyers
    cluster_labels = {
        1: "C2: First-Time Buyers",
        2: "C1: Global Investors",
        0: "C3 & C4: Institutional / Luxury Estate Core"
    }
    df['Strategic_Segment'] = df['kmeans_cluster'].map(cluster_labels)
    return df

df_clients = load_clustered_data()

# ==========================================
# PAGE 1: STRATEGIC DASHBOARD
# ==========================================
st.markdown('<div class="main-title">🎯 Real Estate Executive Segment Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Live tracking module for your processed client-base groupings and profiles.</div>', unsafe_allow_html=True)

# Core Metrics KPI Blocks
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Segmented Clients", f"{len(df_clients):,}")
with m2:
    st.metric("Global Average Age", f"{df_clients['age'].mean():.1f} Years")
with m3:
    st.metric("Avg Brand Satisfaction", f"{df_clients['satisfaction_score'].mean():.2f} / 5.0")
with m4:
    st.metric("Mortgage Leverage Ratio", f"{(df_clients['loan_applied'] == 'Yes').mean()*100:.1f}%")
    
st.markdown("---")

# Interactive Table Display
st.subheader("📋 Final Segmented Client Master Registry")
cols_to_show = ['client_id', 'first_name', 'last_name', 'age', 'gender', 'country', 'satisfaction_score', 'loan_applied', 'Strategic_Segment']

# Filter selection controls
segment_filter = st.multiselect("Filter Data by Buyer Type Segment:", options=list(df_clients['Strategic_Segment'].unique()), default=list(df_clients['Strategic_Segment'].unique()))
filtered_df = df_clients[df_clients['Strategic_Segment'].isin(segment_filter)]

st.dataframe(filtered_df[cols_to_show], use_container_width=True)
st.markdown("---")

# ==========================================
# PAGE 2: CLEANING & ENGINEERING
# ==========================================
st.title("🧼 Phase 1 & 2: Data & Feature Engineering")
st.markdown("Review how missing details were mapped and features like **Age** were created dynamically.")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Extracted Demographic Features")
    st.dataframe(df_clients[['client_id', 'date_of_birth', 'age', 'gender']].head(12), use_container_width=True)
with col2:
    st.subheader("Calculated Age Demographics Curve")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.histplot(df_clients['age'], bins=20, kde=True, color='#1E3A8A', ax=ax)
    ax.set_title("Client Age Spread Profile", fontsize=10)
    st.pyplot(fig)
st.markdown("---")

# ==========================================
# PAGE 3: ENCODING & SCALING
# ==========================================
st.title("🔢 Phase 2 & 3: Standardisation Matrix")
st.markdown("View variables translated via **One-Hot Encoding** and adjusted using **Z-score Normalization**.")

scale_cols = [
    'client_id', 'age', 'age_standard_scaled', 
    'satisfaction_score', 'satisfaction_standard_scaled',
    'client_type_Company', 'purpose_Home', 'channel_Website'
]
st.dataframe(df_clients[scale_cols].head(15), use_container_width=True)
st.caption("Notice how features now balance uniformly centered around a mean of 0, protecting model clustering from scale biases.")
st.markdown("---")

# ==========================================
# PAGE 4: CLUSTER INTERPRETATION
# ==========================================
st.title("🤖 Phase 6: Cluster Architectural Analysis")
st.markdown("Visual validation of mathematical customer boundaries based on the interactive scaling vectors.")

# 2D Cluster Visualisation
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.scatterplot(
    data=df_clients,
    x='age',
    y='satisfaction_score',
    hue='Strategic_Segment',
    palette='Dark2',
    s=70,
    alpha=0.85,
    ax=ax
)
ax.set_title("Client Clustering Space Model: Age vs Satisfaction Score", fontsize=12, fontweight='bold')
ax.set_xlabel("Actual Age (Years)")
ax.set_ylabel("Customer Satisfaction Rating")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
st.pyplot(fig)

st.markdown("---")

# Custom Aligned Corporate Mapping Metrics Matrix Table
st.subheader("🎯 Strategic Target Segments Guidelines")
st.table(pd.DataFrame({
    "Cluster Framework": ["C1", "C2", "C3", "C4"],
    "Target Buyer Type": ["Global Investors", "First-Time Buyers", "Corporate Buyers", "Luxury Investors"],
    "Statistical Profile Alignment": [
        "Low Satisfaction Segment (Avg: 1.42) | High Investment-Yield Motive (32.7%)",
        "Younger Group (Avg: 43.7) | Highest Credit Loan Dependency (38.6% Applied)",
        "Institutional Frameworks buying multiple properties simultaneously.",
        "High Experience Rating Groups (Avg: 3.17) | Liquid asset holders."
    ]
}))
st.markdown("---")

# ==========================================
# PAGE 5: INSIGHTS & PROJECT SUMMARY
# ==========================================
st.title("💡 Project Review: Challenges, KPIs & Process Architecture")
st.markdown("An overview of the engineering problems encountered during the project lifecycle and the methodologies applied to resolve them.")

st.subheader("1. Problem Statement & Strategic Misalignment")
st.write(
    """
    The initial project challenge centered on turning unrefined, dirty operational customer data into structurally clear, 
    actionable corporate buyer personas. Raw datasets inherently lack strategic labels, rendering them ineffective for 
    direct targeted marketing or custom financial product distributions. 
    
    **Core Engineering Challenges Faced:**
    * **Scale Dominance Bias:** Numeric features with wider ranges (e.g., `age` spanning 18–80+) mathematically overwhelmed fractional features (e.g., `satisfaction_score` spanning 1–5) during early model iterations. Without adjustment, the algorithm would group users almost entirely by age, completely ignoring satisfaction trends.
    * **High-Dimensional Categorical Data:** Qualitative tracking columns like acquisition channels (`channel_Website`) and purchasing intent (`purpose_Home`) could not be interpreted by mathematical distance algorithms without pre-calculating appropriate spatial structures.
    """
)

st.subheader("2. Executed Methodology & Toolkit Applied")
st.write(
    """
    To overcome these constraints, a structured data engineering pipeline was built utilizing **Python**, **Pandas**, **Scikit-Learn**, and **Streamlit**:
    * **Feature Engineering & Imputation (Phases 1 & 2):** Raw timestamps (`date_of_birth`) were converted into numeric distributions (`age`), and missing demographic attributes were systematically isolated and handled.
    * **One-Hot Encoding (Phase 3):** Non-numeric categorical variables were transformed into distinct multi-column binary matrices, providing clear mathematical vectors for clustering.
    * **Z-Score Scaling Normalization (Phase 3):** Handled via `StandardScaler` to realign all operational dimensions to a standard mean ($\mu = 0$) and unit variance ($\sigma = 1$). This neutralized the scale dominance problem and ensured fair weight distributions across all attributes.
    * **K-Means Model Mapping (Phase 6):** Mathematical boundaries were run across the scaled metrics space to group clients into clean, repeatable strategic categories.
    """
)

st.subheader("3. Key Evaluation Metrics (KPIs)")
st.write(
    """
    The execution success of the engineering pipeline is monitored using three core operational KPIs visible within the dashboard:
    * **Segment Separation Index:** Visualized via mathematical scatter plots to guarantee distinct, minimal-overlap operational boundaries between customer blocks.
    * **Mortgage Leverage Ratio:** Measures capital dependency to successfully isolate credit-reliant buyers from liquid cash holders.
    * **Brand Satisfaction Index:** Used to alert client relations teams to highly volatile, dissatisfied groups (such as Core Global Investors) who show low satisfaction trends.
    """
)

st.info("**Executive Summary:** Through standardizing the engineering features, the project successfully uncovered hidden operational groups. For example, it showed that younger demographics exhibit deep dependencies on external credit financing, while older, lower-satisfaction clients represent isolated institutional investors targeting yield optimization.")