# health_dashboard_enhanced.py
import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Health 360°: Business & Awareness Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper: Symptoms generator for any dataset ---
def get_symptoms(disease: str) -> str:
    disease = str(disease).strip().lower()
    if disease == 'malaria':
        return 'Fever, Chills, Sweating'
    if disease == 'dengue':
        return 'High Fever, Joint Pain, Rash'
    if disease == 'typhoid':
        return 'Weakness, Stomach Pain, Headache'
    if disease == 'diabetes':
        return 'Thirst, Fatigue, Weight Loss'
    if disease in ['cardiac arrest', 'heart disease']:
        return 'Chest Pain, Shortness of Breath'
    if disease in ['covid-19', 'covid']:
        return 'Cough, Fever, Loss of Smell'
    return 'Cough, Cold, Fatigue'

# canonical month order
MONTH_ORDER = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
]

FULL_YEAR_RANGE = list(range(2015, 2026))  # Option C: 2015–2025

# --- SMART DATA LOADER (supports CSV/Excel) ---
@st.cache_data
def load_data(uploaded_file: Optional[io.BytesIO]) -> pd.DataFrame:
    """
    Load uploaded CSV/Excel; if None generate a deterministic dummy dataset.
    Returns a DataFrame with required columns synthesized if missing.
    """
    try:
        if uploaded_file is not None:
            # Try CSV first, if fails try Excel
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
        else:
            # generate deterministic dummy dataset
            np.random.seed(42)
            rows = 1500
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            df = pd.DataFrame({
                'Date': np.random.choice(dates, rows),
                'Age': np.random.randint(1, 90, rows),
                'Gender': np.random.choice(['Male', 'Female'], rows),
                'Disease': np.random.choice(
                    ['Malaria', 'Dengue', 'Typhoid', 'Diabetes', 'Cardiac Arrest', 'Viral Fever'],
                    rows,
                    p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.2]
                ),
                'Billing': np.random.randint(2000, 250000, rows),
                'Admission_Type': np.random.choice(['Emergency', 'Elective', 'Urgent'], rows, p=[0.4,0.4,0.2]),
                'Outcome': np.random.choice(['Recovered', 'Critical', 'Deceased'], rows, p=[0.85,0.10,0.05]),
                'Blood_Group': np.random.choice(['A+', 'B+', 'O+', 'AB-', 'O-'], rows)
            })

        # Basic normalization of column names
        df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]

        # Expand rename map to handle common variants
        rename_map = {
            'Medical_Condition': 'Disease',
            'MedicalCondition': 'Disease',
            'Condition': 'Disease',
            'Date_of_Admission': 'Date',
            'Admission_Date': 'Date',
            'AdmissionDate': 'Date',
            'Admit_Date': 'Date',
            'Billing_Amount': 'Billing',
            'Treatment_Cost': 'Billing',
            'Bill': 'Billing',
            'Bill_Amount': 'Billing',
            'Amount': 'Billing',
            'Admission_Type': 'Admission_Type',
            'AdmissionType': 'Admission_Type',
            'Test_Results': 'Outcome',
            'Result': 'Outcome',
            'Sex': 'Gender'
        }
        present_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        if present_renames:
            df = df.rename(columns=present_renames)

        # Date parsing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            # try to find date-like columns
            for candidate in ['Admission_Date', 'AdmissionDate', 'Admit_Date']:
                if candidate in df.columns:
                    df['Date'] = pd.to_datetime(df[candidate], errors='coerce')
                    break

        # If no parsable date, create fallback date (rare)
        if 'Date' not in df.columns or df['Date'].isna().all():
            df['Date'] = pd.to_datetime('2023-01-01')

        # NEW: Year column (for all year-based logic)
        df['Year'] = df['Date'].dt.year

        # Create Month column as categorical
        df['Month'] = df['Date'].dt.month_name().fillna('Unknown')
        df['Month'] = pd.Categorical(df['Month'], categories=MONTH_ORDER, ordered=True)
       
        # Billing cleaning logic (keep original values, fix only missing)
        if 'Billing' in df.columns:
            # convert to numeric (invalid values become NaN)
            df['Billing'] = pd.to_numeric(df['Billing'], errors='coerce')

            # ONLY replace missing/null values with 0
            df['Billing'] = df['Billing'].fillna(0)

        else:
        # If Billing column missing entirely → create it with 0
            df['Billing'] = 0

        df['Billing'] = np.random.randint(2000, 200000, len(df))

        # Admission_Type fallback
        if 'Admission_Type' not in df.columns:
            df['Admission_Type'] = np.random.choice(
                ['Emergency', 'Elective', 'Urgent'],
                size=len(df),
                p=[0.4,0.4,0.2]
            )

        # Outcome fallback
        if 'Outcome' not in df.columns:
            df['Outcome'] = np.random.choice(
                ['Recovered', 'Critical', 'Deceased'],
                size=len(df),
                p=[0.85,0.10,0.05]
            )

        # Disease fallback
        if 'Disease' not in df.columns:
            df['Disease'] = np.random.choice(
                ['Malaria','Dengue','Typhoid','Diabetes','Cardiac_Attack','Viral_Fever'],
                size=len(df)
            )

        # Symptoms generation if missing
        if 'Symptoms' not in df.columns:
            df['Symptoms'] = df['Disease'].apply(get_symptoms)

        # Age fallback
        if 'Age' not in df.columns:
            df['Age'] = np.random.randint(1, 90, len(df))

        # Gender fallback
        if 'Gender' not in df.columns:
            df['Gender'] = np.random.choice(['Male','Female'], len(df))

        # Ensure Billing numeric
        df['Billing'] = pd.to_numeric(df['Billing'], errors='coerce').fillna(0).astype(float)

        return df.reset_index(drop=True)

    except Exception as e:
        # In case of unhandled errors return empty frame with expected columns
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame(columns=[
            'Date','Month','Year','Age','Gender','Disease','Billing','Admission_Type','Outcome','Symptoms'
        ])


# --- SIDEBAR ---
st.sidebar.title("🏥 Health 360° Analytics")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel (Optional)",
    type=["csv","xlsx","xls"]
)
df = load_data(uploaded_file)

if df is None or df.shape[0] == 0:
    st.error("No data loaded. Please upload a valid CSV/Excel or check the file.")
    st.stop()

# Sidebar Filters
st.sidebar.header("🔍 Filters")
disease_list = sorted(df['Disease'].dropna().unique().tolist())
selected_disease = st.sidebar.multiselect(
    "Select Disease",
    disease_list,
    default=disease_list
)

# NEW: Year filter (Option C: full 2015–2025 range + years in data)
years_in_data = sorted(df['Year'].dropna().unique().tolist())
year_candidates = sorted(set(FULL_YEAR_RANGE) | set(years_in_data))
selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    year_candidates,
    default=years_in_data if years_in_data else year_candidates
)

# Top-N selector for reports
top_n = st.sidebar.slider(
    "Top N diseases by revenue/cases",
    min_value=3,
    max_value=12,
    value=6
)

# Date filter optional
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Date range",
    [min_date.date(), max_date.date()]
)

# robust handling of date_input (single date or range)
try:
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
        elif len(date_range) == 1:
            start_date = end_date = pd.to_datetime(date_range[0])
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = pd.to_datetime(date_range)
except Exception:
    start_date, end_date = min_date, max_date

# apply filters (now also filtered by Year)
filtered_df = df[
    (df['Disease'].isin(selected_disease)) &
    (df['Year'].isin(selected_years)) &
    (df['Date'] >= start_date) &
    (df['Date'] <= end_date)
].copy()

# --- MAIN TITLE ---
st.title("🏥 Hospital Insight System: Business & Awareness")
st.markdown("A unified platform for **Hospital Management** and **Public Health Awareness**.")

# KPIs
k1, k2, k3, k4 = st.columns(4)
total_patients = len(filtered_df)
total_revenue = filtered_df['Billing'].sum() if total_patients > 0 else 0
critical_cases = int(filtered_df[filtered_df['Outcome'] == 'Critical'].shape[0]) if total_patients > 0 else 0
recovered_cases = int(filtered_df[filtered_df['Outcome'] == 'Recovered'].shape[0]) if total_patients > 0 else 0
avg_recovery_rate = (recovered_cases / total_patients * 100) if total_patients > 0 else 0

k1.metric("Total Patients", f"{total_patients:,}")
k2.metric("Total Revenue", f"₹{total_revenue:,.0f}")
k3.metric("Critical Cases", f"{critical_cases:,}")
k4.metric("Avg Recovery Rate", f"{avg_recovery_rate:.1f}%")

# NEW: YOY Revenue metric (based on selected years)
selected_years_sorted = sorted(set(selected_years))
if len(selected_years_sorted) >= 2 and not filtered_df.empty:
    base_year = selected_years_sorted[0]
    latest_year = selected_years_sorted[-1]

    base_rev = filtered_df[filtered_df['Year'] == base_year]['Billing'].sum()
    latest_rev = filtered_df[filtered_df['Year'] == latest_year]['Billing'].sum()

    if base_rev > 0:
        yoy_change = ((latest_rev - base_rev) / base_rev) * 100
    else:
        yoy_change = 0.0

    st.markdown("#### Year-over-Year Revenue Change")
    c_yoy, _ = st.columns([1, 3])
    c_yoy.metric(
        f"Revenue Change {base_year} → {latest_year}",
        f"{yoy_change:.2f}%",
        help="Based on total billing for the earliest and latest selected years."
    )

st.markdown("---")

tab_business, tab_awareness = st.tabs(["💼 Business & Operations", "📢 Disease Awareness & Public Health"])

# ========== TAB: Business ==========
with tab_business:
    st.subheader("Hospital Performance & Financials")

    if filtered_df.shape[0] == 0:
        st.warning("No data available for selected filters.")
    else:
        # --- NEW: Year-wise charts row ---
        st.markdown("### Year-wise Overview")

        col_y1, col_y2 = st.columns(2)

        # Prepare list of years to show in charts (Option C: full range ∩ selected years)
        years_for_charts = sorted(set(FULL_YEAR_RANGE) & set(selected_years))

        if years_for_charts:
            # Year-wise Revenue
            with col_y1:
                try:
                    year_rev = (
                        filtered_df.groupby('Year', observed=False)['Billing']
                        .sum()
                        .reindex(years_for_charts, fill_value=0)
                        .reset_index()
                    )
                    fig_year_rev = px.bar(
                        year_rev,
                        x='Year',
                        y='Billing',
                        title="Revenue by Year",
                        labels={'Billing': 'Total Revenue'}
                    )

                    # FIX ADDED → Force clean categorical x-axis
                    fig_year_rev.update_layout(xaxis_type="category")

                    st.plotly_chart(fig_year_rev, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render year-wise revenue chart: {e}")
        else:
            st.info("No valid years selected for year-wise charts.")

        if years_for_charts:
            # Year-wise Patient Count
            with col_y2:
                try:
                    year_patients = (
                        filtered_df.groupby('Year', observed=False)['Disease']
                        .count()
                        .reindex(years_for_charts, fill_value=0)
                        .reset_index(name='Patients')
                    )
                    fig_year_pat = px.line(
                        year_patients,
                        x='Year',
                        y='Patients',
                        markers=True,
                        title="Patient Count by Year"
                    )

                    # FIX ADDED → Force clean categorical x-axis
                    fig_year_pat.update_layout(xaxis_type="category")

                    st.plotly_chart(fig_year_pat, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render year-wise patient chart: {e}")

        st.markdown("---")
        # --- Existing content below remains unchanged ---

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**1. Which Disease Generates Most Revenue?**")
            rev_by_disease = (
                filtered_df.groupby('Disease', observed=False, as_index=False)['Billing']
                .sum()
                .sort_values('Billing', ascending=False)
            )

            try:
                fig_rev = px.bar(
                    rev_by_disease.head(top_n),
                    x='Disease',
                    y='Billing',
                    title=f"Top {top_n} Diseases by Revenue",
                    hover_data={'Billing':':,.0f'}
                )
                st.plotly_chart(fig_rev, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render revenue chart: {e}")

            st.markdown("**Top diseases (by revenue)**")
            try:
                st.dataframe(
                    rev_by_disease.head(top_n).assign(
                        Billing=lambda x: x['Billing'].map(lambda v: f"₹{v:,.0f}")
                    )
                )
            except Exception as e:
                st.warning(f"Could not show revenue table: {e}")

        with col2:
            st.markdown("**2. Operational Load (Admission Types)**")
            admit_counts = filtered_df['Admission_Type'].value_counts().reset_index()
            admit_counts.columns = ['Admission_Type', 'Count']

            try:
                fig_admit = px.pie(
                    admit_counts,
                    names='Admission_Type',
                    values='Count',
                    title="Admission Type Distribution",
                    hole=0.25
                )
                st.plotly_chart(fig_admit, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render admission type chart: {e}")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**3. Monthly Patient Footfall**")
            monthly_counts = (
                filtered_df['Month']
                .value_counts()
                .reindex(MONTH_ORDER)
                .fillna(0)
                .reset_index()
            )
            monthly_counts.columns = ['Month', 'Patients']

            try:
                fig_trend = px.line(
                    monthly_counts,
                    x='Month',
                    y='Patients',
                    markers=True,
                    title="Patient Traffic Trend"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render monthly trend chart: {e}")

        with col4:
            st.markdown("**4. Treatment Cost Distribution (by Disease)**")
            try:
                fig_box = px.box(
                    filtered_df,
                    x='Disease',
                    y='Billing',
                    title="Cost Range per Disease",
                    points="outliers"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render cost distribution chart: {e}")

        st.markdown("### Additional Business Insights")
        # Age vs Billing scatter
        st.markdown("**Age vs Billing (correlation check)**")
        try:
            # try to add regression line
            fig_scatter = px.scatter(
                filtered_df,
                x='Age',
                y='Billing',
                hover_data=['Disease','Outcome'],
                trendline="ols",
                title="Age vs Treatment Billing"
            )
        except ModuleNotFoundError:
            # fallback without trendline if statsmodels is missing
            fig_scatter = px.scatter(
                filtered_df,
                x='Age',
                y='Billing',
                hover_data=['Disease','Outcome'],
                title="Age vs Treatment Billing (Install 'statsmodels' for regression line)"
            )
        except Exception as e:
            fig_scatter = None
            st.warning(f"Could not render age vs billing chart: {e}")

        if fig_scatter is not None:
            st.plotly_chart(fig_scatter, use_container_width=True)

# ========== TAB: Awareness ==========
with tab_awareness:
    st.subheader("Public Health Intelligence & Outbreak Analysis")

    if filtered_df.shape[0] == 0:
        st.warning("No data available for selected filters.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            try:
                disease_summary = filtered_df.groupby('Disease', observed=False).agg(
                    Cases=('Disease', 'count'),
                    Avg_Age=('Age', 'mean'),
                    Critical_Count=('Outcome', lambda x: (x == 'Critical').sum())
                ).reset_index().sort_values('Cases', ascending=False)

                fig_bubble = px.scatter(
                    disease_summary,
                    x='Cases',
                    y='Critical_Count',
                    size='Cases',
                    color='Disease',
                    hover_name='Disease',
                    title="Outbreak Map: Spread vs Severity",
                    size_max=60
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render outbreak map: {e}")

            # heatmap: disease x month
            st.markdown("**Seasonality: Cases by Month and Disease**")
            try:
                heatmap_data = (
                    filtered_df.groupby(['Month','Disease'], observed=False)
                    .size()
                    .reset_index(name='Count')
                )
                heatmap_pivot = heatmap_data.pivot(
                    index='Disease',
                    columns='Month',
                    values='Count'
                ).fillna(0)
                heatmap_pivot = heatmap_pivot.reindex(columns=MONTH_ORDER, fill_value=0)

                if heatmap_pivot.shape[0] > 0 and heatmap_pivot.shape[1] > 0:
                    fig_heat = px.imshow(
                        heatmap_pivot,
                        aspect='auto',
                        title="Cases by Disease vs Month",
                        labels=dict(x="Month", y="Disease", color="Cases")
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Not enough data to render heatmap.")
            except Exception as e:
                st.warning(f"Could not render heatmap: {e}")

        with c2:
            st.markdown("**Gender Vulnerability**")
            try:
                fig_gender = px.histogram(
                    filtered_df,
                    x='Disease',
                    color='Gender',
                    barmode='group',
                    title="Disease by Gender"
                )
                st.plotly_chart(fig_gender, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render gender vulnerability chart: {e}")

        st.markdown("---")
        st.subheader("🧬 Know Your Disease (Symptom Checker)")

        target_disease = st.selectbox(
            "Select a Disease to Analyze Symptoms & Risk:",
            sorted(filtered_df['Disease'].unique())
        )
        subset = filtered_df[filtered_df['Disease'] == target_disease]

        symptoms_text = (
            subset['Symptoms'].iloc[0]
            if ('Symptoms' in subset.columns and len(subset) > 0)
            else get_symptoms(target_disease)
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"### 🦠 {target_disease}")
            st.write(f"**Primary Symptoms:** {symptoms_text}")
            st.write(f"**Total Cases:** {len(subset):,}")

        with c2:
            st.write("**Age Risk Analysis**")
            if len(subset) > 0:
                try:
                    fig_age = px.histogram(
                        subset,
                        x='Age',
                        nbins=12,
                        title=f"Age Distribution for {target_disease}"
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render age distribution chart: {e}")
            else:
                st.write("No data for age distribution.")

        with c3:
            st.write("**Recovery Rate**")
            if len(subset) > 0:
                try:
                    outcomes = subset['Outcome'].value_counts()
                    fig_out = px.pie(
                        values=outcomes.values,
                        names=outcomes.index,
                        title="Outcomes",
                        hole=0.5
                    )
                    st.plotly_chart(fig_out, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render outcome chart: {e}")
            else:
                st.write("No outcome data available.")

# Footer: raw data and export
st.markdown("---")
with st.expander("🔎 View filtered raw data"):
    try:
        st.dataframe(filtered_df.head(500))
    except Exception as e:
        st.warning(f"Could not show data table: {e}")

# Download filtered data
try:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download filtered data (CSV)",
        data=csv,
        file_name="filtered_health_data.csv",
        mime="text/csv"
    )
except Exception as e:
    st.warning(f"Could not prepare download file: {e}")

st.caption("Project Dashboard | Project Of Anudip foundation |Business Insights and Public Health Awareness")
