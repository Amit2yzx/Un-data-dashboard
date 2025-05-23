import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as colors

# Set page configuration
st.set_page_config(
    page_title="UN Voting Data Dashboard",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0277BD;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #90CAF9;
        padding-bottom: 0.5rem;
    }
    .section {
        padding: 1.5rem;
        border-radius: 0.7rem;
        margin-bottom: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f1f7fe;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f1f7fe;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .insight-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .country-group {
        font-weight: bold;
        color: #0277BD;
    }
</style>
""", unsafe_allow_html=True)

# Define color schemes for consistent visualization
color_schemes = {
    'vote_types': {'Yes': '#4CAF50', 'No': '#F44336', 'Absent': '#9E9E9E', 'Y': '#4CAF50', 'N': '#F44336', 'A': '#9E9E9E', 'X': '#FFC107'},
    'main_palette': px.colors.qualitative.Bold,
    'agreement_scale': px.colors.sequential.Blues,
    'diverging': px.colors.diverging.RdBu
}

# Header
st.markdown("<h1 class='main-header'>United Nations Voting Data Analysis</h1>", unsafe_allow_html=True)
st.markdown("This dashboard provides insights into UN voting patterns based on historical data.")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('UN DATA.csv', low_memory=False)
    # Extract year from the Date column
    df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
    # Create Topic column
    df['Topic'] = df['Title'].apply(extract_keywords)

    # Define country groups for analysis
    df['Country_Group'] = 'Other'

    # Define country groups
    country_groups = {
        'Western Powers': ['UNITED STATES', 'UNITED KINGDOM', 'FRANCE', 'GERMANY', 'CANADA', 'AUSTRALIA'],
        'BRICS': ['BRAZIL', 'RUSSIAN FEDERATION', 'INDIA', 'CHINA', 'SOUTH AFRICA'],
        'Middle East': ['ISRAEL', 'IRAN (ISLAMIC REPUBLIC OF)', 'SAUDI ARABIA', 'TURKEY', 'EGYPT', 'SYRIA', 'IRAQ'],
        'Asia-Pacific': ['JAPAN', 'CHINA', 'INDIA', 'PAKISTAN', 'INDONESIA', 'AUSTRALIA', 'REPUBLIC OF KOREA'],
        'EU Members': ['FRANCE', 'GERMANY', 'ITALY', 'SPAIN', 'POLAND', 'NETHERLANDS', 'BELGIUM', 'SWEDEN', 'AUSTRIA', 'DENMARK', 'FINLAND', 'IRELAND', 'LUXEMBOURG', 'GREECE', 'PORTUGAL'],
        'African Union': ['SOUTH AFRICA', 'NIGERIA', 'EGYPT', 'ALGERIA', 'MOROCCO', 'KENYA', 'ETHIOPIA', 'GHANA', 'SENEGAL', 'UGANDA']
    }

    # Assign country groups
    for group, countries in country_groups.items():
        for country in countries:
            if country in df.columns:
                # Create a mask for rows where this country has a vote
                mask = ~df[country].isna()
                # Only update the group for rows where this country has a vote
                df.loc[mask, 'Country_Group'] = group

    return df

def extract_keywords(title):
    keywords = [
        'nuclear', 'human rights', 'peace', 'security', 'economic', 'development',
        'terrorism', 'climate', 'disarmament', 'palestine', 'israel', 'middle east',
        'africa', 'asia', 'europe', 'america', 'refugee', 'migration', 'women', 'children',
        'health', 'education', 'environment', 'trade', 'finance', 'technology'
    ]
    title_lower = title.lower()
    for keyword in keywords:
        if keyword in title_lower:
            return keyword
    return 'other'

# Function to calculate agreement percentage between two countries
@st.cache_data
def calculate_agreement(country1, country2, data):
    # Filter rows where both countries have valid votes (not NaN)
    valid_votes = data.dropna(subset=[country1, country2])

    # Count agreements
    agreements = (valid_votes[country1] == valid_votes[country2]).sum()
    total = len(valid_votes)

    if total > 0:
        return (agreements / total) * 100
    else:
        return 0

# Function to generate insights from agreement data
def generate_agreement_insights(agreement_matrix):
    insights = []

    # Find highest agreement pair
    max_agreement = 0
    max_pair = None

    # Find lowest agreement pair
    min_agreement = 100
    min_pair = None

    for i in range(len(agreement_matrix.index)):
        for j in range(len(agreement_matrix.columns)):
            if i != j:  # Skip diagonal (self-agreement)
                country1 = agreement_matrix.index[i]
                country2 = agreement_matrix.columns[j]
                agreement = agreement_matrix.iloc[i, j]

                if agreement > max_agreement:
                    max_agreement = agreement
                    max_pair = (country1, country2)

                if agreement < min_agreement:
                    min_agreement = agreement
                    min_pair = (country1, country2)

    if max_pair:
        insights.append(f"Highest agreement: {max_pair[0]} and {max_pair[1]} agree on {max_agreement:.1f}% of votes")

    if min_pair:
        insights.append(f"Lowest agreement: {min_pair[0]} and {min_pair[1]} agree on only {min_agreement:.1f}% of votes")

    return insights

# Load data
try:
    df = load_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.markdown("## Filters")
st.sidebar.markdown("Use these filters to focus your analysis on specific time periods, councils, or topics.")

# Year range filter
years = sorted(df['Year'].dropna().unique())
if len(years) > 0:
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
else:
    filtered_df = df

# Council filter
councils = df['Council'].unique()
selected_council = st.sidebar.multiselect(
    "Select Council",
    options=councils,
    default=councils
)
if selected_council:
    filtered_df = filtered_df[filtered_df['Council'].isin(selected_council)]

# Topic filter
topics = sorted(df['Topic'].unique())
selected_topic = st.sidebar.multiselect(
    "Select Topic",
    options=topics,
    default=topics
)
if selected_topic:
    filtered_df = filtered_df[filtered_df['Topic'].isin(selected_topic)]

# Country group filter
st.sidebar.markdown("---")
st.sidebar.markdown("### Country Groups")
country_groups = ['Western Powers', 'BRICS', 'Middle East', 'Asia-Pacific', 'EU Members', 'African Union', 'Other']
selected_groups = st.sidebar.multiselect(
    "Filter by Country Groups",
    options=country_groups,
    default=country_groups
)

# Display basic info about the filtered data
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)

# Create metrics with improved styling
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Total Resolutions", f"{len(filtered_df):,}")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Date Range", f"{int(filtered_df['Year'].min())} - {int(filtered_df['Year'].max())}")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Councils", len(filtered_df['Council'].unique()))
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Topics Covered", len(filtered_df['Topic'].unique()))
    st.markdown("</div>", unsafe_allow_html=True)

# Add a quick insight about the data
total_yes = filtered_df['YES COUNT'].sum()
total_no = filtered_df['NO COUNT'].sum()
total_absent = filtered_df['ABSENT COUNT'].sum()
total_votes = total_yes + total_no + total_absent

st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
st.markdown(f"**Quick Insight:** In the selected period, UN resolutions received **{total_yes:,}** YES votes ({total_yes/total_votes*100:.1f}%), **{total_no:,}** NO votes ({total_no/total_votes*100:.1f}%), and **{total_absent:,}** ABSENT votes ({total_absent/total_votes*100:.1f}%).")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Display sample data
with st.expander("View Sample Data"):
    st.dataframe(filtered_df.head(10))

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["Resolution Analysis", "Voting Patterns", "Country Analysis", "Topic Analysis"])

with tab1:
    st.markdown("<h2 class='sub-header'>Resolution Analysis</h2>", unsafe_allow_html=True)

    # Resolutions by council
    st.markdown("### Resolutions by Council")
    council_counts = filtered_df['Council'].value_counts().reset_index()
    council_counts.columns = ['Council', 'Count']

    fig = px.bar(
        council_counts,
        x='Council',
        y='Count',
        color='Council',
        title='Number of Resolutions by Council'
    )
    fig.update_layout(xaxis_title='Council', yaxis_title='Number of Resolutions')
    st.plotly_chart(fig, use_container_width=True)

    # Resolutions over time
    st.markdown("### Resolutions Over Time")
    yearly_counts = filtered_df.groupby('Year').size().reset_index(name='Count')

    fig = px.line(
        yearly_counts,
        x='Year',
        y='Count',
        markers=True,
        title='Number of Resolutions by Year'
    )
    fig.update_layout(xaxis_title='Year', yaxis_title='Number of Resolutions')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h2 class='sub-header'>Voting Patterns</h2>", unsafe_allow_html=True)

    # Average votes per resolution
    st.markdown("### Average Votes per Resolution")
    vote_columns = ['YES COUNT', 'NO COUNT', 'ABSENT COUNT']
    vote_data = filtered_df[vote_columns].mean().reset_index()
    vote_data.columns = ['Vote Type', 'Average Count']

    fig = px.bar(
        vote_data,
        x='Vote Type',
        y='Average Count',
        color='Vote Type',
        title='Average Votes per Resolution'
    )
    fig.update_layout(xaxis_title='Vote Type', yaxis_title='Average Count')
    st.plotly_chart(fig, use_container_width=True)

    # Vote distribution
    st.markdown("### Vote Distribution")
    vote_dist = pd.DataFrame({
        'Vote Type': ['Yes', 'No', 'Absent'],
        'Average Percentage': [
            filtered_df['YES COUNT'].sum() / filtered_df['TOTAL VOTES'].sum() * 100,
            filtered_df['NO COUNT'].sum() / filtered_df['TOTAL VOTES'].sum() * 100,
            filtered_df['ABSENT COUNT'].sum() / filtered_df['TOTAL VOTES'].sum() * 100
        ]
    })

    fig = px.pie(
        vote_dist,
        values='Average Percentage',
        names='Vote Type',
        title='Overall Vote Distribution',
        color='Vote Type',
        color_discrete_map={'Yes': 'green', 'No': 'red', 'Absent': 'gray'}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='sub-header'>Country Analysis</h2>", unsafe_allow_html=True)

    # Select countries for analysis
    major_countries = ['UNITED STATES', 'RUSSIAN FEDERATION', 'CHINA', 'UNITED KINGDOM', 'FRANCE']

    # Allow user to select countries
    selected_countries = st.multiselect(
        "Select Countries to Analyze",
        options=[col for col in df.columns if col not in ['Council', 'Date', 'Title', 'Resolution', 'TOTAL VOTES', 'NO-VOTE COUNT', 'ABSENT COUNT', 'NO COUNT', 'YES COUNT', 'Link', 'token', 'Year', 'Topic']],
        default=major_countries
    )

    if selected_countries:
        # Voting patterns for selected countries
        st.markdown("### Voting Patterns by Country")

        # Create subplots
        fig = make_subplots(rows=len(selected_countries), cols=1,
                           subplot_titles=[f"Voting Pattern: {country}" for country in selected_countries],
                           vertical_spacing=0.05)

        for i, country in enumerate(selected_countries, 1):
            vote_counts = filtered_df[country].value_counts().reset_index()
            vote_counts.columns = ['Vote', 'Count']

            fig.add_trace(
                go.Bar(x=vote_counts['Vote'], y=vote_counts['Count'], name=country),
                row=i, col=1
            )

        fig.update_layout(height=300*len(selected_countries), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Agreement analysis
        if len(selected_countries) > 1:
            st.markdown("### Voting Agreement Between Countries")

            # Function to calculate agreement percentage
            @st.cache_data
            def calculate_agreement(country1, country2, data):
                # Filter rows where both countries have valid votes (not NaN)
                valid_votes = data.dropna(subset=[country1, country2])

                # Count agreements
                agreements = (valid_votes[country1] == valid_votes[country2]).sum()
                total = len(valid_votes)

                if total > 0:
                    return (agreements / total) * 100
                else:
                    return 0

            # Create agreement matrix
            agreement_data = []
            for country1 in selected_countries:
                for country2 in selected_countries:
                    if country1 != country2:
                        agreement = calculate_agreement(country1, country2, filtered_df)
                        agreement_data.append({
                            'Country 1': country1,
                            'Country 2': country2,
                            'Agreement (%)': agreement
                        })

            agreement_df = pd.DataFrame(agreement_data)

            # Create heatmap
            agreement_matrix = agreement_df.pivot(index='Country 1', columns='Country 2', values='Agreement (%)')

            fig = px.imshow(
                agreement_matrix,
                text_auto='.1f',
                color_continuous_scale='YlGnBu',
                title='Voting Agreement Between Countries (%)'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("<h2 class='sub-header'>Topic Analysis</h2>", unsafe_allow_html=True)

    # Resolutions by topic
    st.markdown("### Resolutions by Topic")
    topic_counts = filtered_df['Topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']

    fig = px.bar(
        topic_counts,
        x='Topic',
        y='Count',
        color='Topic',
        title='Number of Resolutions by Topic'
    )
    fig.update_layout(xaxis_title='Topic', yaxis_title='Number of Resolutions')
    st.plotly_chart(fig, use_container_width=True)

    # Topics over time
    st.markdown("### Topics Over Time")
    topic_year = filtered_df.groupby(['Year', 'Topic']).size().reset_index(name='Count')

    fig = px.line(
        topic_year,
        x='Year',
        y='Count',
        color='Topic',
        markers=True,
        title='Topics Over Time'
    )
    fig.update_layout(xaxis_title='Year', yaxis_title='Number of Resolutions')
    st.plotly_chart(fig, use_container_width=True)

    # Vote distribution by topic
    st.markdown("### Vote Distribution by Topic")

    # Calculate average yes, no, absent votes by topic
    topic_votes = filtered_df.groupby('Topic')[['YES COUNT', 'NO COUNT', 'ABSENT COUNT']].mean().reset_index()

    # Melt the dataframe for easier plotting
    topic_votes_melted = pd.melt(
        topic_votes,
        id_vars=['Topic'],
        value_vars=['YES COUNT', 'NO COUNT', 'ABSENT COUNT'],
        var_name='Vote Type',
        value_name='Average Count'
    )

    fig = px.bar(
        topic_votes_melted,
        x='Topic',
        y='Average Count',
        color='Vote Type',
        barmode='group',
        title='Average Votes by Topic'
    )
    fig.update_layout(xaxis_title='Topic', yaxis_title='Average Count')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit for UN Voting Data Analysis")
