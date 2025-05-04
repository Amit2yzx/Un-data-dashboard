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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Major Powers Voting", "Resolution Analysis", "Voting Patterns", "Country Groups", "Topic Analysis"])

with tab1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Major Powers Voting Analysis</h2>", unsafe_allow_html=True)

    # Filter for General Assembly votes only
    ga_df = filtered_df[filtered_df['Council'] == 'General Assembly']

    if len(ga_df) == 0:
        st.warning("No General Assembly votes found in the selected data. Please adjust your filters.")
    else:
        st.markdown("### Voting Agreement Heatmap Between Major Powers")
        st.markdown("This heatmap shows the percentage agreement between major powers in General Assembly votes only.")

        # Define the major countries to analyze
        major_countries = ['INDIA', 'CHINA', 'UNITED STATES', 'ISRAEL', 'RUSSIAN FEDERATION', 'FRANCE', 'JAPAN', 'PAKISTAN', 'UNITED KINGDOM']

        # Check which countries are actually in the dataset
        available_countries = [country for country in major_countries if country in ga_df.columns]

        if len(available_countries) < 2:
            st.warning("Not enough major countries found in the dataset. Please check your data.")
        else:
            # Create agreement matrix
            agreement_matrix = pd.DataFrame(index=available_countries, columns=available_countries)

            for country1 in available_countries:
                for country2 in available_countries:
                    agreement_matrix.loc[country1, country2] = calculate_agreement(country1, country2, ga_df)

            # Convert to numeric values
            agreement_matrix = agreement_matrix.astype(float)

            # Create heatmap
            fig = px.imshow(
                agreement_matrix,
                text_auto='.1f',
                color_continuous_scale='RdBu_r',  # Red-Blue diverging colorscale
                color_continuous_midpoint=50,  # Set midpoint at 50%
                title='Voting Agreement Between Major Powers in General Assembly (%)',
                labels=dict(x="Country", y="Country", color="Agreement %")
            )

            fig.update_layout(
                height=600,
                xaxis=dict(tickangle=45),
                yaxis=dict(tickangle=0)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Generate insights
            insights = generate_agreement_insights(agreement_matrix)

            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### Key Insights")
            for insight in insights:
                st.markdown(f"- {insight}")

            # Add more specific insights about the major powers
            # Find countries that vote most similarly to the US
            us_agreements = agreement_matrix.loc['UNITED STATES'].sort_values(ascending=False)
            if len(us_agreements) > 1:
                us_closest_ally = us_agreements.index[1]  # Index 0 is US itself
                us_closest_agreement = us_agreements.iloc[1]
                st.markdown(f"- The United States' closest voting ally is {us_closest_ally} with {us_closest_agreement:.1f}% agreement")

            # Find countries that vote most differently from China
            if 'CHINA' in agreement_matrix.index:
                china_agreements = agreement_matrix.loc['CHINA'].sort_values()
                china_furthest = china_agreements.index[1]  # Index 0 might be a very low agreement
                china_furthest_agreement = china_agreements.iloc[1]
                st.markdown(f"- China has the lowest voting agreement with {china_furthest} at {china_furthest_agreement:.1f}%")

            st.markdown("</div>", unsafe_allow_html=True)

            # Show voting patterns for each major country
            st.markdown("### Voting Patterns of Major Powers")

            # Create a dataframe to hold vote counts for each country
            vote_patterns = pd.DataFrame()

            for country in available_countries:
                vote_counts = ga_df[country].value_counts()
                vote_patterns[country] = vote_counts

            # Fill NaN values with 0
            vote_patterns = vote_patterns.fillna(0)

            # Create a stacked bar chart
            vote_patterns_melted = vote_patterns.reset_index().melt(
                id_vars='index',
                var_name='Country',
                value_name='Count'
            )
            vote_patterns_melted.columns = ['Vote', 'Country', 'Count']

            fig = px.bar(
                vote_patterns_melted,
                x='Country',
                y='Count',
                color='Vote',
                title='Voting Patterns of Major Powers in General Assembly',
                color_discrete_map={
                    'Y': color_schemes['vote_types']['Y'],
                    'N': color_schemes['vote_types']['N'],
                    'A': color_schemes['vote_types']['A'],
                    'X': color_schemes['vote_types']['X']
                }
            )

            fig.update_layout(
                xaxis_title='Country',
                yaxis_title='Number of Votes',
                legend_title='Vote Type',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add explanation of vote types
            st.markdown("""
            **Vote Types:**
            - **Y**: Yes
            - **N**: No
            - **A**: Abstain
            - **X**: Absent
            """)

    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
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
        title='Number of Resolutions by Council',
        color_discrete_sequence=color_schemes['main_palette']
    )
    fig.update_layout(
        xaxis_title='Council',
        yaxis_title='Number of Resolutions',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # Resolutions over time
    st.markdown("### Resolutions Over Time")
    yearly_counts = filtered_df.groupby('Year').size().reset_index(name='Count')

    fig = px.line(
        yearly_counts,
        x='Year',
        y='Count',
        markers=True,
        title='Number of Resolutions by Year',
        line_shape='spline',
        render_mode='svg'
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Resolutions',
        height=450
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

    # Resolutions by topic over time
    st.markdown("### Resolution Topics Over Time")

    # Group by year and topic
    topic_year = filtered_df.groupby(['Year', 'Topic']).size().reset_index(name='Count')

    # Only include years with sufficient data
    year_counts = topic_year.groupby('Year')['Count'].sum()
    valid_years = year_counts[year_counts > 5].index
    topic_year_filtered = topic_year[topic_year['Year'].isin(valid_years)]

    # Get top topics
    top_topics = topic_year_filtered.groupby('Topic')['Count'].sum().nlargest(8).index
    topic_year_filtered = topic_year_filtered[topic_year_filtered['Topic'].isin(top_topics)]

    fig = px.line(
        topic_year_filtered,
        x='Year',
        y='Count',
        color='Topic',
        title='Top Resolution Topics Over Time',
        line_shape='spline',
        render_mode='svg'
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Resolutions',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
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
        title='Average Votes per Resolution',
        color_discrete_map={
            'YES COUNT': color_schemes['vote_types']['Yes'],
            'NO COUNT': color_schemes['vote_types']['No'],
            'ABSENT COUNT': color_schemes['vote_types']['Absent']
        }
    )
    fig.update_layout(
        xaxis_title='Vote Type',
        yaxis_title='Average Count',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # Vote distribution
    st.markdown("### Overall Vote Distribution")
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
        color_discrete_map={
            'Yes': color_schemes['vote_types']['Yes'],
            'No': color_schemes['vote_types']['No'],
            'Absent': color_schemes['vote_types']['Absent']
        },
        hole=0.4
    )
    fig.update_layout(
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Vote trends over time
    st.markdown("### Voting Trends Over Time")

    # Group by year and calculate vote percentages
    yearly_votes = filtered_df.groupby('Year')[['YES COUNT', 'NO COUNT', 'ABSENT COUNT']].sum().reset_index()
    yearly_votes['Total'] = yearly_votes['YES COUNT'] + yearly_votes['NO COUNT'] + yearly_votes['ABSENT COUNT']
    yearly_votes['Yes %'] = yearly_votes['YES COUNT'] / yearly_votes['Total'] * 100
    yearly_votes['No %'] = yearly_votes['NO COUNT'] / yearly_votes['Total'] * 100
    yearly_votes['Absent %'] = yearly_votes['ABSENT COUNT'] / yearly_votes['Total'] * 100

    # Melt for plotting
    yearly_votes_melted = pd.melt(
        yearly_votes,
        id_vars=['Year'],
        value_vars=['Yes %', 'No %', 'Absent %'],
        var_name='Vote Type',
        value_name='Percentage'
    )

    fig = px.line(
        yearly_votes_melted,
        x='Year',
        y='Percentage',
        color='Vote Type',
        title='Voting Trends Over Time',
        line_shape='spline',
        color_discrete_map={
            'Yes %': color_schemes['vote_types']['Yes'],
            'No %': color_schemes['vote_types']['No'],
            'Absent %': color_schemes['vote_types']['Absent']
        }
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Percentage of Votes',
        height=500,
        yaxis=dict(range=[0, 100])
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

    # Add insights about voting patterns
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("### Key Insights on Voting Patterns")

    # Calculate some insights
    most_yes_topic = filtered_df.groupby('Topic')[['YES COUNT']].sum().reset_index()
    most_yes_topic['Total Votes'] = filtered_df.groupby('Topic')[['TOTAL VOTES']].sum().reset_index()['TOTAL VOTES']
    most_yes_topic['Yes %'] = most_yes_topic['YES COUNT'] / most_yes_topic['Total Votes'] * 100
    most_yes_topic = most_yes_topic.sort_values('Yes %', ascending=False).iloc[0]

    most_no_topic = filtered_df.groupby('Topic')[['NO COUNT']].sum().reset_index()
    most_no_topic['Total Votes'] = filtered_df.groupby('Topic')[['TOTAL VOTES']].sum().reset_index()['TOTAL VOTES']
    most_no_topic['No %'] = most_no_topic['NO COUNT'] / most_no_topic['Total Votes'] * 100
    most_no_topic = most_no_topic.sort_values('No %', ascending=False).iloc[0]

    st.markdown(f"- Resolutions on **{most_yes_topic['Topic']}** receive the highest percentage of YES votes ({most_yes_topic['Yes %']:.1f}%)")
    st.markdown(f"- Resolutions on **{most_no_topic['Topic']}** receive the highest percentage of NO votes ({most_no_topic['No %']:.1f}%)")

    # Find year with highest consensus
    yearly_consensus = yearly_votes.copy()
    yearly_consensus['Consensus'] = yearly_consensus['Yes %']
    highest_consensus_year = yearly_consensus.sort_values('Consensus', ascending=False).iloc[0]

    st.markdown(f"- The year **{int(highest_consensus_year['Year'])}** had the highest level of consensus with {highest_consensus_year['Yes %']:.1f}% YES votes")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Country Groups Analysis</h2>", unsafe_allow_html=True)

    # Filter by selected country groups
    if selected_groups:
        filtered_by_group_df = filtered_df[filtered_df['Country_Group'].isin(selected_groups)]
    else:
        filtered_by_group_df = filtered_df

    # Count resolutions by country group
    st.markdown("### Voting Patterns by Country Group")

    # Create a dataframe to analyze voting patterns by country group
    country_columns = [col for col in df.columns if col not in ['Council', 'Date', 'Title', 'Resolution', 'TOTAL VOTES', 'NO-VOTE COUNT', 'ABSENT COUNT', 'NO COUNT', 'YES COUNT', 'Link', 'token', 'Year', 'Topic', 'Country_Group']]

    # Get vote counts by country group
    vote_by_group = {}

    for group in selected_groups:
        # Get countries in this group
        group_countries = [country for country in country_columns
                          if country in filtered_by_group_df.columns and
                          any(filtered_by_group_df[filtered_by_group_df['Country_Group'] == group][country].notna())]

        if not group_countries:
            continue

        # Initialize vote counts
        vote_counts = {'Y': 0, 'N': 0, 'A': 0, 'X': 0}
        total_votes = 0

        # Count votes for each country in the group
        for country in group_countries:
            country_votes = filtered_by_group_df[country].value_counts()
            for vote_type in vote_counts.keys():
                if vote_type in country_votes.index:
                    vote_counts[vote_type] += country_votes[vote_type]
                    total_votes += country_votes[vote_type]

        # Calculate percentages
        vote_percentages = {}
        for vote_type, count in vote_counts.items():
            if total_votes > 0:
                vote_percentages[vote_type] = (count / total_votes) * 100
            else:
                vote_percentages[vote_type] = 0

        vote_by_group[group] = vote_percentages

    # Create dataframe for visualization
    vote_group_data = []
    for group, votes in vote_by_group.items():
        for vote_type, percentage in votes.items():
            vote_group_data.append({
                'Country Group': group,
                'Vote Type': vote_type,
                'Percentage': percentage
            })

    vote_group_df = pd.DataFrame(vote_group_data)

    # Create stacked bar chart
    if not vote_group_df.empty:
        fig = px.bar(
            vote_group_df,
            x='Country Group',
            y='Percentage',
            color='Vote Type',
            title='Voting Patterns by Country Group',
            color_discrete_map=color_schemes['vote_types'],
            barmode='stack'
        )

        fig.update_layout(
            xaxis_title='Country Group',
            yaxis_title='Percentage of Votes',
            height=500,
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation of vote types
        st.markdown("""
        **Vote Types:**
        - **Y**: Yes
        - **N**: No
        - **A**: Abstain
        - **X**: Absent
        """)
    else:
        st.warning("No data available for the selected country groups.")

    # Agreement between country groups
    st.markdown("### Agreement Between Country Groups")

    # Function to calculate agreement between groups
    def calculate_group_agreement(group1, group2, data):
        # Get countries in each group
        group1_countries = [country for country in country_columns
                           if country in data.columns and
                           any(data[data['Country_Group'] == group1][country].notna())]

        group2_countries = [country for country in country_columns
                           if country in data.columns and
                           any(data[data['Country_Group'] == group2][country].notna())]

        if not group1_countries or not group2_countries:
            return 0

        # Calculate average agreement between all country pairs
        agreements = []
        for country1 in group1_countries:
            for country2 in group2_countries:
                agreement = calculate_agreement(country1, country2, data)
                agreements.append(agreement)

        if agreements:
            return sum(agreements) / len(agreements)
        else:
            return 0

    # Create agreement matrix between groups
    group_agreement_matrix = pd.DataFrame(index=selected_groups, columns=selected_groups)

    for group1 in selected_groups:
        for group2 in selected_groups:
            group_agreement_matrix.loc[group1, group2] = calculate_group_agreement(group1, group2, filtered_by_group_df)

    # Convert to numeric values
    group_agreement_matrix = group_agreement_matrix.astype(float)

    # Create heatmap
    fig = px.imshow(
        group_agreement_matrix,
        text_auto='.1f',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=50,
        title='Voting Agreement Between Country Groups (%)',
        labels=dict(x="Country Group", y="Country Group", color="Agreement %")
    )

    fig.update_layout(
        height=600,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add insights about country groups
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("### Key Insights on Country Groups")

    # Find highest and lowest agreement between groups
    max_agreement = 0
    max_pair = None
    min_agreement = 100
    min_pair = None

    for i in range(len(group_agreement_matrix.index)):
        for j in range(len(group_agreement_matrix.columns)):
            if i != j:  # Skip diagonal (self-agreement)
                group1 = group_agreement_matrix.index[i]
                group2 = group_agreement_matrix.columns[j]
                agreement = group_agreement_matrix.iloc[i, j]

                if agreement > max_agreement:
                    max_agreement = agreement
                    max_pair = (group1, group2)

                if agreement < min_agreement and agreement > 0:
                    min_agreement = agreement
                    min_pair = (group1, group2)

    if max_pair:
        st.markdown(f"- Highest agreement: **{max_pair[0]}** and **{max_pair[1]}** agree on {max_agreement:.1f}% of votes")

    if min_pair:
        st.markdown(f"- Lowest agreement: **{min_pair[0]}** and **{min_pair[1]}** agree on only {min_agreement:.1f}% of votes")

    # Find group with most Yes votes
    most_yes_group = None
    most_yes_percentage = 0

    for group, votes in vote_by_group.items():
        if 'Y' in votes and votes['Y'] > most_yes_percentage:
            most_yes_percentage = votes['Y']
            most_yes_group = group

    if most_yes_group:
        st.markdown(f"- **{most_yes_group}** has the highest percentage of YES votes ({most_yes_percentage:.1f}%)")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Topic Analysis</h2>", unsafe_allow_html=True)

    # Resolutions by topic
    st.markdown("### Resolutions by Topic")
    topic_counts = filtered_df['Topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']

    # Sort by count
    topic_counts = topic_counts.sort_values('Count', ascending=False)

    fig = px.bar(
        topic_counts,
        x='Topic',
        y='Count',
        color='Topic',
        title='Number of Resolutions by Topic',
        color_discrete_sequence=color_schemes['main_palette']
    )
    fig.update_layout(
        xaxis_title='Topic',
        yaxis_title='Number of Resolutions',
        height=500,
        xaxis={'categoryorder':'total descending'}
    )
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

    # Sort by YES COUNT
    topic_order = topic_votes.sort_values('YES COUNT', ascending=False)['Topic'].tolist()
    topic_votes_melted['Topic'] = pd.Categorical(topic_votes_melted['Topic'], categories=topic_order, ordered=True)

    fig = px.bar(
        topic_votes_melted.sort_values('Topic'),
        x='Topic',
        y='Average Count',
        color='Vote Type',
        barmode='group',
        title='Average Votes by Topic',
        color_discrete_map={
            'YES COUNT': color_schemes['vote_types']['Yes'],
            'NO COUNT': color_schemes['vote_types']['No'],
            'ABSENT COUNT': color_schemes['vote_types']['Absent']
        }
    )
    fig.update_layout(
        xaxis_title='Topic',
        yaxis_title='Average Count',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Controversial topics
    st.markdown("### Most Controversial Topics")

    # Calculate controversy score (ratio of NO to YES votes)
    topic_controversy = filtered_df.groupby('Topic')[['YES COUNT', 'NO COUNT']].sum().reset_index()
    topic_controversy['Controversy Score'] = topic_controversy['NO COUNT'] / (topic_controversy['YES COUNT'] + 1)  # Add 1 to avoid division by zero

    # Filter to topics with sufficient votes
    min_votes = 10
    topic_controversy = topic_controversy[topic_controversy['YES COUNT'] + topic_controversy['NO COUNT'] > min_votes]

    # Sort by controversy score
    topic_controversy = topic_controversy.sort_values('Controversy Score', ascending=False).head(10)

    fig = px.bar(
        topic_controversy,
        x='Topic',
        y='Controversy Score',
        color='Topic',
        title='Most Controversial Topics (Higher Score = More Controversial)',
        color_discrete_sequence=px.colors.sequential.Reds_r
    )
    fig.update_layout(
        xaxis_title='Topic',
        yaxis_title='Controversy Score',
        height=500,
        xaxis={'categoryorder':'total descending'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add insights about topics
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("### Key Insights on Topics")

    # Most common topic
    most_common_topic = topic_counts.iloc[0]['Topic']
    most_common_count = topic_counts.iloc[0]['Count']

    # Most controversial topic
    most_controversial = topic_controversy.iloc[0]['Topic']
    controversy_score = topic_controversy.iloc[0]['Controversy Score']

    # Topic with highest consensus
    topic_consensus = filtered_df.groupby('Topic')[['YES COUNT', 'TOTAL VOTES']].sum().reset_index()
    topic_consensus['Consensus %'] = (topic_consensus['YES COUNT'] / topic_consensus['TOTAL VOTES']) * 100
    topic_consensus = topic_consensus[topic_consensus['TOTAL VOTES'] > min_votes]
    highest_consensus = topic_consensus.sort_values('Consensus %', ascending=False).iloc[0]

    st.markdown(f"- **{most_common_topic}** is the most common topic with {most_common_count} resolutions")
    st.markdown(f"- **{most_controversial}** is the most controversial topic with a controversy score of {controversy_score:.2f}")
    st.markdown(f"- **{highest_consensus['Topic']}** has the highest consensus with {highest_consensus['Consensus %']:.1f}% YES votes")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Dashboard created for UN Voting Data Analysis</p>
            <p>Data source: United Nations Digital Library</p>
            <p>© 2023 - All rights reserved</p>
        </div>
        """,
        unsafe_allow_html=True
    )
