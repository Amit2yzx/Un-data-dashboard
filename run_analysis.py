import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load the UN voting data
file_path = 'UN DATA.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Check column names and data types
print("\nColumn names:")
for col in df.columns:
    print(f"- {col}")

print("\nData types:")
print(df.dtypes)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Distribution of votes by council
council_counts = df['Council'].value_counts()
plt.figure(figsize=(10, 6))
council_counts.plot(kind='bar')
plt.title('Number of Resolutions by Council')
plt.xlabel('Council')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('council_counts.png')
print("\nSaved plot: council_counts.png")

# Analyze voting patterns (YES, NO, ABSENT, etc.)
vote_columns = ['YES COUNT', 'NO COUNT', 'ABSENT COUNT']
vote_data = df[vote_columns]

# Calculate average votes per resolution
avg_votes = vote_data.mean()
plt.figure(figsize=(10, 6))
avg_votes.plot(kind='bar')
plt.title('Average Votes per Resolution')
plt.xlabel('Vote Type')
plt.ylabel('Average Count')
plt.tight_layout()
plt.savefig('avg_votes.png')
print("\nSaved plot: avg_votes.png")

# Select a few major countries to analyze
countries = ['UNITED STATES', 'RUSSIAN FEDERATION', 'CHINA', 'UNITED KINGDOM', 'FRANCE']

# Function to count vote types for a country
def count_vote_types(country):
    vote_counts = df[country].value_counts()
    return vote_counts

# Create a plot for each country
plt.figure(figsize=(15, 10))
for i, country in enumerate(countries, 1):
    plt.subplot(2, 3, i)
    vote_counts = count_vote_types(country)
    vote_counts.plot(kind='bar')
    plt.title(f'Voting Pattern: {country}')
    plt.xlabel('Vote Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('country_voting_patterns.png')
print("\nSaved plot: country_voting_patterns.png")

# Function to calculate agreement percentage between two countries
def calculate_agreement(country1, country2):
    # Filter rows where both countries have valid votes (not NaN)
    valid_votes = df.dropna(subset=[country1, country2])

    # Count agreements
    agreements = (valid_votes[country1] == valid_votes[country2]).sum()
    total = len(valid_votes)

    if total > 0:
        return (agreements / total) * 100
    else:
        return 0

# Create a matrix of agreement percentages
agreement_matrix = pd.DataFrame(index=countries, columns=countries)

for country1 in countries:
    for country2 in countries:
        agreement_matrix.loc[country1, country2] = calculate_agreement(country1, country2)

# Convert to numeric values
agreement_matrix = agreement_matrix.astype(float)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Voting Agreement Between Countries (%)')
plt.tight_layout()
plt.savefig('voting_agreement.png')
print("\nSaved plot: voting_agreement.png")

# Extract keywords from resolution titles
def extract_keywords(title):
    keywords = ['nuclear', 'human rights', 'peace', 'security', 'economic', 'development', 'terrorism']
    title_lower = title.lower()
    for keyword in keywords:
        if keyword in title_lower:
            return keyword
    return 'other'

# Apply the function to create a new column
df['Topic'] = df['Title'].apply(extract_keywords)

# Count resolutions by topic
topic_counts = df['Topic'].value_counts()

# Plot
plt.figure(figsize=(12, 6))
topic_counts.plot(kind='bar')
plt.title('Number of Resolutions by Topic')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('topic_counts.png')
print("\nSaved plot: topic_counts.png")

# Extract year from the Date column
df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year

# Count resolutions by year
yearly_counts = df['Year'].value_counts().sort_index()

# Plot
plt.figure(figsize=(15, 6))
yearly_counts.plot(kind='line', marker='o')
plt.title('Number of Resolutions by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('yearly_counts.png')
print("\nSaved plot: yearly_counts.png")

print("\nAnalysis complete. All plots have been saved as PNG files.")
