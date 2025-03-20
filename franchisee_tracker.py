import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import gdown
import os

# Initialize session state variables if they don't exist
if 'city' not in st.session_state:
    st.session_state.city = None
if 'category' not in st.session_state:
    st.session_state.category = ""
if 'business_name' not in st.session_state:
    st.session_state.business_name = ""
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Load the cleaned data
#df = pd.read_csv('florida_with_sentiment.csv')
# Example: Load file from Google Drive
#url = "https://drive.google.com/file/d/1oOR6oVkmkU8LQQXnDWUjvCpp0GmKjhCB/view?usp=sharing"
#df = pd.read_csv(url)

# Only download if file doesn't exist
if not os.path.exists('florida_with_sentiment.csv'):
    url = 'https://drive.google.com/file/d/1oOR6oVkmkU8LQQXnDWUjvCpp0GmKjhCB/view?usp=sharing'
    gdown.download(url, 'florida_with_sentiment.csv', quiet=False, fuzzy=True)

# Load as usual
df = pd.read_csv("florida_with_sentiment.csv", low_memory=False, encoding="utf-8")

                    
# Filter data based on city and category
def filter_data(city, category, business_name=None):
    filtered = df[(df['city'] == city) & (df['categories'].str.contains(category, case=False, na=False))]
    if business_name:
        filtered = filtered[filtered['name'].str.contains(business_name, case=False, na=False)]
    return filtered

# Function to perform sentiment analysis per review
def sentiment_analysis(reviews):
    sentiments = []
    for review in reviews:
        blob = TextBlob(review)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
    return sentiments

# Function to format postal code
def format_postal_code(postal_code):
    try:
        # Convert to integer first to remove decimals
        return str(int(float(postal_code)))
    except (ValueError, TypeError):
        return str(postal_code)

# Callback function when submit button is clicked
def on_submit():
    st.session_state.submitted = True

# Streamlit UI components
st.title("Florida Franchisee Tracker")

st.markdown("📌 **Data Source:** [Yelp Open Dataset](https://www.yelp.com/dataset)")
st.caption("The Yelp Open Dataset is a subset of Yelp data that is intended for educational use. It provides real-world data related to businesses including reviews, photos, check-ins, and attributes like hours, parking availability, and ambience.")

st.write("""
This app helps you analyze customer sentiment across different franchisee locations in Florida. 

Start by selecting a city and business category below.
""")

# City selection
st.subheader("Step 1: Select Location and Business Type")
st.write("Choose a city in Florida and specify what type of business you're interested in.")

city = st.selectbox("Select a city", df['city'].unique(), key="city_select")
st.session_state.city = city

# Category input (e.g., 'restaurant')
st.write("Enter a business category (e.g., restaurant, coffee, healthcare, etc.)")
category = st.text_input("Search for a category", value=st.session_state.category, key="category_input")
st.session_state.category = category

# Optional business name input
st.write("Optionally, you can narrow your search to a specific business name.")
business_name = st.text_input("Search for a specific business name (optional)", value=st.session_state.business_name, key="business_name_input")
st.session_state.business_name = business_name

# Button to submit the search
st.write("Once you've made your selections, click Submit to analyze the data.")
submit_button = st.button("Submit", on_click=on_submit)

# Only proceed with analysis if submitted
if st.session_state.submitted:
    if st.session_state.city and st.session_state.category:
        # Filter data based on city, category, and optional business name
        filtered_data = filter_data(st.session_state.city, st.session_state.category, st.session_state.business_name)

        if len(filtered_data) > 0:
            # Display filtered businesses
            st.success(f"Found {len(filtered_data)} businesses in {st.session_state.city} under the category '{st.session_state.category}'.")
            st.write("Now analyzing customer sentiment for these businesses...")

            # Format postal codes in the filtered data
            filtered_data['postal_code'] = filtered_data['postal_code'].apply(format_postal_code) 

            # Count how many locations each business has by unique business_id
            business_counts = filtered_data.drop_duplicates('business_id').groupby('name').size().reset_index(name='franchisee_count')
            business_counts = business_counts.sort_values('franchisee_count', ascending=False)
            
            # Display the list of businesses and how many franchisees each has
            st.subheader("Step 2: Select a Business for Detailed Analysis")
            st.write("""
            The table below shows all unique businesses matching your search and how many locations each has.
            For a meaningful franchisee analysis, we'll focus on businesses with multiple locations.
            """)

            # Display the table of unique businesses and their franchisee count
            st.dataframe(business_counts)

            # Filter to only show businesses with more than 2 stores
            multiple_franchise_businesses = business_counts[business_counts['franchisee_count'] > 2]['name'].tolist()
            
            # Initialize session state for selected business if it doesn't exist
            if 'selected_business' not in st.session_state:
                st.session_state.selected_business = None if not multiple_franchise_businesses else multiple_franchise_businesses[0]
            
            if multiple_franchise_businesses:
                st.write("Select one business from the dropdown to see detailed sentiment analysis across all its locations.")
                # Allow the user to pick a business from the filtered list
                selected_business = st.selectbox(
                    "Choose a franchisee with multiple locations to analyze:", 
                    multiple_franchise_businesses,
                    index=multiple_franchise_businesses.index(st.session_state.selected_business) if st.session_state.selected_business in multiple_franchise_businesses else 0,
                    key="business_select"
                )
                
                # Update the session state
                st.session_state.selected_business = selected_business
                
                # Filter data for selected business
                selected_business_data = filtered_data[filtered_data['name'] == selected_business]
                
                if not selected_business_data.empty:
                    st.subheader(f"Step 3: Detailed Analysis for {selected_business}")
                    st.write("""
                    Now we'll look at detailed sentiment analysis for your selected business. 
                    This will help identify which locations are performing better than others.
                    """)

                    # In the sentiment analysis section, modify to group by business_id and include address
                    sentiment_summary = selected_business_data.groupby(['business_id', 'name', 'address'])['sentiment'].agg(
                        positive_reviews=lambda x: (x > 0).sum(),
                        negative_reviews=lambda x: (x < 0).sum(),
                        neutral_reviews=lambda x: (x == 0).sum(),
                        mean_sentiment='mean',
                        sentiment_count='count'
                    ).reset_index()
                    
                    # Display sentiment summary table for selected business
                    st.write("""
                    #### Sentiment Summary Table
                    This table shows the number of positive, negative and neutral reviews for each location, 
                    along with the average sentiment score and total number of reviews.
                    
                    Higher mean sentiment scores (closer to 1.0) indicate more positive customer experiences.
                    """)
                    st.dataframe(sentiment_summary[['name', 'address', 'positive_reviews', 'negative_reviews', 
                                                  'neutral_reviews', 'mean_sentiment', 'sentiment_count']])
                    
                    # Get unique business IDs for this business
                    unique_business_ids = selected_business_data['business_id'].unique().tolist()
                    
                    # Create a mapping from business_id to address for display purposes
                    id_to_address = dict(zip(selected_business_data['business_id'], selected_business_data['address']))
                    
                    # Add an "All Locations" option
                    location_options = ["All Locations"]
                    
                    # Add the addresses corresponding to each business_id
                    location_addresses = [id_to_address.get(bid, "Unknown Address") for bid in unique_business_ids]
                    location_options.extend(location_addresses)
                    
                    # Initialize session state for selected address if it doesn't exist
                    if 'selected_address' not in st.session_state:
                        st.session_state.selected_address = "All Locations"

                                        # Create a map for just the selected business
                    st.write("""
                    #### Location Map
                    This map shows all locations for your selected business. Color indicates the sentiment at each location.
                    - **Green markers** indicate positive sentiment
                    - **Red markers** indicate negative sentiment
                    - **Gray markers** indicate neutral sentiment
            
                    Click on any marker to see the business name, postal code, and sentiment score.
                    """)
                    
                    # Initialize the map centered on the first location of the selected business
                    selected_lat = selected_business_data['latitude'].mean()
                    selected_lon = selected_business_data['longitude'].mean()
                    
                    business_map = folium.Map(location=[selected_lat, selected_lon], zoom_start=9)
                    business_marker_cluster = MarkerCluster().add_to(business_map)
                    
                    # Add markers for each location of the selected business
                    for _, row in selected_business_data.iterrows():
                        lat = row['latitude']
                        lon = row['longitude']
                        sentiment = row['sentiment']
                        postal_code = row['postal_code']
                        
                        # Color based on sentiment (positive: green, negative: red, neutral: gray)
                        if sentiment > 0:
                            color = 'green'
                        elif sentiment < 0:
                            color = 'red'
                        else:
                            color = 'gray'
                        
                        # Add marker with pop-up
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=8,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            popup=f"{selected_business} ({postal_code}): Sentiment: {sentiment:.2f}"
                        ).add_to(business_marker_cluster)
                    
                    # Render the business-specific map in Streamlit
                    st.components.v1.html(business_map._repr_html_(), height=600)
                    
                    # Dropdown to select specific location by address
                    st.subheader("Step 4: Location-Specific Analysis")
                    st.write("""
                    You can analyze sentiment for a specific location by selecting its address below.
                    
                    Choose "All Locations" to see charts for each location side by side.
                    
                    Choose specific location to see chart and details statistics
                    """)
                    
                    selected_address = st.selectbox(
                        "Choose a location by address:",
                        location_options,
                        index=location_options.index(st.session_state.selected_address) if st.session_state.selected_address in location_options else 0,
                        key="address_select"
                    )
                    
                    # Update session state
                    st.session_state.selected_address = selected_address
                    
                    # Display sentiment distribution for the selected location
                    if selected_address == "All Locations":
                        st.write("Showing sentiment distribution charts for all locations:")
                        # When "All Locations" is selected, iterate through each location and display charts
                        for i, bid in enumerate(unique_business_ids):
                            address = id_to_address.get(bid, "Unknown Address")
                            location_data = selected_business_data[selected_business_data['business_id'] == bid]
                            if len(location_data) > 0:
                                st.write(f"#### Location: {address}")
                                fig, ax = plt.subplots()
                                sns.histplot(location_data['sentiment'], bins=20, kde=True, ax=ax)
                                ax.set_title(f"Sentiment Distribution for {selected_business} ({address})")
                                ax.set_xlabel("Sentiment Score")
                                ax.set_ylabel("Review Count")
                                st.pyplot(fig)
                    else:
                        # When a specific address is selected, show only that location
                        st.write(f"Showing detailed sentiment analysis for location: {selected_address}")
                        # Find the business_id corresponding to the selected address
                        address_to_id = {v: k for k, v in id_to_address.items()}
                        selected_bid = address_to_id.get(selected_address)
                        
                        if selected_bid:
                            location_data = selected_business_data[selected_business_data['business_id'] == selected_bid]
                            if len(location_data) > 0:
                                # Add location-specific stats
                                positive_count = (location_data['sentiment'] > 0).sum()
                                negative_count = (location_data['sentiment'] < 0).sum()
                                neutral_count = (location_data['sentiment'] == 0).sum()
                                avg_sentiment = location_data['sentiment'].mean()
                                
                                st.write(f"""
                                **Location Statistics:**
                                - Positive reviews: {positive_count} ({positive_count/len(location_data)*100:.1f}%)
                                - Negative reviews: {negative_count} ({negative_count/len(location_data)*100:.1f}%)
                                - Neutral reviews: {neutral_count} ({neutral_count/len(location_data)*100:.1f}%)
                                - Average sentiment: {avg_sentiment:.2f}
                                - Total reviews: {len(location_data)}
                                """)
                                
                                fig, ax = plt.subplots()
                                sns.histplot(location_data['sentiment'], bins=20, kde=True, ax=ax)
                                ax.set_title(f"Sentiment Distribution for {selected_business} ({selected_address})")
                                ax.set_xlabel("Sentiment Score")
                                ax.set_ylabel("Review Count")
                                st.pyplot(fig)
            else:
                st.warning("No businesses with multiple franchisees found. Try broadening your search criteria.")
        else:
            st.error("No businesses found matching your criteria. Please try different search terms.")
    else:
        st.warning("Please select a city and category to begin your search.")

# Add a footer with explanations
st.markdown("---")
st.subheader("About This App")
st.write("""
**How to use this app:**
1. Select a city and business category (and optionally a specific business name)
2. Review the map showing all matching businesses
3. Select a specific business with multiple locations from the dropdown
4. Analyze sentiment across all locations or drill down into a specific location

**Understanding sentiment scores:**
- Scores range from -1.0 (very negative) to 1.0 (very positive)
- A score of 0 indicates neutral sentiment
- Scores are calculated using natural language processing on customer reviews

**What the colors mean:**
- Green: Positive sentiment (> 0)
- Red: Negative sentiment (< 0)
- Gray: Neutral sentiment (= 0)
""")
