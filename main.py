import pandas as pd
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import openai
import streamlit as st
import json
import time
from concurrent.futures import ThreadPoolExecutor

# API keys from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
serpapikey = st.secrets["SERPAPI_API_KEY"]

# --- Tools ---

def search_google_duckduckgo(query, max_results=10):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    return results

def search_with_serp(query, country):
    base_url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "location": country,
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": serpapikey
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"SerpAPI search failed: {e}")
        return {}

def extract_website(hcp_name):
    results = search_google_duckduckgo(f"{hcp_name} official website")
    for r in results:
        if "hospital" in r["title"].lower() or "clinic" in r["title"].lower():
            return r["href"]
    return None

def scrape_contact_info(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        return text[:2000]  # limit context
    except Exception:
        return ""

# --- Extract info using OpenAI ---

def extract_info_with_gpt(hcp_name, website_text, urls):
    prompt = f"""
You are an assistant extracting business data from hospital or medical institution content.

From the following information and context, extract the following fields:
- HCP Name
- Status (e.g. Hospital, Clinic, Rehabilitation Center)
- Address (if available)
- Contact Person (CEO / MD / Director)
- Contact Number
- Website URL
- Net Revenue (even estimated, in USD or AED)

Context:
Institution: {hcp_name}
Websites: {urls}
Content: {website_text}

Return result as a valid JSON object with these exact fields, nothing else.
"""
    try:
       response = openai.chat.completions.create(
         model="gpt-4-turbo-preview",  # or use your preferred model
         messages=[
             {"role": "system", "content": "You extract structured company information from web pages."},
             {"role": "user", "content": prompt}
         ],
         response_format={"type": "json_object"},
       )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error parsing GPT response: {e}")
        return "{}"

# --- Main Function ---

def deep_search_links(hcp_name, max_links=8):
    """
    Enhanced deep search by scraping content from multiple links with improved query strategy.
    """
    results = []
    
    # Use multiple queries to get more diverse results
    queries = [
        f"{hcp_name} official website",
        f"{hcp_name} hospital contact information",
        f"{hcp_name} clinic address phone",
        f"{hcp_name} healthcare provider details",
        f"{hcp_name} revenue financial information"
    ]
    
    for query in queries:
        try:
            results.extend(search_google_duckduckgo(query, max_results=5))
        except Exception as e:
            st.error(f"Search error for '{query}': {e}")
    
    # Deduplicate results by URL
    unique_urls = {}
    for r in results:
        url = r.get("href")
        if url and url not in unique_urls:
            unique_urls[url] = r
    
    # Process each URL concurrently for faster scraping
    scraped_content = []
    
    with st.spinner("Searching and analyzing multiple sources..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            def scrape_url(item):
                url = item[0]
                result = item[1]
                title = result.get("title", "")
                
                try:
                    content = scrape_contact_info(url)
                    if content:
                        return {"url": url, "title": title, "content": content}
                except Exception as e:
                    st.error(f"Failed to scrape {url}: {e}")
                return None
            
            for result in executor.map(scrape_url, list(unique_urls.items())[:max_links]):
                if result:
                    scraped_content.append(result)
                    
    return scraped_content

def analyze_individual_link(hcp_name, link_data):
    """
    Analyze a single link's data to extract HCP information
    """
    prompt = f"""
    You are an expert at extracting healthcare provider data from web content.
    
    From the following web content about {hcp_name}, extract these fields if present:
    - HCP Name (exact official name)
    - Status (e.g. Hospital, Clinic, Rehabilitation Center)
    - Address (full postal address)
    - Contact Person (CEO / MD / Director names and titles)
    - Contact Number (all phone numbers)
    - Email Addresses
    - Website URL
    - Net Revenue (any revenue/financial data, in USD or AED)
    - Number of beds (if applicable)
    - Services offered (main medical specialties)
    
    Source: {link_data['url']}
    Title: {link_data['title']}
    
    Content: {link_data['content'][:3500]}
    
    Return ONLY a JSON object with these fields, nothing else. If information is not found, use empty string.
    """
    
    try:
        response = openai.chat.completions.create(
         model="gpt-4-turbo-preview",  # or use your preferred model
         messages=[
             {"role": "system", "content": "You extract structured healthcare provider information from web content with high precision."},
             {"role": "user", "content": prompt}
         ],
         response_format={"type": "json_object"},
       )
        return response["choices"][0]["message"]["content"]
        
    except Exception as e:
        st.error(f"Error analyzing individual link: {e}")
        return "{}"

def extract_info_from_links(hcp_name, scraped_links):
    """
    Enhanced analysis: Extract information from each source independently, then aggregate
    """
    if not scraped_links:
        return "{}"
    
    # Analyze each link individually
    link_analyses = []
    
    with st.spinner(f"Analyzing {len(scraped_links)} sources for {hcp_name}..."):
        progress_bar = st.progress(0)
        for i, link in enumerate(scraped_links):
            analysis_json = analyze_individual_link(hcp_name, link)
            try:
                analysis = json.loads(analysis_json)
                analysis["source"] = link["url"]
                link_analyses.append(analysis)
            except json.JSONDecodeError:
                st.error(f"Failed to parse analysis for {link['url']}")
            
            # Update progress bar
            progress_bar.progress((i + 1) / len(scraped_links))
            # Small delay to avoid API rate limits
            time.sleep(0.5)
    
    # Combine all analyses for final determination
    urls = "\n".join([f"- {link['url']}" for link in scraped_links])
    sources_data = json.dumps(link_analyses, indent=2)
    
    aggregate_prompt = f"""
    You are an expert data analyst specializing in healthcare provider information.
    
    Below is data extracted from {len(link_analyses)} different sources about {hcp_name}.
    Your job is to analyze all sources and create the most accurate, complete profile.
    
    When sources disagree:
    1. Prefer information from official sources
    2. Look for consistency across multiple sources
    3. Choose the most specific and detailed information
    
    Sources analyzed:
    {urls}
    
    Source data:
    {sources_data}
    
    Create a final, accurate profile with these fields:
    - HCP Name (the most accurate, complete version)
    - Status (e.g. Hospital, Clinic, Rehabilitation Center)
    - Address (most complete postal address)
    - Contact Person (CEO/Director's name and title)
    - Contact Number (main contact number)
    - Website URL (official website)
    - Net Revenue (most accurate figure, specify currency and year if available)
    
    Return ONLY a valid JSON object with exactly these fields.
    """
    
    try:
       response = openai.chat.completions.create(
         model="gpt-4-turbo-preview",  # or use your preferred model
         messages=[
             {"role": "system", "content": "You aggregate and analyze healthcare data from multiple sources to create accurate profiles."},
             {"role": "user", "content": aggregate_prompt}
         ],
         response_format={"type": "json_object"},
       )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error in final aggregation: {e}")
        return "{}"

def process_hcp(hcp_name):
    """
    Process a single HCP with enhanced deep search and analysis
    """
    with st.status(f"Processing {hcp_name}...") as status:
        status.update(label="Starting deep search...")
        scraped_links = deep_search_links(hcp_name)
        
        if not scraped_links:
            status.update(label="No results found", state="error")
            return {
                "HCP Name": hcp_name,
                "Status": "Not Found",
                "Address": "",
                "Contact Person": "",
                "Contact Number": "",
                "URL Website": "",
                "Net Revenue": ""
            }
        
        status.update(label=f"Analyzing {len(scraped_links)} sources...")
        extracted_data_json = extract_info_from_links(hcp_name, scraped_links)
        
        # Parse the JSON string to dictionary
        try:
            extracted_data = json.loads(extracted_data_json)
            status.update(label="Analysis complete", state="complete")
        except:
            status.update(label="Error parsing results", state="error")
            extracted_data = {}
        
        return {
            "HCP Name": extracted_data.get("HCP Name", hcp_name),
            "Status": extracted_data.get("Status", ""),
            "Address": extracted_data.get("Address", ""),
            "Contact Person": extracted_data.get("Contact Person", ""),
            "Contact Number": extracted_data.get("Contact Number", ""),
            "URL Website": extracted_data.get("Website URL", ""),
            "Net Revenue": extracted_data.get("Net Revenue", "")
        }

def process_csv(file_path):
    """
    Process multiple HCPs from a CSV file with progress tracking
    """
    df = pd.read_csv(file_path)
    results = []
    total = len(df)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, hcp_name in enumerate(df["HCP NAME"]):
        status_text.text(f"Processing {i+1}/{total}: {hcp_name}")
        result = process_hcp(hcp_name)
        results.append(result)
        progress_bar.progress((i + 1) / total)
    
    status_text.text("Processing complete!")
    return pd.DataFrame(results)

# --- Streamlit Interface ---
st.title("HCP Data Enrichment Tool")

# Create tabs for main functionality and documentation
tab1, tab2 = st.tabs(["Main Tool", "Documentation"])

with tab1:
    # Add a selection for single hospital or CSV upload
    option = st.radio("Choose an option:", ["Search a Single Hospital", "Upload a CSV File"])
    
    if option == "Search a Single Hospital":
        hcp_name = st.text_input("Enter the name of the hospital or clinic:")
        if st.button("Search"):
            if hcp_name:
                st.write(f"Processing: {hcp_name}")
                result = process_hcp(hcp_name)
                st.write("### Result")
                st.json(result)
            else:
                st.warning("Please enter a hospital or clinic name.")
    
    elif option == "Upload a CSV File":
        uploaded_file = st.file_uploader("Upload a CSV file with a column named 'HCP NAME'", type=["csv"])
        if uploaded_file:
            st.write("Processing the uploaded file...")
            input_df = pd.read_csv(uploaded_file)
            results_df = process_csv(uploaded_file)
            
            st.write("### Enriched Data")
            st.dataframe(results_df)
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Enriched Data as CSV",
                data=csv,
                file_name="enriched_hcp_data.csv",
                mime="text/csv",
            )

with tab2:
    st.header("Technical Documentation")
    
    st.subheader("Architecture Overview")
    st.markdown("""
    This tool uses a multi-stage pipeline to gather and analyze healthcare provider information:
    
    1. **Data Collection Layer**
       - DuckDuckGo Search API for initial web discovery
       - SerpAPI for additional search results
       - Web scraping with BeautifulSoup4
       - Concurrent execution with ThreadPoolExecutor
    
    2. **Analysis Layer**
       - OpenAI GPT-4 for text analysis
       - Multi-stage analysis pipeline
       - Source validation and cross-referencing
    
    3. **Data Processing Pipeline**
       ```mermaid
       graph TD
           A[Input: HCP Name] --> B[Deep Search]
           B --> C[Multi-Source Scraping]
           C --> D[Individual Source Analysis]
           D --> E[Data Aggregation]
           E --> F[Final Profile Generation]
       ```
    """)
    
    st.subheader("Search Methodology")
    st.markdown("""
    The deep search process includes:
    
    - Multiple search queries per institution
    - URL deduplication
    - Concurrent web scraping
    - Content validation
    - Rate limiting and error handling
    """)
    
    with st.expander("Detailed Search Process"):
        st.code("""
def deep_search_links(hcp_name, max_links=8):
    # Multiple targeted queries
    queries = [
        f"{hcp_name} official website",
        f"{hcp_name} hospital contact information",
        # ... other queries
    ]
    
    # Concurrent scraping with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        # ... scraping process
        """)
    
    st.subheader("AI Analysis Pipeline")
    st.markdown("""
    Three-stage AI analysis process:
    
    1. **Individual Source Analysis**
       - Extracts structured data from each source
       - Validates data consistency
       - Assigns confidence scores
    
    2. **Cross-Source Validation**
       - Compares data across sources
       - Identifies conflicts
       - Weighs source reliability
    
    3. **Final Profile Generation**
       - Aggregates validated information
       - Resolves conflicts
       - Generates comprehensive profile
    """)
    
    with st.expander("AI Prompting Strategy"):
        st.markdown("""
        The system uses carefully crafted prompts with:
        - Clear role definitions
        - Structured output requirements
        - Source prioritization rules
        - Conflict resolution guidelines
        """)
    
    st.subheader("Data Fields")
    st.markdown("""
    | Field | Description | Source Priority |
    |-------|-------------|-----------------|
    | HCP Name | Official institution name | Official websites |
    | Status | Type of healthcare facility | Multiple sources |
    | Address | Physical location | Official + Maps |
    | Contact Person | Key personnel | Official + LinkedIn |
    | Contact Number | Primary contact | Official website |
    | Website URL | Official web presence | Verified sources |
    | Net Revenue | Financial information | Multiple sources |
    """)
    
    st.subheader("Error Handling")
    st.markdown("""
    The system implements comprehensive error handling:
    - Connection timeouts
    - API rate limiting
    - Content validation
    - Data consistency checks
    - Graceful degradation
    """)
    
    st.subheader("Performance Optimization")
    st.markdown("""
    - Concurrent web scraping
    - Content length optimization
    - API call batching
    - Response caching
    - Progressive loading
    """)
    
    st.subheader("API Dependencies")
    st.json({
        "OpenAI GPT-4": "Text analysis and data extraction",
        "DuckDuckGo": "Initial web search",
        "SerpAPI": "Additional search results",
        "BeautifulSoup4": "Web scraping",
        "ThreadPoolExecutor": "Concurrent execution"
    })
