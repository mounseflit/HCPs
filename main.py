import pandas as pd
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import openai
import streamlit as st

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

def extract_info_with_gpt(hcp_name, website_text, url):
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
Website: {url}
Content: {website_text}

ADDITIONAL DATA : {search_with_serp(hcp_name, "United Arab Emirates")}

Return result as a valid JSON object with these exact fields, nothing else.
"""
    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",  # or use your preferred model
        messages=[
            {"role": "system", "content": "You extract structured company information from web pages."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    )
    
    try:
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error parsing GPT response: {e}")
        return "{}"

# --- Main Function ---

def deep_search_links(hcp_name, max_links=5):
    """
    Perform a deep search by scraping content from multiple links.
    """
    results = search_google_duckduckgo(f"{hcp_name} hospital or clinic")
    scraped_content = []
    
    for r in results[:max_links]:
        url = r.get("href")
        if url:
            try:
                print(f"Scraping URL: {url}")
                content = scrape_contact_info(url)
                if content:
                    scraped_content.append({"url": url, "content": content})
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
    
    return scraped_content

def extract_info_from_links(hcp_name, scraped_links):
    """
    Extract information from the scraped content of multiple links.
    """
    combined_content = " ".join([link["content"] for link in scraped_links])
    urls = ", ".join([link["url"] for link in scraped_links])
    
    return extract_info_with_gpt(hcp_name, combined_content, urls)

def process_hcp(hcp_name):
    """
    Process a single HCP by performing a deep search and extracting information.
    """
    scraped_links = deep_search_links(hcp_name)
    if not scraped_links:
        return {
            "HCP Name": hcp_name,
            "Status": "Not Found",
            "Address": "",
            "Contact Person": "",
            "Contact Number": "",
            "URL Website": "",
            "Net Revenue": ""
        }
    
    extracted_data_json = extract_info_from_links(hcp_name, scraped_links)
    
    # Parse the JSON string to dictionary
    import json
    try:
        extracted_data = json.loads(extracted_data_json)
    except:
        extracted_data = {}
    
    return {
        "HCP Name": extracted_data.get("HCP Name", hcp_name),
        "Status": extracted_data.get("Status", ""),
        "Address": extracted_data.get("Address", ""),
        "Contact Person": extracted_data.get("Contact Person", ""),
        "Contact Number": extracted_data.get("Contact Number", ""),
        "URL Website": ", ".join([link["url"] for link in scraped_links]),
        "Net Revenue": extracted_data.get("Net Revenue", "")
    }


def process_csv(input_df):
    results = []
    for hcp_name in input_df["HCP NAME"]:
        print(f"Processing: {hcp_name}")
        result = process_hcp(hcp_name)
        print(result)
        results.append(result)
    return pd.DataFrame(results)


# --- Streamlit Interface ---
st.title("HCP Data Enrichment Tool")

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
        results_df = process_csv(input_df)
        
        st.write("### Enriched Data")
        st.dataframe(results_df)
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Enriched Data as CSV",
            data=csv,
            file_name="enriched_hcp_data.csv",
            mime="text/csv",
        )
