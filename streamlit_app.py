import json
import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from sheets_parser import find_company_people, process_leads
from url_getter import get_contact_details

# Load environment variables
load_dotenv(override=True)

# Initialize session state variables
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "results" not in st.session_state:
    st.session_state.results = []

st.title("DataBridge LinkedIn Pipeline")

# Input method selection
input_method = st.tabs(["Upload Excel", "Manual Input"])

with input_method[0]:
    # File upload section
    st.subheader("1. Upload Excel File")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file:
        try:
            # Load Excel file and get sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names

            # Sheet selection
            default_sheet = "Filtered Leads" if "Filtered Leads" in sheet_names else sheet_names[0]
            selected_sheet = st.selectbox(
                "Select sheet", options=sheet_names, index=sheet_names.index(default_sheet)
            )

            # Process the leads using the modified function
            try:
                company_names, company_to_row = process_leads(excel_file, selected_sheet)
                companies = company_names.dropna().unique().tolist()
                
                # Show data preview
                st.write("Preview of company data:")
                company_preview = pd.DataFrame({"Company": companies})
                st.dataframe(company_preview.head())

                total_companies = len(companies)
                st.info(f"Found {total_companies} unique companies to process")

            except KeyError as e:
                st.error(f"Selected sheet must contain a 'Company' column. Error: {str(e)}")
            except Exception as e:
                st.error(f"Error processing sheet: {str(e)}")

        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")

with input_method[1]:
    # Manual input section
    st.subheader("1. Enter Company Names")
    manual_input = st.text_area(
        "Enter company names (one per line)",
        height=200,
        help="Enter each company name on a new line"
    )
    
    if manual_input:
        # Process manual input
        companies = [company.strip() for company in manual_input.split('\n') if company.strip()]
        total_companies = len(companies)
        
        if total_companies > 0:
            st.info(f"Found {total_companies} companies to process")
            
            # Show data preview
            st.write("Preview of company data:")
            company_preview = pd.DataFrame({"Company": companies})
            st.dataframe(company_preview.head())

# Common processing section (shown if companies are available from either method)
if 'companies' in locals() and companies:
    # Add number input for company limit
    num_companies = st.number_input(
        "Number of companies to process (0 for all)",
        min_value=0,
        max_value=total_companies,
        value=min(5, total_companies),  # Default to 5 or max available
        help="Set to 0 to process all companies"
    )

    # Process button
    if st.button("Start Processing"):
        # Apply company limit if specified
        if num_companies > 0:
            companies = companies[:num_companies]
            st.info(f"Processing first {num_companies} companies")
        
        # Create necessary directories
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("processed_results").mkdir(parents=True, exist_ok=True)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Rest of your existing processing code...
        async def process_companies():
            total = len(companies)
            for i, company in enumerate(companies):
                try:
                    # Update status
                    status_text.text(f"Processing company {i+1}/{total}: {company}")
                    st.session_state.current_company = company
                    
                    # Check if already processed
                    result_file = Path(f"results/{company}_contacts.json")
                    if result_file.exists():
                        try:
                            with open(result_file) as f:
                                result = f.read()
                                if result.strip():  # Check if file is not empty
                                    data = json.loads(result)
                                    st.info(f"Using cached results for {company}")
                                else:
                                    # If file is empty, process company again
                                    result = await find_company_people(company)
                                    data = json.loads(result)
                        except (json.JSONDecodeError, FileNotFoundError):
                            # If file is corrupted or missing, process company again
                            result = await find_company_people(company)
                            data = json.loads(result)
                    else:
                        # Step 1: Find company people
                        result = await find_company_people(company)
                        data = json.loads(result)
                        
                        # Save initial results
                        with open(result_file, "w") as f:
                            f.write(result)
                    
                    # Step 2: Get detailed contact info if contacts were found
                    if data.get("contacts"):
                        for contact in data["contacts"]:
                            try:
                                contact_details = await get_contact_details(
                                    contact["name"], 
                                    company
                                )
                                # Update contact with details
                                contact.update(contact_details)
                            except Exception as e:
                                st.warning(f"Could not get details for {contact['name']}: {str(e)}")
                    
                    # Save final results
                    st.session_state.results.append(data)
                    
                    # Update progress
                    progress = (i + 1) / total
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error processing {company}: {str(e)}")
                    continue
            
            st.session_state.processing_complete = True
            status_text.text("Processing complete!")

        # Run the async processing
        asyncio.run(process_companies())

# Display results section
if st.session_state.results:
    st.subheader("2. Results")

    # Display summary statistics
    total_companies = len(st.session_state.results)
    companies_with_contacts = sum(1 for r in st.session_state.results if r["contacts"])
    total_contacts = sum(len(r["contacts"]) for r in st.session_state.results)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Companies", total_companies)
    col2.metric("Companies with Contacts", companies_with_contacts)
    col3.metric("Total Contacts Found", total_contacts)

    # Display detailed results
    st.subheader("Detailed Results")
    for company_data in st.session_state.results:
        with st.expander(
            f"{company_data['company_name']} ({len(company_data['contacts'])} contacts)"
        ):
            if company_data["contacts"]:
                for contact in company_data["contacts"]:
                    st.write(f"📎 {contact['name']}")
                    if "linkedin_url" in contact:
                        st.write(f"🔗 {contact['linkedin_url']}")
                    if "contact_info" in contact:
                        st.write("📧 Contact Info:", contact["contact_info"])
            else:
                st.write("No contacts found")

    # Export results button
    if st.button("Export Results"):
        # Convert results to DataFrame for easy export
        export_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"linkedin_results_{export_time}.json"
        with open(filename, "w") as f:
            json.dump(st.session_state.results, f, indent=2)
        st.success(f"Results exported to {filename}")
