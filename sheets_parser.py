import json
from pathlib import Path
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig
from browser_use.browser.context import BrowserContext
import asyncio
import os
from dotenv import load_dotenv
from browser_use import Controller

load_dotenv(override=True)

def load_spreadsheet(file_path: str) -> pd.DataFrame:
    """
    Load a spreadsheet file (CSV, Excel, etc.) into a pandas DataFrame
    
    Args:
        file_path (str): Path to the spreadsheet file
        
    Returns:
        pd.DataFrame: Loaded spreadsheet data
    """
    # Get file extension
    file_ext = file_path.split('.')[-1].lower()
    
    try:
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError, PermissionError) as e:
        raise ValueError(f"Error loading spreadsheet: {str(e)}")
    

def process_leads(file_path: str) -> pd.DataFrame:
    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)
    # Get list of sheet names
    # sheet_names = excel_file.sheet_names
    # print("Available sheets:", sheet_names)
    # Load first sheet as default
    filtered_leads = excel_file.parse('Filtered Leads')

    first_row = filtered_leads.iloc[0].to_list()
    row_name_to_index = {col : i for i, col in enumerate(first_row)}
    # print("First row:", first_row)
    company_names = filtered_leads.iloc[:, row_name_to_index['Company']]
    company_to_row = {company : i for i, company in enumerate(company_names)}
    # print("Company names:", company_names.to_list())

    return company_names, company_to_row


LINKEDIN_USERNAME = 'databridgesuperuser@gmail.com'
LINKEDIN_PASSWORD = 'DataBridge@1'

from pydantic import BaseModel, SecretStr

class Contact(BaseModel):
    name: str
    # linkedin_url: str

class CompanyProfile(BaseModel):
    company_name: str
    contacts: list[Contact]

async def find_company_people(company_name: str):
    config = BrowserConfig(
        chrome_instance_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        headless=True
    )
    browser = Browser(config=config)
    # browser_context_config = BrowserContextConfig(
    #     allowed_domains=['linkedin.com'],
    # )
    browser_context = BrowserContext(browser=browser) #, config=browser_context_config)
    
    contorller = Controller(
        output_model=CompanyProfile
    )

    LLM = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash-exp', api_key=SecretStr(os.getenv('GEMINI_API_KEY'))
    )# ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o")
    
    agent = Agent(
        task=f"Got to linkedin.com. Search for {company_name} and select the first result - the result may not match exactly what the company name is, but as long as the result is a company, it's ok to select that one. Go to this company's linkedin page. Go to the people section and find 3 people - preferably executives - working at the company. Try to avoid names such as 'LinkedIn Member' or non-human names. Return the names of these people. If you can't find people within the first two pages, return an empty list of people - that's fine. There is a chance that the company doesn't have any people working there or that the name of the company is not correctly spelled.",
        llm=LLM,
        browser_context=browser_context,
        controller=contorller
    )
    result = await agent.run()
    # print(result)
    final = result.final_result()
    return final

async def main():
    company_names, company_to_row = process_leads("samples/trial.xlsx")
    for company in company_names[300:401]:
        # Check if result file already exists
        results_dir = Path("samples/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        result_file = results_dir / f"{company}_contacts.json"
        
        if result_file.exists():
            print(f"Skipping {company} - results already exist")
            continue
            
        try:
            result = await find_company_people(company)
            result = result if result else json.dumps({"company_name": company, "contacts": []})
        except Exception as e:
            print(f"Error while finding company people for {company}: {str(e)}")
            result = json.dumps({"company_name": company, "contacts": []})
        
        # try:
        #     final_typed : CompanyProfile = CompanyProfile.model_validate_json(result)
        #     print(f"{company}: {final_typed}")
        # except Exception as e:
        #     print(f"Error validating result for {company}")
        #     result.model_dump_json()
        #     result = CompanyProfile(company_name=company, contacts=[])
        
        # Save result to JSON file
        with open(result_file, "w") as f:
            f.write(result)


# this works, but might be better to split this up into different workflows - 
# 1st to find the relevant people for each fo these companies (one-shot)
# second, for each person found, list their profile details through linkedin such as url, name
# third, craft personal message for each linkedin url (this could be merged into 2, but i'm not completely sure)

# ask about if the're comfortable running it on their own machine - liek if they're ok to run this on their linkedin accounts (we can run it to, but will need their passwrods etc.)
# .... 

asyncio.run(main())