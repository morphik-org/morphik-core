import os
import json
from typing import List, Dict, Optional
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig, Controller
from browser_use.browser.context import BrowserContext
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Get all JSON files in samples/results
source_dir = "samples/results"
json_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]

companies: List[Dict] = []
for f in json_files:
    with open(os.path.join(source_dir, f), "r") as file:
        data = json.load(file)
    companies.append(data)
    print(f"Loaded company: {data['company_name']}")

load_dotenv(override=True)


class ContactDetails(BaseModel):
    name: str
    linkedin_url: Optional[str]
    contact_info: Optional[dict]


class CompanyProfileWithDetails(BaseModel):
    company_name: str
    contacts: list[ContactDetails]


async def get_contact_details(person_name: str, company_name: str, model: str = "gemini"):
    config = BrowserConfig(
        chrome_instance_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        headless=False,
    )
    browser = Browser(config=config)
    browser_context_config = BrowserContextConfig(
        maximum_wait_page_load_time=10,
        wait_between_actions=5,
    )
    browser_context = BrowserContext(browser=browser, config=browser_context_config)

    controller = Controller(output_model=ContactDetails)

    if model == "gemini":
        LLM = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            api_key=SecretStr(os.getenv("GEMINI_API_KEY"))
        )
    else:  # gpt4
        LLM = ChatOpenAI(
            model="gpt-4o",
            api_key=SecretStr(os.getenv("OPENAI_API_KEY"))
        )

    print(f"Getting details for {person_name} at {company_name}")

    search_query = f"{person_name} {company_name}"

    agent = Agent(
        task=f"""Go to linkedin.com. Search for '{search_query}' on linkedin.com - do not use google. Click on the first profile result that matches both the person and company.
                Get their LinkedIn profile URL. Click on 'Contact info' if available and collect all contact information.
                Return the person's name, LinkedIn URL, and any contact information found in a structured format.""",
        llm=LLM,
        browser_context=browser_context,
        controller=controller,
    )

    result = await agent.run()
    final_result = result.final_result()
    print(f"Obtained final result: {final_result}")
    return final_result


results_dir = "samples/processed_results"
results_dir_path = Path(results_dir)
results_dir_path.mkdir(parents=True, exist_ok=True)


async def process_companies():
    companies_to_process = [company for company in companies if company.get("contacts")]

    for company in companies_to_process:
        company_name = company["company_name"]

        # Check if company already processed
        result_file = results_dir_path / f"{company_name}_contacts.json"
        if result_file.exists():
            print(f"Skipping {company_name} - already processed")
            continue

        updated_contacts = []
        for contact in company["contacts"]:
            try:
                contact_details = await get_contact_details(contact["name"], company_name)
                updated_contacts.append(contact_details)
            except Exception as e:
                print(f"Error getting details for {contact['name']} at {company_name}: {str(e)}")
                continue

        # Update company with new contact details
        company["contacts"] = updated_contacts

        # Save updated company info
        with open(result_file, "w") as f:
            json.dump(company, f, indent=2)

        print(f"Updated contact details for {company_name}")


# Run the async process
asyncio.run(process_companies())
