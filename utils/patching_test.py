import json
import os
from typing import List as TypeList
from typing import Optional

import openai
from dotenv import load_dotenv
from patching import MakePatchSchema
from pydantic import BaseModel, Field

# Example usage
if __name__ == "__main__":
    load_dotenv(override=True)

    class Person(BaseModel):
        name: str
        age: Optional[int] = None
        tags: TypeList[str] = Field(default_factory=list)

    # Create an instance of the model
    person = Person(name="Arnav", age=30, tags=["developer", "python"])

    # Create a patch schema for the Person model
    PersonPatchSchema = MakePatchSchema(Person)
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Print the JSON schema for debugging
    print(json.dumps(PersonPatchSchema.model_json_schema(), indent=2))

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates JSON Patch operations."},
            {
                "role": "user",
                "content": f"""Generate a JSON Patch operation to change the name of the person to 'Ada'
                  and remove the age field. The original data is: {person.model_dump()}""",
            },
        ],
        response_format=PersonPatchSchema,
    )
    response_json = response.choices[0].message.parsed
    print(response_json)
    patched_person = response_json.apply(person)
    print(patched_person)
