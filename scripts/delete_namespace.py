import os

import turbopuffer
from dotenv import load_dotenv

load_dotenv()

tpuf = turbopuffer.Turbopuffer(
    region="aws-us-west-2",  # pick the right region: https://turbopuffer.com/docs/regions
    api_key=os.getenv("TURBOPUFFER_API_KEY"),
)

ns = tpuf.namespace("example")
# If an error occurs, this call raises a turbopuffer.APIError if a retry was not successful.
result = ns.delete_all()

print(result)
