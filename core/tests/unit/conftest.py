import os

# Docker installs run LiteLLM against its bundled model list. LiteLLM reads this when it is first imported, so set it
# before any test module imports litellm.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
