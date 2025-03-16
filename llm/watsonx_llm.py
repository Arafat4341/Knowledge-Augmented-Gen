from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
import os
from dotenv import load_dotenv

def load_watsonx_llm():
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))

    model_id = 'meta-llama/llama-3-3-70b-instruct'
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5
    }
    credentials = {
        "url": os.getenv("IBM_URL"),
        "apikey": os.getenv("IBM_API_KEY")
    }
    project_id = os.getenv("IBM_PROJECT_ID")

    model = Model(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)
    return WatsonxLLM(model=model)
