import copy
import requests
from ._openai import OpenAI,prompt_to_messages, add_text_to_chat_mode
from typing import List, Dict
import json


def createPromptInputs(prompt: str):
    return [
        {
            "name": "prompt",
            "shape": [1],
            "datatype": "BYTES",
            "data": [prompt]
        },
    ]


def createChatInputs(messages) -> List[Dict]:
    roles = []
    contents = []
    for message in messages:
        roles.append(message["role"])
        contents.append(message["content"])
    return [
        {
            "name": "role",
            "shape": [len(roles)],
            "datatype": "BYTES",
            "data": roles
        },
        {
            "name": "content",
            "shape": [len(contents)],
            "datatype": "BYTES",
            "data": contents
        }
    ]

class Seldon(OpenAI):

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60, token=None, endpoint=None, temperature=0.0, chat_mode="auto", organization=None, rest_call=True):
        super().__init__(model, caching=caching, max_retries=max_retries,max_calls_per_min=max_calls_per_min, token=token,endpoint=endpoint, temperature=temperature,chat_mode=chat_mode, organization=organization, rest_call=rest_call)
        self.caller = self._seldon_call
        self.cache.clear()

    def _seldon_rest_stream_handler(self, response):
        yield response

    def _seldon_call(self, **kwargs):
        """ Call the Seldon API using the REST API.
        """
        # Define the request headers
        headers = copy.copy(self._rest_headers)
        #if self.token is not None:
        #    headers['Authorization'] = f"Bearer {self.token}"

        # Define the request data
        stream = kwargs.get("stream", False)
        data = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", None),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 1.0),
            "n": kwargs.get("n", 1),
            "stream": False, # we don't support streaming at present
            "logprobs": kwargs.get("logprobs", None),
            'stop': kwargs.get("stop", None),
            "echo": kwargs.get("echo", False)
        }
        if self.chat_mode:
            data["model_type"] = "chat.completions"
            # needs to be split into role and content pairs
            inputs = createPromptInputs(prompt_to_messages(kwargs["prompt"]))
            del data['prompt']
            del data['echo']
            del data['stream']
        else:
            data["model_type"] = "completions"
            inputs = createPromptInputs(kwargs["prompt"])

        inference_request = {
            "inputs": inputs,
            "parameters": {
             "llm_parameters": {
                 k:v for k,v in data.items() if k!= "prompt" and k != "messages"
             }
          }
        }

        # Send a POST request and get the response
        seldon_response = requests.post(self.endpoint, headers=headers, json=inference_request, stream=False)
        if seldon_response.status_code != 200:
            raise Exception("Response is not 200: " + seldon_response.text)
        # Convert Seldon response
        seldon_response = seldon_response.json()
        response = {}
        for output in seldon_response["outputs"]:
            if output["name"] == "output_all":
                response = json.loads(output["data"][0])
                print(response)

        if stream:
            return self._seldon_rest_stream_handler(response)

        if self.chat_mode:
            response = add_text_to_chat_mode(response)
        return response
