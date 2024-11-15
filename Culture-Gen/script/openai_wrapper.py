import openai
openai.api_key = None
import backoff
import os, json, time

class OpenAIWrapper:
    """
        Wrapper for OpenAI API.
    """
    def __init__(self, path):
        self.model_path = path
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    # @backoff.on_exception(backoff.expo, (openai.error.APIError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIConnectionError, openai.error.Timeout))
    def generate(self, prompt=None, temperature=1, max_tokens=512, top_p=1, n=1, get_logprobs=False):
        texts = []
        yes_probs = None
        if self.model_path == "text-davinci-003":
            if get_logprobs:
                yes_probs = []
                response = openai.Completion.create(
                    engine=self.model_path,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                    logprobs=5,
                )
                for choice in response["choices"]:
                    texts.append(choice["text"])
                for choice in response["choices"]:
                    yes_prob = None
                    first_index = 0
                    for i, token in enumerate(choice["logprobs"]["tokens"]):
                        if token != "\n":
                            first_index = i
                            break
                    for token in choice["logprobs"]["top_logprobs"][first_index]:
                        if token.lower().strip() == 'yes':
                            yes_prob = choice["logprobs"]["top_logprobs"][first_index][token]
                            break
                    if yes_prob is None:
                        yes_prob = float('-inf')
                    yes_probs.append(yes_prob)
            else:
                response = openai.Completion.create(
                    engine=self.model_path,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                )
                for choice in response["choices"]:
                    texts.append(choice["text"])
        else:
            response = openai.ChatCompletion.create(
                model=self.model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
            )
            for choice in response["choices"]:
                texts.append(choice['message']['content'])
        return texts, yes_probs
    
    def launch_batch(self, tasks, file_name, temperature=1, max_tokens=512, top_p=1, n=1, get_logprobs=False):
        queries = []
        for task_id, prompt in tasks:
            query = {
                "custom_id": task_id,
                "method": "POST",
                "url": "v1/chat/completions",
                "body": {
                    "model": self.model_path,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "n": n,
                }
            }
            queries.append(query)

        with open(file_name, "w") as f:
            for query in queries:
                f.write(json.dumps(query) + "\n")
        
        client = openai.OpenAI()
        batch_input_file = client.files.create(
            file=open(file_name, "rb"),
            purpose="batch"
        )

        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="v1/chat/completions",
            completion_window="24h",
        )
        
        print("Waiting for batch job to complete...")
        while 1:
            batch_job = client.batches.retrieve(batch_job.id)
            if batch_job.status == "completed":
                result_file_id = batch_job.output_file_id
                result = client.files.content(result_file_id).content

                result_file_name = file_name.replace(".jsonl", "_result.jsonl")
                with open(result_file_name, "wb") as f:
                    f.write(result)

                print("Batch job completed")
                break
            elif batch_job.status == "failed" or batch_job.status == "expired":
                raise Exception("Batch job failed")
            time.sleep(10)
        
        responses = []
        with open(result_file_name, "r") as f:
            for line in f:
                response = json.loads(line)
                responses.append(response)
        
        results = []
        for response in responses:
            task_id = response["custom_id"]
            result = response["response"]["body"]["choices"][0]["message"]["content"]
            results.append((task_id, result))

        return results