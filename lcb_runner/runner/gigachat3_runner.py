from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class GigaChat3Runner(BaseRunner):
    # Class-level client to avoid pickling issues with multiprocessing
    client = OpenAI(
        api_key="pass",  # No API key required
        base_url="https://grigoriyklopov50--example-vllm-inference-serve-dev.modal.run/v1",
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        
        self.client_kwargs: dict[str | str] = {
            "model": "ai-sage/GigaChat3-10B-A1.8B",  # Actual model name for the API
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": args.n,
            "timeout": args.openai_timeout,
        }

    def _run_single(self, prompt: list[dict[str, str]], n: int = 10) -> list[str]:
        assert isinstance(prompt, list)

        if n == 0:
            print("Max retries reached. Returning empty response.")
            return [""] * self.args.n

        try:
            response = GigaChat3Runner.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 10 seconds...")
            sleep(10)
            return self._run_single(prompt, n=n - 1)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        
        return [c.message.content for c in response.choices]
