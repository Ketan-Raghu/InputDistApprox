from vllm import LLM, SamplingParams

MODEL_PATH = "/home/ketan/LLMs/models/Qwen_Qwen2.5-7B-Instruct/"

llm = LLM(model=MODEL_PATH)

sampling_params = SamplingParams(temperature=0.7, top_p=0.5, max_tokens=1024)
7
prompts = [
    "You are a helpful assistant\n\n<PROMPT>Explain what a neural network is in one paragraph.</PROMPT>\n\n<RESPONSE>",
    "You are a helpful assistant\n\n<PROMPT>Write a short Python function that reverses a string.</PROMPT>\n\n<RESPONSE>",
    "You are a helpful assistant\n\n<PROMPT>What is the capital of France?</PROMPT>\n\n<RESPONSE>",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Output: {output.outputs[0].text}")
    print("-" * 60)
