# Caveats to Training Setup

### Domain Mismatch

Typical distillation handles $F: X -> Y$ distilling into $F': X -> Y$ whereas this study is trying to predict the input distribution meaning we're distilling into $F^{-1}: Y -> X$. This means we cannot take advantage of dark knowledge and logit information typically used in distillation and have to rely on my direct input:response (text:text) pairs in order to apply the same model since the current input embeddings expect text rather than logit vectors.

The same information can likely be captured by running each prompt $n$ times with a higher temperature to gain a sense of variance by prompt. While this increases data generation and training times both by a factor of $n$ for the same number of prompts, it may be worthwhile for the model to map the output space better.

### Switching from LLaMA-3.1 to Qwen2.5

- LLaMA was optimized for only Western languages, but trained on other languages including Chinese
- Qwen was explicitly optimized for English and Chinese meaning the Chinese format detection in English languages that arose in LLaMA allowing the prompt to be reverse engineered being in Chinese may not arise, or may arise differently
    - Good and bad because it means it's not as close of a parallel, but frontier models are trained on many more languages so this is more generalizable

### Scaling into Reasoning-Capable models
- To run these experiments on reasoning capable models we need to fully understand the reasoning chain/scaffold being used and be able to invert it + recursively counter the reasoning
- Would need to also consider system prompts, but that's not that bad because they can easily be included in the input conditioning or the final output
- Need to investigate scaffolding and conditioning more (is it better to generate the input autoregressively based on the output + conditioned on the system prompt?)
    - Currently not using a system prompt
- Need to analyze prompting default scaffold more (ideally just remove any preset scaffold)