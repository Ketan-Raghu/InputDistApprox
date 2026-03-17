# Caveats to Training Setup

### Domain Mismatch

Typical distillation handles $F: X -> Y$ distilling into $F': X -> Y$ whereas this study is trying to predict the input distribution meaning we're distilling into $F^{-1}: Y -> X$. This means we cannot take advantage of dark knowledge and logit information typically used in distillation and have to rely on my direct input:response (text:text) pairs in order to apply the same model since the current input embeddings expect text rather than logit vectors.

The same information can likely be captured by running each prompt $n$ times with a higher temperature to gain a sense of variance by prompt. While this increases data generation and training times both by a factor of $n$ for the same number of prompts, it may be worthwhile for the model to map the output space better.