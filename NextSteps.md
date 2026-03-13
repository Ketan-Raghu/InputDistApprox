# Next Steps

Focus on exploring the learned input space, need to probe for outliers than verify against the base network (find logits that generate misclassified images)

## Probing the Input Space

We need an efficient method of searching the input space for potential issues. We could probe along the boundary for 'unconfident' misclassifications (for example [0.099, 0.099, 0.109, 0.099...]), but a largest undetected problem is 'confident' misclassifications (for example [0.991, 0.001, 0.001, 0.001...]).

We've previous demonstrated that logits encode a sense of confidence meaning 'unconfident' outputs that result in hallucination (in terms of LLMs) can be detected from the logits alone. In a broader context, we're more concerned with finding confident misclassifications which are harder to detect at inference. For example, a simple algorithm can be used to send any classifications <0.90 to be manually classified by a human.

This may be intractable on the standard MNIST dataset since it is well-defined and likely does not contain confident misclassifications that humans wouldn't make. For example improper processing resulting in a 2 looking like a 7. It may be possible on a augmented dataset.

Verifiability is straightfoward, check against the base network.

## Current Primary Problems

<u>Data reconstruction</u> - Classification networks such as the one used to explore MNIST inherently lose data due to latent compression (784 input --> 10 output), which means perfect reconstruction of the input space is not possible without overfitting, but overfitting on the training set prevents generalization of the model's input space.

<u>Network Parallelism</u> - The goal of using a flipped network was to create the most direct parallel for inversing a network which would ideally learn the networks input space through distillation rather than generalizing beyond and removing the critical outliers in the base model's input space. However, this is a fairly naive approach and a different architecture may generalize the input space better.