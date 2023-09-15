# Transformer/Roformer

The main models we've been investigating recently have been the roformer/transformer models. The main difference between the two is that the Roformer model uses Rotary Embeddings (https://arxiv.org/pdf/2104.09864.pdf). They are supposed to help with varying length inputs, and in general, just perform regular without. We use the huggingface implementation.

Transformer is the default pytorch implementation.

These models work with the character-level input, the byte-level input, and the visual representation inputs.

For characters, it is relatively straightforward. In the same way that subtokens would be indexed and fed into a transformer for other tasks, we instead use characters, index them, and feed them into the transformers.

The byte-level inputs function the same except the indexing is the same mapping as the byte number (0-255).

In both these cases, the indices are used to extract an embedding from an Embedding Matrix, which is the underlying input to the model.

The visual representations are based on this paper (https://arxiv.org/abs/2104.08211). We use our own implementation. The visual representations also require a fonts/ directory (.ttf files). Each character is rendered individually and a series of convolutions are applied. The output of these convolutions replace the embeddings and Embedding Matrix in the prototypical case.

# CLD3 Model / Ngram Model

We initially started with this model as it is the same as the Google CLD3 model. They do not have a paper, but you can read about it here (https://github.com/google/cld3).

Each NGram model has a series of orders that it focuses on. In the Google diagram, this would be 1,2,3. The orders are parameters for our codebase. For each order, all ngrams of that order (i.e., unigrams) are extracted. These ngrams are then hashed to a range--this is the same range as the size of the vocabulary. This is how embeddings are extracted from an Embedding Matrix. We then average the embeddings across all ngrams.

The average of all orders are concatenated together as input to a linear model. It is a small model, so it should be faster and lower profile than the transformers.

This code was not optimized for speed, but honestly the n-gram extraction can be quite slow, and we haven't found the greatest results anyway.

# UNET

Unets are usually used in image segmentation--identifying which pixels are part of a given object in an image and which aren't.

This was our initial architecture for token-level labelling since it can do the same thing in fewer dimensions for language.

It was finnicky to train, so the results were not initially promising.