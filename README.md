# formosa_grand_challenge
It's the baseline model (seq2seq with attention mechanism) for the formosa grand challenge, which is modified from [pratical-pytorch seq2seq-translation-batched](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#).
I'm planning to integrate the Word2Vec in the model soon. You also can try to use the embedding produced from CBOW code or other pre-trained Word2Vec to replace the torch.nn.Embedding(num_embeddings, embedding_dim) in the encoder and decoder models.

## Happy train; Happy gain!

