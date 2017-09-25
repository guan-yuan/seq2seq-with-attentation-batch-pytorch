# formosa_grand_challenge
It's the baseline model (seq2seq with attention mechanism) for the formosa grand challenge, which is modified from [pratical-pytorch seq2seq-translation-batched](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#) and [Tensorflow Sequence-to-Sequence Models](https://www.tensorflow.org/tutorials/seq2seq).
I'm planning to integrate the Word2Vec in the model soon. You also can try to use the embedding produced from CBOW code or other pre-trained Word2Vec to replace the torch.nn.Embedding(num_embeddings, embedding_dim) in the encoder and decoder models.

## Requirements
pytorch v0.2.0<br>
scikit-learn<br>
sconce<br>

## Dataset:
[Link](https://drive.google.com/drive/folders/0B4-rB9HD2WbEMXhScHVBOHFqeTA?usp=sharing)<p>
Download from the "Link", then replace the "./data" folder.<p>

Note: This dataset is modified from the [NOAH'S ARK LAB Short-Text Conversation dataset](http://www.noahlab.com.hk/topics/DeepLearning4NLPDatasets). <p>

Please cite the following paper if you use the data in your work.<br>
Neural Responding Machine for Short-Text Conversation. Lifeng Shang, Zhengdong Lu, and Hang Li. ACL 2015.


## Access trained parameters and records
[Link](https://drive.google.com/drive/folders/0B4-rB9HD2WbENFh5VGROcUNxekE?usp=sharing)<p>
Download from the "Link", then replace the "./save" folder.

## Happy train; Happy gain!

