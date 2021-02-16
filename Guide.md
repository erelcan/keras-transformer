# Transfomers

## Prelude

This guide aims to aid comprehension of the concepts and their realization (specifically in Keras). We will point out important sources which explains the concepts in detail over visualizations and intuitions throughout the guide. We will also go over the layers by explaining the input/output shapes, Keras tricks etc. Therefore, readers will have a better confidence on using Keras. Though some tricks and strategies will be applicable independent of the underlying framework.

## Prerequisites

Recommended reading on transformers:
- The original paper: Attention Is All You Need [1]
- Illustrated Transformer [2]
- Transformers From Scratch [3]
- The Annotated Transformer [4]

## Key Points

Unlike Bahdanau's Attention, Transformer **parallelizes** the computation of the attention weights.
- The word in each position flows through its own path in the encoder [2].
- There are dependencies between these paths in the self-attention layer.
- The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer.

**Self attention** sees its input as a *set*, not a *sequence* [2, 3].
- If we permute the input sequence, the output sequence will be exactly the same, except permuted also (i.e. self-attention is permutation equivariant).
- We will mitigate this somewhat when we build the full transformer, but the self-attention by itself actually ignores the sequential nature of the input.
- To address this, **positional encodings** are used. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.

To learn the intuition for self-attention and queries/keys/values, the following sources may help: [3, 5, 13]

Scaling dot-product attention prevents from pushing the softmax-function into regions where it has extremely small gradients [1].

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different subspaces. With a single attention head, averaging inhibits this [1].

To prevent information leak in the decoder, mask the unseen time-steps when computing attention.

Sharing same weight matrix betweem two embedding layers and the pre-softmax linear transformation [1].
- This requires parameter tying among these layers.

## Keras Tips

### Batch Matrix Multiplication

Dot product yield a tensor with dimensions as the concatanetation of the following dimensions:
- All dimensions except the last one, for the first tensor.
- All dimensions except the one before the last one, for the second tensor.

Note that the last dimension for the first tensor and the one before the last dimension for the second tensor must the same!

**Example:**

- T1: shape -> (1, 2, 3, 4)

- T2: shape -> (8, 7, 4, 5)

- dot(T1, T2): shape -> (1, 2, 3, 8, 7, 5)

In contrast, batch dot product yields a tensor with dimensions by keeping the following dimensions:
- All dimensions till the last one from the first tensor.
- The last dimension of the second tensor.

Both tensors shouls have the same dimension lengths up to last 2 dimensions.

The last dimension for the first tensor and the one before the last dimension for the second tensor must the same!

**Example:**

- T1: shape -> (9, 8, 7, 4, 2)

- T2: shape -> (9, 8, 7, 2, 5)

- batch_dot(T1, T2): shape -> (9, 8, 7, 4, 5)

Please note that by "keeping the dimension" we only refer to the tensor shapes, values are computed due to dot product.

Please, see [6, 7] for details.

Also note that, there may be changes in implementation of the library (e.g. [12]). Hence, always run small examples before using such operations.

### Keras Layer vs. Keras Model

Model exposes:
- Built-in training, evaluation, and prediction loops (model.fit(), model.evaluate(), model.predict()).
- The list of its inner layers, via the model.layers property.
- Saving and serialization APIs.

However, handling inner models may not be straightforward as layers, especially when saving/loading (or serialization).

Please, see [8] for details.

### Custom Layers

For building a custom layer, we should extend Layer class and implement its astract methods.
- Initialization
  - Get required information from the constructor and keep them in desired fields.
  - In general keeping the fields as private would be safer.
  - We create instance of sub-layers in the initialization.
- Build
  - Here is where we create the weights of the layer.
  - Ensure you have correct shapes of the weights.
  - Use parameters (probably passed down from the constructor) to initialize the weights.
- Call
  - Here is where the execution of the node takes place.
  - Beware!! Always keep the computation graph in mind!!
  - The call method is called once when the computation graph is being created (also probably once more from some other place, but need to check).
  - Hence, prepare the logic accordingly!
  - Do always use tensors! Using other data representations (numpy etc.) may not be added to the computation graph, hence gradient computations may not be as expected.
  - Although Keras handles loops ad conditionals for simple statements, we should not rely on them as long as we can.
    - For instance, we check if a simple flag is on/off. We may have a loop to create a list of layers.
    - However, further logic may not be well represented on the computation graph.
  - Ensure the return shapes are compatible with compute_output_shape method.
  - Know your inputs (whether they are tensors or list of tensors).
  - Decide on whether to support masking or not.
  - Also, check and implement "call" for additional input parameters.
  - Be careful when refering to batch size. As it is None when the layer is being created, it would be a problem to refer to it. Therefore, it is better to use parameters passed down from the constructor. 
- compute_output_shape
  - It takes input shape and computes the output shape returned by the call method.
  - If input shape is an array of tuples, handle accordingly.
  - Returns a tuple or a list of tuples.
- compute_mask
  - It may be tricky to handle masking. It is advised to consider how it will affect the following layers.
  - Please see [10, 11] for details.
- get_config
  - Must implement for serialization.
  - Ensure that you can get the parameters passed form the constructor by this method.
  - If you have wrappers, instead of layers; implementing this may be tricky. Ensure that all required sub-layers can be created with the information given by the get_config.

### Tensor Copy and References

- Keep in mind that it might not be straightforward to track copies of tensors, especially when multiple devices are considered.
- Sharing parameters or working on copies must be taken with great care. This also applies for similar operations yielding copies of data~.
- Please, see [15] as an example.

### Parameter Tying

When implementing the transformer, we should tie the parameters of the embedding layers (similar to [14]) and the projection layer. Moreover, projection layer should use a transposed version of this weights.

Implementing this in Keras is not straightforward. I will share some approaches and why they fail in our case. Then, I will present a definite solution.

Simplest approach to share embeddings:
- Create an embedder, then use it both for encoder and decoder [16].
- However, this is insufficient to handle the projection layer.

CyberZHG implements tied embeddings in such a way that it passes the weights of the embedder to the projection layer [17, 18]. It may have several drawbacks:
- Since CyberZHG uses K.identity on weights, it creates a deepcopy; this may not allow weight updates on the same weights for both embedders and projection layer. Please, see the issue at [15].
- It may have similar issue when the model is loaded and training continues (though should check it).

nematus-transformer implements a single layer which has both embedding and projection [19].
- However, it does not extend Layer class. 
- Hence, it may not be reliable as it is not clear how computation graph will use this; and may be fragile to serialize.
- Also, our main focus is how to solve the problem by using Keras layers.

The most common way to handle parameter tying along with a projection layer (a layer using the transpose of the weights) is to pass the embedding layer to the projection layer [20-25].
- Wrong way to do this is to take the transpose in the build method.
  - In non-eager mode this creates a graph the right way. In eager mode this fails, probably because the value of a transposed variable is stored at build (== the first call), and then used at subsequent calls.
  - The solution is to store the variable unaltered at build, and put the transpose operation into the call method.
- Even when we apply the transpose operation at call method, the solution is still incomplete.
  - Serialization when saving/loading does not handle when a layer is passed in the constructor (may have been fixed in new versions~).
- So how to handle such cases when we need to pass a layer to the other?
  - **Answer1: Keras Wrapper**
    - When using wrappers, ensure that you build the wrapped layer in case it has not been build, before proceeding to remaining build operations of the wrapper.
    - However, there is still some extra work to do when loading a model. The problem is that when we save the model and try to load; the weigths of the inner layer is not tied to the weights of the embedders. It builds another embedder inside and copies the weights.
    - Therefore, we need to discard the newly build layer and tie the "real" embedder to the projector layer.
    - By simply adding a set method in the projection layer. Then, call it after loading the model and pass the embedding layer to tie.
    - It works fine, kind of a non-straightforward approach.
    - See some built-in wrappers [26] to comprehend further.
  - **Answer2: Custom Layer with Parametrized Call**
    - To solve shared-weight problem for the save/load case as well; we may create a custom layer.
    - It will have an embedding layer; and a "conditional call" and "compute_output_shape".
    - We have to be able to pass argument to the call so that it can decide whether to embed or project.
      - K.switch did not work (maybe for some other reasons..), but we were able to modify the graph with conditional
    statements.
    - We pass mode info and depending on the mode embed or project functionality is executed.
    - However, we needed one more trick! How will other layers know the output shape?
      - Output shape depends on the mode argument passed to call.
      - We keep a parameter in the layer to track the current mode. Then, whe output shape is called, it returns according to the current mode. In the way we implement, computation graph considers the condition rather than taking the first value of the mode. This is crucial for us!!!

- Also, please see the test in *TestTiedEmbeddings.py*!!!

### Gradient of Slices and Reshape

- Slice and reshape operations are handled in the computation graphs. However, please re-check whether ":" operator is also handled.
- See the discussion: [40]

### Custom Loss and Metrics

- Introduction on loss functions and explanation on logits: [41, 42]
- Custom losses in Keras: [43-46]
- Masking and loss function: [45, 47]
- How to test a custom loss function: [48]

Crucial for custom loss serialization:
- If we have custom losses, we must add them to custom objects when loading a saved model.
- However, the key string must be the same as the corresponding function name!!!
- Otherwise, the loss function couldn't be found!
- For inference purposes, we may load with compile=False. Then, compile without loss and optimizer.
- However, if we would like to resume training, we need the saved optimizer state.
  - Therefore, we must compile on load and let the loader to load the optimizer state.

If your targets are one-hot encoded, use categorical_crossentropy. But if your targets are integers, use sparse_categorical_crossentropy.

Since the target sequences are padded, it is important to apply a padding mask when calculating the loss. Please check [45] for padding aware softmax.

An example loss masking:

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
```

### Masking

- If a layer supports masking it should take mask as input to call and process accordingly.
- To propagate masks to the next layers, it should implement compute_mask method.
- If compute mask is not implemented, it will use the one from the base class.
  - If self.supports_masking is False, then it checks whether the mask is not None (or if it is a list whether it has not None elements). If not None, then it raises an error to indicate that the mask is not supoorted. If mask is None, it returns None.
  - If self.supports_masking is True, then it passes the mask as it is.
  - **Therefore be careful if you do not want to propagate the mask!!**
- Masks are handled *automatically* in when computing the loss.
  - Please, see model.compile().
  - The mask is multiplied with the sample_weights; and then applied to the loss computation.
  - Therefore, if this is not what you want from masking in loss, do not propagate the mask. Instead, wrap the loss function such that the custom loss takes care of the mask.
  - How to achieve this depends on the use case.
    - If you know the mask in prior and can create a custom loss and pass the mask directly, then do it.
    - If the mask has to be computed along the way, then propagate the mask. Then, use the mask with custom loss or built-in loss.
    - If somehow, you need to use masking in some layers but not in the loss, just stop propagating the mask. If you can't do this implicitly, use a custom layer such that takes input and a mask; returns the input as is; but returns the mask as None.
- Being aware of such crucial points is way to go for designing a robust model.
- Please see a similar discussion in [47].
- Also, please see the test in TestMasking.py.

### Separeting Activation

We can use activation layers instead of dense layers:
- The activation layer in keras is equivalent to a dense layer with the same activation passed as an argument [49].


### Learning Rate Scheduler

- See [60-62] for custom learning rate schedulers.
- See [63-65] for learning rate finder.


## Additional Concepts

### LogSumExp

A trick to improve numerical stability when computing log(sum(exp(xi))) over all i.
- Simply, subtract maximum xi from all the exponentials.~ 
- See [27, 28] for better explanation

### Normalization

Please see the following sources for detailed explanation of the normalization methods: [29-31]

### Label Smoothing

When data have targets which may be mis-labeled, applying label smoothing may help. In a way, it avoids hard targets and injects the label uncertainty in the loss function.

Please, see [32, 33] for details.

To be able to use label smoothing, we need one-hot vectors. Ground-truth labels must be one-hot encoded. For predictions, we probably should have a tensor with the same shape as the ground truth:
- If we use softmax, we may assume that for each element in the sequence, we have a probability distribution over classes. Hence, it's output can be used.
- If we use logits, we may assume that for each element in the sequence, we have logits for each class. Hence, it's output can be used.
- For smoothing, we must use categorical_cross_entropy, but not sparse_categorical_cross_entropy as we can't apply label smoothing on integers~.
- We can also have a custom loss, where we convert to one-hot in the loss, and call categorical_cross_entropy..
- Therefore, when label_smoothing is True, we should arrange batches accordingly (i.e. one-hot encoding ground-truth).
- For performance reasons, using logits will be faster~.

### Positional Encoding

As embedding in transformer is permutation invariant, we add positional encodings to the input; so that the positional information can be utilized.

Here are several interesting questions and discussions:
- About convolutional implementation of positional encodings: [34, 35]
- A derivation for linear relationships in the transformer’s positional encoding: [36]
- Why add positional embedding instead of concatenate: [37, 38]
- tensor2tensor implements positional embeddings in a slightly different way such that they concatanate sines and cosines back to back rather than interleaving. Here is the corresponding discussion: [39]
  - Note that we implement positional encodings as in the original paper. Please, see the implementation.


### Training and Inference

**Training strategies [50, 51]**
- *Teacher forcing:*
  - A strategy for training recurrent neural networks that uses model output from a prior time step as an input.
- *Search Candidate Output Sequences:*
  - Predict a discrete value output, such as a word, to perform a search across the predicted probabilities for each word to generate a number of likely candidate output sequences.
  - A common search procedure for this post-hoc operation is the beam search.
- *Curriculum learning:*
  - A variation of forced learning is to introduce outputs generated from prior time steps during training to encourage the model to learn how to correct its own mistakes.
  - The approach is called curriculum learning and involves randomly choosing to use the ground truth output or the generated output from the previous time step as input for the current time step.
  - The curriculum changes over time in what is called scheduled sampling where the procedure starts at forced learning and slowly decreases the probability of a forced input over the training epochs.

See for training tips: [52-54]

**Inference**
- May apply beam search decoder and variants [55, 56].

See examples: [57-59]


### Byte-Pair Embedding (BPE)

BPE is a kind of subword embedding technique [66].

[67] presents a library which provides a collection of pre-trained subword embeddings in 275 languages, based on Byte-Pair Encoding (BPE) and trained on Wikipedia. It also has a multi-language embeddings.

In our study, we would like to use multi-language embeddings which is shared by both the encoder and the decoder. However, it does not allow multi-language embeddings with less than 100K sized vocabulary and dimension length 300. Due to limitations on our computational resources, we applied separate embeddings on encoder and decoder in our demo case.

### Datasets

- https://lionbridge.ai/datasets/25-best-parallel-text-datasets-for-machine-translation-training/
- http://www.statmt.org/europarl/
- http://opus.nlpl.eu/
- https://panlex.org/source-list/
- https://data.world/datasets/ercot

## References

[1] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

[2] [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

[3] [Transformers From Scratch](http://peterbloem.nl/blog/transformers)

[4] [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[5] [What exactly are keys, queries, and values in attention mechanisms?](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms)

[6] [Understand batch matrix multiplication](http://christopher5106.github.io/deep/learning/2018/10/28/understand-batch-matrix-multiplication.html)

[7] [Understanding batch_dot() in Keras with Tensorflow backend](https://stackoverflow.com/questions/54057742/understanding-batch-dot-in-keras-with-tensorflow-backend)

[8] [Difference between tf.keras.layers.Layer vs tf.keras.Model](https://stackoverflow.com/questions/55109696/tensorflow-difference-between-tf-keras-layers-layer-vs-tf-keras-model)

[9] [Making new layers and models via subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)

[10] [How to support masking in custom tf.keras.layers.Layer
](https://stackoverflow.com/questions/55176818/how-to-support-masking-in-custom-tf-keras-layers-layer)

[11] [Masking and padding with Keras](https://www.tensorflow.org/guide/keras/masking_and_padding)

[12] [batch_dot has different behaviour in keras.backend and tf.keras.backend](https://github.com/keras-team/keras/issues/13300)

[13] [Deep Learning: The Transformer](https://medium.com/@b.terryjack/deep-learning-the-transformer-9ae5e9c5a190)

[14] [Using the Output Embedding to Improve Language Models](https://arxiv.org/pdf/1608.05859.pdf)

[15] [tf.copy() as alternative to tf.identity()](https://github.com/tensorflow/tensorflow/issues/11186)

[16] [Keras - How to construct a shared Embedding() Layer for each Input-Neuron](https://stackoverflow.com/questions/42122168/keras-how-to-construct-a-shared-embedding-layer-for-each-input-neuron)

[17] [CyberZHG-keras_embed_sim](https://github.com/CyberZHG/keras-embed-sim/blob/master/keras_embed_sim/embeddings.py)

[18] [CyberZHG-Transformer](https://github.com/CyberZHG/keras-transformer/blob/master/keras_transformer/transformer.py)

[19] [nematus-transformer](https://github.com/EdinburghNLP/nematus/blob/master/nematus/transformer_layers.py)

[20] [How to tie word embedding and softmax weights in keras?](https://stackoverflow.com/questions/47095673/how-to-tie-word-embedding-and-softmax-weights-in-keras)

[21] [Keras Weight Tying / Sharing](https://forums.fast.ai/t/keras-weight-tying-sharing/11102)

[22] [Loading a layer that uses weights from another layer](https://github.com/keras-team/keras/issues/10485)

[23] [Tying Autoencoder Weights in a Dense Keras Layer](https://stackoverflow.com/questions/53751024/tying-autoencoder-weights-in-a-dense-keras-layer)

[24] [Building an Autoencoder with Tied Weights in Keras](https://medium.com/@lmayrandprovencher/building-an-autoencoder-with-tied-weights-in-keras-c4a559c529a2)

[25] [layers_tied](https://gist.github.com/dswah/c6b3e4d47d933b057aab32c9c29c4221)

[26] [Keras-Wrappers](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py)

[27] [How does the subtraction of the logit maximum improve learning?](https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning)

[28] [LogSumExp-Wiki](https://en.wikipedia.org/wiki/LogSumExp)

[29] [Lecture 49 Layer, Instance, Group Normalization](https://www.youtube.com/watch?v=NE61nLoM-Fo)

[30] [An Overview of Normalization Methods in Deep Learning](https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)

[31] [An Intuitive Explanation of Why Batch Normalization Really Works](http://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/)

[32] [Label Smoothing: An ingredient of higher model accuracy](https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0)

[33] [When Does Label Smoothing Help?](https://medium.com/@nainaakash012/when-does-label-smoothing-help-89654ec75326)

[34] [Why would you implement the position-wise feed-forward network of the transformer with convolution layers?](https://ai.stackexchange.com/questions/15524/why-would-you-implement-the-position-wise-feed-forward-network-of-the-transforme)

[35] [Multi-Head attention mechanism in transformer and need of feed forward neural network](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

[36] [Linear Relationships in the Transformer’s Positional Encoding](https://timodenk.com/blog/linear-relationships-in-the-transformers-positional-encoding/)

[37] [Why add positional embedding instead of concatenate?](https://github.com/tensorflow/tensor2tensor/issues/1591)

[38] [Positional Encoding in Transformer](https://www.reddit.com/r/MachineLearning/comments/cttefo/d_positional_encoding_in_transformer/exs7d08/)

[39] [tensor2tensor-fix positional embedding](https://github.com/tensorflow/tensor2tensor/pull/177)

[40] [Can auto differentiation handle separate functions of array slices?](https://stackoverflow.com/questions/35021018/can-auto-differentiation-handle-separate-functions-of-array-slices)

[41] [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

[42] [How to choose cross-entropy loss in TensorFlow?](https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow)

[43] [Keras-Losses](https://keras.io/api/losses/)

[44] [Advanced Keras — Constructing Complex Custom Losses and Metrics](https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618)

[45] [Vandergoten-Transformer](http://vandergoten.ai/2018-09-18-attention-is-all-you-need/)

[46] [How to implement my own loss function?](https://github.com/keras-team/keras/issues/2662)

[47] [How do I mask a loss function in Keras with the TensorFlow backend?](https://stackoverflow.com/questions/47057361/how-do-i-mask-a-loss-function-in-keras-with-the-tensorflow-backend)

[48] [How to test a custom loss function in keras?](https://stackoverflow.com/questions/50862101/how-to-test-a-custom-loss-function-in-keras)

[49] [Keras functional api explanation of activation() layer?](https://datascience.stackexchange.com/questions/38981/keras-functional-api-explanation-of-activation-layer)

[50] [What is Teacher Forcing for Recurrent Neural Networks?](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)

[51] [What is a Transformer?](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)

[52] [This is how to train better transformer models](https://towardsdatascience.com/this-is-how-to-train-better-transformer-models-d54191299978)

[53] [Training Tips for the Transformer Model](https://arxiv.org/pdf/1804.00247.pdf)

[54] [Learning Deep Transformer Models for Machine Translation](https://arxiv.org/pdf/1906.01787.pdf)

[55] [How to Implement a Beam Search Decoder for Natural Language Processing](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/)

[56] [Beam Search Strategies for Neural Machine Translation](https://arxiv.org/pdf/1702.01806.pdf)

[57] [Create The Transformer With Tensorflow 2.0](https://machinetalk.org/2019/04/29/create-the-transformer-with-tensorflow-2-0/)

[58] [TF-create_the_transformer](https://www.tensorflow.org/tutorials/text/transformer#create_the_transformer)

[59] [TF-evaluate](https://www.tensorflow.org/tutorials/text/transformer#evaluate)

[60] [Keras learning rate schedules and decay](https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/)

[61] [Keras loss-based learning rate scheduler](https://www.kaggle.com/fergusoci/keras-loss-based-learning-rate-scheduler)

[62] [TensorBoard Scalars: Logging training metrics in Keras](https://www.tensorflow.org/tensorboard/scalars_and_keras)

[63] [WittmannF-LRFinder_Playground](https://github.com/WittmannF/LRFinder/blob/master/LRFinder_Playground.ipynb)

[64] [WittmannF-lr_finder_keras](https://gist.github.com/WittmannF/c55ed82d27248d18799e2be324a79473)

[65] [jeremyjordan-lr_finder](https://gist.github.com/jeremyjordan/ac0229abd4b2b7000aca1643e88e0f02)

[66] [Information from parts of words: subword models](https://medium.com/analytics-vidhya/information-from-parts-of-words-subword-models-e5353d1dbc79)

[67] [BPEmb: Subword Embeddings in 275 Languages](https://nlp.h-its.org/bpemb/)