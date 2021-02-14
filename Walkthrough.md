# Walkthrough

## Nomenclator

- Here is the shorthand notation for variables throughout the document:
  - b: batch size
  - t: sequence length
    - tE: sequence length for encoder input
    - tD: sequence length for decoder input
    - tq: sequence length for query
    - tk: sequence length for key
    - tv: sequence length for value
  - f: feature length
    - fE: feature length for encoder input
    - fD: feature length for decoder input
    - fq: feature length for query
    - fk: feature length for key
    - fv: feature length for value
    - fhq: feature length per head for query
    - fhk: feature length per head for key
    - fhv: feature length per head for value
  - To specify lengths for encoder and decoder, subscript them with E or D (e.g. tq_E, tq_D).
  - h: number of heads
  - embE: embedding length for encoder
  - embD: embedding length for decoder

- We present a masking sub-section to demonstrate the whole flow of the masks. Hence, details in layer sub-sections may not be complete. 


## Transformer

**input:** [(b, tE), (b, tD)]

**output:** (b, tq_D, fv_E)

Input is a list of 2 tensors corresponding to encoder input and decoder input.

A padding mask is created both for encoder and decoder at the beginning. This will be passed through layers, and also will be used in the custom loss to ignore padding.

Apply embedding on both encoder and decoder inputs.
- Embedding for encoder: (b, tE) -> (b, tE, embE)
- Embedding for decoder: (b, tD) -> (b, tD, embD)
- As we have the built-in keras embedding layer, prepare input accordingly outside the transformer. Embedding layers can be customized in future releases.
- **There is no fE and fD in the inputs as built-in embedding layer requires 2D input. However, for different embedding layers, we may have such 3rd dimension!**

We support several embedding approaches and manage them in the transformer to keep the project and embedding layers simple:
- Shared weights and non-shared weights
  - If shared, both encoder and decoder embedding layers and the projection layer share the same weights. Though, projection layer uses the transpose of the weights.
    - We also allow only decoder embedder and projection layer to share the same weights, while encoder embedder has separate weights.
    - Also, it is possible to separate all weights.
    - We also enable using pre-trained weights!
      - However, we create/load pre-trained weights over a factory ratehr than passing through the constructor of the parent layer, in order to ease serialization.
  - If shared, embedding weights would represent a common space.
    - E.g. for machine translation task, embedding weights will be shared for both languages.
    - It is better to use such an approach with pre-trained weights trained on sub-words from each language.
- We allow using pre-trained weights.
  - Such weights can be shared or utilized per embedder/projector.
- We allow trainable and frozen weights for embedders and the projection layer.
- As suggested in the paper, we multiply the weights with sqrt(dmodel) where dmodel is the embedding length (embE and embdD).

After adding positional encodings, shapes remain same:
- For encoder side: (b, tE, embE)
- For decoder side: (b, tD, embD)

Then, we apply encoder-decoder:
- Takes list of tensors with shape [(b, tE, embE), (b, tD, embD)] as input.
- Returns tensor with shape (b, tq_D, fv_E).

Finally, we apply the projection layer:
- Returns a tensor with the same shape as the encoder-decoder output: (b, tq_D, fv_E)

If the return_logits True, we directly return the result of the projection layer. Otherwise we apply softmax and then return.
  - Use this option accordingly to the loss function.

**Note that we expect fv to be same for queries and values to be able to run Add layer as it adds over last dimension with lengths fv and fq. Also, in the decoder for being able to run the 2nd sublayer's Add, fv_E and fv_D must be equal. Therefore, as paper does, we should ensure that the length of the last dimension is always the same outside multi-head attention and dense layers of feed-forward sub-layer (Inside multi-head attention, and for the dense layers; the last dimension can change). Hence, we stick together with d_model parameter! For other use-cases transformer network needs a re-consideration and the architecture in the paper will not be sufficient (e.g. Add layer..).**


## PositionalEncodingLayer

**input:** (b, t, emb)

**output:** (b, t, emb)

Note that the encoding length should be equal to the length of the last dimension of the input. In this case, it should be equal to the embedding length.

- Creates the signal and adds to the input.
- We can fetch the signal by calling get_signal.
- No need to override the compute_output_shape method of Layer (the base class);
as the output_shape is the same as the input_shape.
- We implemented 2 versions of positional encoding.
  - First one is as in the paper.
  - The second one is as in tensor2tensor, but with a corrected interleaving.
  - In tensor2tensor, they consider the given timescales, rather than using just a constant.
- For the computation, we compute logs and than exponentiate, for better stability.
- Interleaving trick:
  - Although most implementations just create the encodings by using numpy etc., we implemented this in pure keras.
  - After creating sin_encodings and cos_encodings, we expand them on the last dimension.
  - Then, we concatanate them on the new dimension.
  - Finally, we reshape them to reduce the newly created dimension.
  - Reshape operation, in this way, does the interleaving what we are looking for!
- Please see *TestPositionalEncoding.py*.


## EncoderDecoder

**input:** [(b, tE, embE), (b, tD, embD)]

**output:** (b, tD, d_model) where embE = embD = d_model

For now, allowing only single encoder input and decoder input rather than q/v/k for each.
- First implementation has that feature which takes a list of 6 tensors (as an alternative)
and arranges the inputs and masks.
- However, for the sake of simplicity, we decided to remove that. In case,
you need such a feature, implement a new encoder-decoder or add input/mask arranger.

Contains EncoderBlockStack and DecoderBlockStack.


## EncoderBlockStack

**input:** (b, tE, embE)

**output:** (b, tE, d_model)

Applies dropout on the inputs and then applies encoder blocks.
- Encoder blocks are auto-created given the number of blocks.


## DecoderBlockStack

**input:** [(b, tE, d_model), (b, tD, embD)]

**output:** (b, tD, d_model)

Takes encoder (stack) output anf decoder embeddings as input.

Applies dropout on the decoder input and then applies decoder blocks.
- Decoder blocks are auto-created given the number of blocks.


## EncoderBlock

**input:** (b, tE, embE/d_model)

**output:** (b, tE, d_model)

EncoderBlock applies SelfAttentionSublayer and PositionWiseFeedForwardSublayer.


## DecoderBlock

**input:** [(b, tE, d_model), (b, tD, embD/d_model)]

**output:** (b, tD, d_model)

DecoderBlock applies Masked-SelfAttentionSublayer, SelfAttentionSublayer and PositionWiseFeedForwardSublayer.


## SelfAttentionSublayer

**input:** (b, t, f) or [(b, tq, fq), (b, tk, fk), (b, tv, fv)]

**output:** (b, t, f)

Applies MultiHeadAttention, LayerNormalization, Adder and Droput.

Takes head_num, context_mask_type and dropout_rate as arguments.
- Passes head_num, context_mask_type to MultiHeadAttention.

For transformer in the paper, only "narrow" head_type should be allowed since the adder would require same feature length with the query.

If inputs is a list, add Q in the adder. Assume that the inputs is [Q, K, V] if it is an array.
  - We should assert that fv equals to fq; otherwise adder will fail.


## PositionWiseFeedForwardSublayer

**input:** (b, tq, fv)

**output:** (b, tq, fv)

Applies 2 dense layers, dropout, adder, layer normalization.

Takes d_model, inner_length and dropout_rate as arguments.
- inner_length is expected to be larger than d_model (not restricted in the code, but advised in theory).
- Input shape will be equal to the output shape.
  - Also, input mask and output mask will have the same shape.

We may apply the following masking, but it seems redundant since the linear transformations on the last
dimension. Hence, masked time-steps do not effect each other!
```python
if mask is not None:
    inputs *= K.cast(K.expand_dims(mask, axis=-1), K.floatx())
```


## MultiHeadAttention

**input:** (b, t, f) or [(b, tq, fq), (b, tk, fk), (b, tv, fv)]

**output:** (b, tq, fo/d_model)

Flow:
- If input is a single tuple, it is copied as queries, keys and values.
- Applies projection on queries, keys and values.
- Then, DotProductAttention is applied.
- Finally, output projection is applied.

Although output shape is fixed to d_model in the paper, we allow user to be able to define fo for other use-cases.

MultiHeadAttention supports both wide and narrow attention!
- However, transformer in the paper is appropriate only for narrow attention. (Otherwise, layers gets larger as # of block increases..)

**To reduce computation cost, heads are processed altogether.**
- To accomplish this, shape of queries, keys and values are altered before passing to DotProductAttention; then attended values are reshaped accordingly.

Details:
- Let inputs shapes to be: shape_q = (b, tq, fq), shape_k = (b, tk, fk) and shape_v = (b, tv, fv) where b is batch_size, t represents sequence length and f represents embedding/feature size
- Let weight shapes to be: shape_Wq = (d_model, dk), shape_Wk = (d_model, dk), shape_Wv = (d_model, dv) and shape_Wo = (head_num * dv, d_model)
- According to the paper, fq = fk = fv = d_model, also dk = dv = d_model // head_num
- After linear projections, projected input shapes should be:
  - shape_Pq = (b, tq, dk), shape_Pk = (b, tk, dk) and shape_Pv = (b, tv, dv)
- dk can be different than dv as dk's cancel out when computing QK.
- Weights will have dk and dv multiplied by head_num since we will handle all heads in single tensor.
- attentded_values has shape (batch_size * head_num, tq, dv)
  - After moving heads from batch dimension, shape is (batch_size, tq, dv * head_num)
- **Do not ever use batch_size as argument to a layer!!!**
  - For handling head management (reshaping~), it looks like we need batch size.
  - However, let keras infer it!
  - It is no guaranteed way to obtain batch size in a layer **at all times**!!
  - In a case which layer depends on a batch_size in its constructor:
    - It must be serialized for a pre-defined batch size.
    - Such approach disables (continueing to) training with a different batch size.
    - More critically, in prediction, we can only pass batches with that pre-defined batch size; which creates much more inconvenience.
  

## Dot-Product Attention

**input:** [(b, tq, fq), (b, tk, fk), (b, tv, fv)]

**output:** (b, tq, fv)

**output_with_attention:** [(b, tq, fv), (b, tq, tk)]

- Assumes that inputs is a list/tuple, it has query (Q), keys (K) and values (V)
- Otherwise, raises error!
- Any custom projection is done on Q, K and V before calling this method.

- Shape of Q, K and V are (batch_size * head_num, seq_len, head_length).
- Note that this method is agnostic of heads, therefore we should prepare Q, K and V appropriately before inputting them into this method. Corresponding weights should be - created outside; and reshaped according to the number of heads. (This allows re-usability)
- Choice to apply wide or narrow attention is up to the caller of this method.

Use context_mask to decide which keys to mask when computing attention for a query.
- For example, left-context-mask can be used to prevent contribution of future keys. (Which is used for computing attention for the decoder: a.k.a. masked-attention)
- For re-usability, in case there are other strategies to mask, we get it as argument rather than fixing one.
- Note that context_mask_type is string.

To compute inside_softmax, queries and keys should have a shape of (batch_size, sequence_length, feature_length).
- batch_dot will keep the batch dimension fixed, the dimensions given by the axes will be reduced.
- Specifying the axes removes the need for transposing the keys tensor.
- scaler is as defined in the paper to counteract the effect of large dot products.

The shape of inside softmax is (batch_size * head_num or 1, sequence_length_of_queries, sequence_length_of_keys)
- Hence, we need to expand_dims of the mask (in batch dimension) as context_mask (e.g. for left_context_mask) is 2D; but this is handled implicitly on multiplication~.
- We compute QK mask by accounting for padding mask and context mask.
- Also, we apply padding mask for values before batch_dot with attention_weights.
  - We just zero out "values" which has False/0 in the mask.

"attention_weights" represents the weights of each pair of queries and keys, in the sequence.

Here is how output_shape is derived:
- Let shapeQ = (b, tq, fq), shapeK = (b, tk, fk) and shapeV = (b, tv, fv) where b is batch_size or batch_size * head_num
- QK and softmax(QK/sqrt(fq)) have shape: (b, tq, tk)
- soft(QK/sqrt(fq))V has shape: (b, tq, fv)
- Also note that tk must be equal to tv.
- Hence, output shape will be (b, tq, fv) whereas attention shape is (b, tq, tk)

Also, please see the *TestContextMask.py* and *TestQKMask.py*.


## TiedEmbedderProjector

Please read Parameter Tying in the Guide for moe details.

This layer acts like both and embedder and projector.
- Main use is for parameter tying of embedders and projector.
- **Critical part is the parametrized call!**
- Mode switch effects the call output and output shape computation.
- Straightforward to serialize.

To initialize with pre-trained weights, we use a factory to retrieve corresponding weights rather than passing the weigths through the constructor.
- Ease serialization.
- Saved model will be smaller.

We also kept our previous implementation which requires tied-layer setting after model load.
- Please observe *TiedProjectionLayer.py*.
- This is a wrapper class such that taking another layer (i.e. embedding layer in this case); and utilizing (transpose of) its weights during projection.
  - In build, we should assure that the inner layer is built first.
  - Do not apply any backend functions in build as the weight have not been initialzied.
    - E.g. we have to move the transpose into the call function.
  - Also, override get_weights method to return the transpose of the inner layer. (should be optional though)
  - Please, refer to the guide for Parameter Tying!

We also have ProjectionLayer to be used when no parameter sharing.
- It can simply be thought as a dense layer though.


## Masking

For grasping the overall picture, masking related discussion for each layer is presented altogether, in this section.

Also note that, we support when masks are different for queries, keys and values unlike in the original paper (which splits input to queries, keys and values; so their masks).

- **Transformer** does not take and return mask.
  - However, masks are created, utilized, passed internally.
- Positional encoding layer, adds the position signal to the input. However, we do not apply the mask on this addition, for not to lose the positional info.
- Internally created masks are passed to EncoderDecoder which is called after embedders and positional encoding layer.
- **Multi-head Attention** takes a padding mask with shape (batch_size, sequence_length).
  - It splits the last dimension of the input into heads. Hence, the time-steps (dimension 1) do not change. Therefore, we can apply the mask to each head as is.
  - For reducing the computation costs, multi-head attention creates heads in the same tensor. Therefore, we should compose the mask accordingly.
  - In this respect, mask is expanded and tiled to have (batch_size, num_of_heads, sequence_length). Then, reshaped to (batch_size * num_of_heads, sequence_length).
- In **Dot-product Attention**, we should account for both the padding mask and the context mask.
  - We should apply the masks before applying the softmax.
  - Rather than setting masked elements to -infinity, we set them to -1e9. Since, infinity may cause nan.
  - Context masks is produced by context_utils; returning a lower triangle.
  - For padding masks, we should first create a (tq, tk) mask for each element in the batch dimension of the inside-softmax (which has a shape of (b, tq, tk)). To obtain this, we can apply batch_dot on expand_dims(Mq, -1) and expand_dims(Mk, 1).
  - After the softmax, before multiplying V, we can apply its own mask (Mv); but this time setting its elements to zero (should verify this further~).
    - When queries, keys and values are same, some of the masks might be redundant.
  - Dot-product Attention layer returns the mask on queries. However, Multi-head attention will not return this tiled multi-headed mask; but single mask.
  - Notice that skip input addition has the same mask on queries.
- Addition, Layer Normalization and Pointwise Feed Forward Layer work on each time-step separately. Hence, time-steps to be masked do not effect others.
  - For Pointwise Feed Forward Layer; input is (b, tq, fv), W1 is (fv, dff), W2 is (dff, dmodel).
  - dot(input, W1) produces (b, tq, dff) and works on each time-step separately. Similarly when applying dot product on the result and W2.
  - Therefore, we may not apply mask through these layers. However, we can pass on the mask on the queries. Inside, encoder/decoder stack, the mask on queries will be passed from encoder/decoder to other. Then, multi-head attention will utilize this information. Hence, no need to apply the mask in others.
- Encoder stack will pass the mask on queries as key/value masks for 2nd sublayer of the decoder. In that sublayer, decoder will also use the mask on decoder queries.
  - In the decoder, among sublayers, the mask on the queries will be passed.
- **Encoder-decoder** will not return a mask to the projection layer. We may assume that if a pad token inputs its output from the decoder should also be a pad token. Hence, we may return one-hot representation of the pad token, for the masked time-steps. Since, we are going to ignore such time-steps in the loss function; we do not handle them explicitly. Also, for some cases maybe the assumption on the next of pad token may not hold.
- In the loss, we will apply the mask which is obtained from true values. Hence, rather than passing mask on predicted values, we need to utilize the mask on the ground-truth. Therefore, we implement a custom loss.


## Decoding

We present 2 decoding schemes.
- Greedy Decoding
- Beam Search Decoding

DecoderABC is the abstract class, which loads model and artifacts, to decode given sequences.

We also have DecodingCompatibleProcessorABC in the processors package.
- This standardize processors which are more capable for decoder' usage.
- Specifically, we use it for encoding sequences to subwords and decoding decoder outputs back to sequences.
- We may need a better convention for encoder/decoder and encode/decode over layers, decoding and processors :)

## Generators

Our design decouples domain specific generators and the inner generator which is required for internal usage.
- This allows usage of transformer (or for other custom layers/models), for any domain.
- InnerGenerator draws the interface for the outer generators and accompanying processor.
- Hence, we can bring data into required format in the outer generator or in the processor if we can't directly modify the outer generator.
- InnerGenerator handles how data is to be passed to the model (controlling usage of remaining batch, handling pass_count [whether infinite or limited], keeping required meta info for the data generation process etc.).
- As trainer is aware of the interface of InnerGenerator, it does not need to know the properties of the outer generators.
  - **This is what brings the domain-independence to our training process!**


## Training

Trainers need to implement _get_model and train.
- TrainerABC keeps artifacts such as processors, custom_objects etc.
- Trainers should manage them in their flow appropriately so they can be saved/loaded without errors.
- We may consider moving train fucntion to TrainerABC in the future.


## Demo

A demo is prepared for machine translation.
- Our generator allows loading any language pair from "http://storage.googleapis.com/download.tensorflow.org/data/...".
- We intended to directly apply the architecture in the paper along with multi-language byte-pair embeddings.
  - However, due to computation limitations, we work on a much smaller network; and much less sample.
  - Also, rather than sharing multi-language byte-pair embeddings (since only pre-trained multi-language model with 100K vocabulary and 300 embedding-length exists in bpemb library), we separated encoder_embedder and decoder_embedder-projector weights. Though, we tied decoder_embedder and projector.
- Demo is defined over a simple DSL and executed.
  - It is convenient to modify the demo/layer-parameters etc. over the DSL.
  - Also, components of the library is highly re-usable.