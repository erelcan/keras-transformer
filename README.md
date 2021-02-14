# Transformer

This project implements *"Attention is All You Need"* paper. 

We present a detailed guide to comprehend the transformer concepts and required Keras functionality/tricks **(Please see Guide.md and Walkthrough.md).**

We hope that the software and the documentation will allow the community to create and automatize custom Keras layers/models/solutions in a more robust and faster manner.

We also share a machine translation demo which can be setup over a *DSL.*

We would appreciate any contribution:)
- Would be lovely if any large models can be trained and results are shared.
  - Couldn't test for now due to limited computational resources.
- Although we run formal and informal tests during development; we would appreciate more tests.
  - Also any suggestions and requests.
- Please also contribute to the docs if you have other Keras tricks or alternative approaches.


## Key Contributions

- Model training and decoding can be defined over a DSL; and therefore execution is automated.
- Abstracts training basics (checkpointing, artifact management etc.) from custom trainers.
- Keeps a definite interface for generators (the so-called *inner-generator*) feeding the models.
  - Users may provide their custom generators (the so-called *outer-generator*) to integrate any data source.
  - Thus, we separate the ingestion logic from ML/DL architectures/models.
- Abstraction (for training basics, generators, preprocessors, decoding, callbacks etc.) allows definite interfaces which eases automatization.
- Custom layers are re-usable and clear.
- **Many interesting/hard (Keras) problems are solved:**
  - Parameter Tying
    - Tying embedders and projector altogether.
    - Handling save/load case!
    - Parametrized call usage to allow a layer to behave conditionally.
  - Domain-agnostic training with inner-outer generators.
  - Model and callback serialization.
  - Multi-head attention on same tensor.
  - Positional encoding - Interleaving Trick.
  - Padding/Additional masking when computing attention attention.
  - Custom loss with custom padding mask.
  - Subword embeddings.
  - Beam Search Decoding.
  