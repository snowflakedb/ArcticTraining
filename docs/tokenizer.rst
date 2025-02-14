.. _tokenizer:

=================
Tokenizer Factory
=================

The Tokenizer Factory is responsible for generating the tokenizer used in
training. The tokenizer is in turn used by the :class:`~.DatasetFactory` to
tokenize the input data. A Tokenizer Factory can be created by inheriting from
the :class:`~.TokenizerFactory` class and implementing the
:meth:`~.TokenizerFactory.create_tokenizer` method.

.. autoclass:: arctic_training.tokenizer.factory.TokenizerFactory

Attributes
----------

Similar to other Factory classes in ArcticTraining, the TokenizerFactory class
must have a :attr:`~.TokenizerFactory.name` attribute that is used to identify
the factory when registering it with ArcticTraining and a
:attr:`~.TokenizerFactory.config` attribute type hint that is used to validate
the config object passed to the factory.

Properties
----------

A TokenizerFactory has several attributes that can be used to access information
about the trainer and distributed state at runtime, including
:attr:`~.TokenizerFactory.device`, :attr:`~.TokenizerFactory.trainer`,
:attr:`~.TokenizerFactory.world_size`, and :attr:`~.TokenizerFactory.global_rank`.

Methods
-------

TokenizerFactories have just one method that must be defined:
:meth:`~.TokenizerFactory.create_tokenizer`. The
:meth:`~.TokenizerFactory.create_tokenizer` method should return the tokenizer
object created using the passed tokenizer config.

HuggingFace Tokenizer Factory
-----------------------------

A custom tokenizer factory can be created from the :class:`~.TokenizerFactory`
building block, but ArcticTraining also comes with a TokenizerFactory
implementation that can be used out of the box: :class:`~.HFTokenizerFactory`.
The :class:`~.HFTokenizerFactory` will load tokenizers from HuggingFace Hub
given a path to a local repo or the tokenizer name.
