.. ArcticTraining documentation master file, created by
   sphinx-quickstart on Thu Dec 12 16:59:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ArcticTraining documentation
============================

ArcticTraining is a Python library designed to make prototyping and
experimenting with LLM training workflows as simple, flexible, and efficient as
possible. Whether you’re fine-tuning models with standard approaches or
exploring cutting-edge training algorithms, ArcticTraining provides the tools
and structure to streamline your work and focus on what matters most.

Why ArcticTraining?
-------------------

At the heart of ArcticTraining is a modular design built around building
blocks—factory objects that handle the creation and management of training
assets like models, datasets, optimizers, and checkpoint engines. These
components are connected through well-defined interfaces that ensure
compatibility and make it easy to swap, customize, or extend any part of the
training workflow.

To further reduce boilerplate and simplify customization, ArcticTraining
includes a robust callback system that wraps core class methods. This system
allows you to tweak and extend functionality without modifying the underlying
code, making complex adjustments straightforward and manageable.

Key Features
------------

- **Extensible Building Blocks**: Core components like trainers, model factories,
  and data sources come with narrow, well-defined interfaces, making it easy to
  reuse and customize them for your specific needs.
- **Powerful Callback System**: Extend or modify behavior by injecting custom logic
  into key points of the training workflow without rewriting existing classes.
- **Seamless CLI Integration**: Define training configurations in a YAML file and
  run them with a simple command, optionally including custom Python code for
  advanced use cases.
- **Pre-Built Trainers**: Start with examples like the Supervised Fine-Tuning (SFT)
  trainer, which showcases how minimal changes can create a fully functional
  training pipeline.
- **Rapid Prototyping**: Experiment with new training algorithms by swapping
  components or redefining specific methods, all while ensuring compatibility
  across the workflow.

Quick Start
-----------

To get started with ArcticTraining check out the :ref:`quick start guide <quickstart>`

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick-start
   install
   usage
   config
   callbacks
   trainer
   data
   checkpoint
   model
   tokenizer
   optimizer
   scheduler
   synth
