.. _callbacks:

=========
Callbacks
=========

A robust callback system is included with ArcticTraining to allow for easy
customization and extension of the base building blocks. The
:ref:`Trainer<trainer>`, :ref:`Data Factory<data>`, :ref:`Checkpoint
Engine<checkpoint>`, :ref:`Model Factory<model>`, :ref:`Optimizer
Factory<optimizer>`, :ref:`Scheduler Factory<scheduler>`, and :ref:`Tokenizer
Factory<tokenizer>` classes all allow for extensibility via callbacks.

The callback system is implemented in the
:class:`~arctic_training.callback.mixin.CallbackMixin` that each of the above
building blocks inherits.

.. autoclass:: arctic_training.callback.mixin.CallbackMixin

Defining a Custom Callback
--------------------------

A callback consists of a string that indicates the callback event and a callable
object that is executed when the callback is triggered. For example, a callback
that logs when an object is initalized can be defined like so:

.. code-block:: python

    from arctic_training import logger

    def pre_init_callback(self):
        logger.info(f"{self.__class__.__name__} initializing")

    init_cb = ("pre-init", pre_init_callback)
    trainer.callbacks.append(init_cb)

Callback Events
^^^^^^^^^^^^^^^

Callback event strings take the form of ``{pre|post}-{event}`` where ``event``
is the name of a method in the class that inherits from
:class:`~arctic_training.callback.mixin.CallbackMixin`. For example, the
``pre-init`` event is triggered before the object is initialized (i.e., before
``__init__`` is called) and the ``post-init`` event is triggered after the
object is initialized (i.e., after ``__init__`` is called).

Any of the methods described in the documentation for :ref:`Trainer<trainer>`,
:ref:`Data Factory<data>`, :ref:`Checkpoint Engine<checkpoint>`, :ref:`Model
Factory<model>`, :ref:`Optimizer Factory<optimizer>`, :ref:`Scheduler
Factory<scheduler>`, and :ref:`Tokenizer Factory<tokenizer>` can have a callback
added.

For example, in the :ref:`Trainer<trainer>` class, the ``train``, ``epoch``,
``step``, ``loss``, and ``checkpoint`` methods can all have callbacks added.
This provides for a high degree of customization and extension of the training
loop for existing and custom trainers built with ArcticTraining.

Callback Functions
^^^^^^^^^^^^^^^^^^

The callable object that is executed when the callback is triggered should take
as input the object that the callback is attached to (i.e., ``self``).

``pre-`` callback functions may also accept any combination of arguments that
the method it is attached to accepts. For example a callback for the ``loss``
method of a Trainer could take either no arguments or the training batch as
input:

.. code-block:: python

    def pre_loss_callback_1(self):
        print("Loss callback triggered")

    def pre_loss_callback_2(self, batch):
        print(f"Loss callback triggered with batch {batch}")
        return batch

It's important to note that if a callback function accepts an argument as input,
it would also return that argument (in the same order as it was passed).

``post-`` callback functions should also accept a ``return_value`` argument
that contains the return value of the method it is attached to:

.. code-block:: python

    def post_loss_callback(self, return_value):
        print(f"Loss callback triggered with return value {return_value}")
        return return_value

Adding Callbacks
----------------

To add a callback to a Trainer, Data Factory, Model Factory or other
ArcticTraining class, the callback tuple can be added to the ``callbacks``
attribute of the object. For example, to a pre-step callback to a custom Trainer:

.. code-block:: python

    from arctic_training import Trainer

    def pre_step_callback(self, batch):
        print(f"Step callback triggered with batch {batch}")
        return batch

    class MyTrainer(Trainer):
        name = "my_trainer"
        callbacks = [("pre-step", pre_step_callback)]

Callbacks can also be added directly to the object by defining a method with a
name of the form ``{pre|post}_{event}_callback``. To add an equivalent callback
in this way:

.. code-block:: python

    class MyTrainer(Trainer):
        name = "my_trainer"

        def pre_step_callback(self, batch):
            print(f"Step callback triggered with batch {batch}")
            return batch

Callback Inheritence
--------------------

Callbacks methods are inherited from parent classes and can be chained together.
Take the following example:

.. code-block:: python

    class TrainerA(Trainer):
        name = "trainer_a"

        def post_epoch_callback(self):
            print("Trainer A post epoch callback")

    class TrainerB(TrainerA):
        name = "trainer_b"

        callbacks = [("post-epoch", lambda self: print("Trainer B post epoch callback 1"))]

        def post_epoch_callback(self):
            print("Trainer B post epoch callback 2")

In this case, ``TrainerB`` will inherit the original post-epoch callback from
``TrainerA`` and add two additional callbacks. When ``step()`` is run during
training, the following would be the output:

.. code-block:: shell

    Trainer A post epoch callback
    Trainer B post epoch callback 1
    Trainer B post epoch callback 2
