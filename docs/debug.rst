
=======
Debug
=======

Debug hanging multi-process
---------------------------

assuming ``pip install py-spy`` has been performed already

To get all the tracebacks at once:

.. code-block:: bash

   pgrep -P $(pgrep -P $(pgrep -n arctic_training) | tail -1) | xargs -I {} py-spy dump --pid {}

but probably we want just the first few calls in ``MainSthread`` thread:

.. code-block:: bash

   pgrep -P $(pgrep -P $(pgrep -n arctic_training) | tail -1) | xargs -I {} py-spy dump --pid {} | grep -A5 MainThread
