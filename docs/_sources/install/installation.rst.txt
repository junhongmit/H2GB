Installation
============

:h2gb:`ℋ²GB` is available for :python:`Python>=3.6`, and installation is easy!

.. note::
   We do not recommend installation as a root user on your system :python:`Python`.
   Please setup a virtual environment, *e.g.*, via :conda:`null` `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_, or create a `Docker image <https://www.docker.com/>`_.

Installing :h2gb:`ℋ²GB` requires external library like :pytorch:`PyTorch` and :pyg:`PyG`. You can easily set up a conda environment to use :h2gb:`ℋ²GB`. For example, in the following, we create a virtual environment named `H2GB`:

.. code-block:: none

   conda create -n H2GB python=3.9 -y
   conda activate H2GB

Then, Install the required packages using the following commands:

.. code-block:: none

   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install pyg -c pyg

Now you are ready for installing :h2gb:`ℋ²GB`!

Installation via PyPi
---------------------

To install :h2gb:`ℋ²GB`, the easiest way is to simply run:

.. code-block:: none

   pip install torch_geometric


Installation from Github
------------------------

In case a specific version of the dependency is not supported, you can alternatively install by cloning the original Github repository:

.. code-block:: none

   git clone https://github.com/junhongmit/H2GB.git

