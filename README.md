# End-to-end neural relation extraction using deep biaffine attention

This program provides an implementation of a neural network model for joint extraction of named entities and their semantic relations,  as described in [my paper](https://arxiv.org/abs/1812.11275):

    @InProceedings{NguyenV_ECIR2019,
    author    = {Dat Quoc Nguyen and Karin Verspoor},
    title     = {{End-to-end neural relation extraction using deep biaffine attention}},
    booktitle = {Proceedings of the 41st European Conference on Information Retrieval},
    year      = {2019}
    }
    
### Installation

jointRE requires the following software packages:

* `Python 2.7`
* [`DyNet` v2.0](http://dynet.readthedocs.io/en/latest/python.html)

      $ virtualenv -p python2.7 .DyNet
      $ source .DyNet/bin/activate
      $ pip install cython numpy
      $ pip install dynet==2.0.3

Once you installed the prerequisite packages above, you can clone or download (and then unzip) jointRE.

### Usage

[jNERE](https://github.com/datquocnguyen/jointRE/tree/master/jNERE) and [jECRE](https://github.com/datquocnguyen/jointRE/tree/master/jECRE) correspond to two evaluation setup scenarios  NER&RC and EC&RC used in my paper, respectively.  Checkout `run.sh` in `scripts` folder. It should be self-explanatory.


### LICENSE
This software is provided AS IS with NO SUPPORT. These programs have no warranty, guarantee, express or implied representation of any kind whatsoever. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed.

I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.
