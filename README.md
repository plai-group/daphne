# daphne

This is the codebase accompanying our [AISTATS 2020
paper](https://github.com/mlresearch/v108/tree/gh-pages/weilbach20a)

~~~bibtex
@inproceedings{DBLP:conf/aistats/WeilbachBWH20,
  author    = {Christian Weilbach and
               Boyan Beronov and
               Frank Wood and
               William Harvey},
  editor    = {Silvia Chiappa and
               Roberto Calandra},
  title     = {Structured Conditional Continuous Normalizing Flows for Efficient
               Amortized Inference in Graphical Models},
  booktitle = {The 23rd International Conference on Artificial Intelligence and Statistics,
               {AISTATS} 2020, 26-28 August 2020, Online [Palermo, Sicily, Italy]},
  series    = {Proceedings of Machine Learning Research},
  volume    = {108},
  pages     = {4441--4451},
  publisher = {{PMLR}},
  year      = {2020},
  url       = {http://proceedings.mlr.press/v108/weilbach20a.html},
  timestamp = {Mon, 29 Jun 2020 18:03:58 +0200},
  biburl    = {https://dblp.org/rec/conf/aistats/WeilbachBWH20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
~~~

It is built on a compiler following the book [An Introduction to Probabilistic
Programming](https://arxiv.org/abs/1809.10756) and integrates the [faithful
inversion algorithm](https://arxiv.org/abs/1712.00287), which was inspired by
the book "Probabilistic Graphical Models" Daphne Koller and Nir Friedman.

The compiler transforms a first order programming language in
[Clojure](https://clojure.org/) syntax into an intermediate graph representation
that lends itself to different inference methods and automatic translation into
an amortizing neural network as is done in this repository.


## Usage

To run this compiler you need to have [leiningen](https://leiningen.org/) and a
JVM for Java 8 or later installed.

Make sure that you have [hy-lang](https://hylang.com/) for Python 3 installed
and check for `hy2py` on your path. The Python dependencies are documented in
`requirements.txt`.

To generate the arithmetic circuit example from the paper provided as the
following Daphne program,

~~~clojure
(let [z0 (sample (laplace 20.0 2.0))
      z1 (sample (laplace 10.0 2.0))

      z2 (sample (normal (+ z0 z1) 0.1))
      z3 (sample (normal (* z0 z1) 0.1))

      z4 (sample (normal 7.0 2.0))
      z5 (sample (normal z3 z4))

      x0 (observe (normal (+ z3) 0.1))
      x1 (observe (normal z5 0.1))]
      [z0 z1 z2 z3 z4 z5])
~~~

you can run 

~~~bash
lein run programs/arithmetic_circuit.daphne autogen_arithmetic_circuit.py
~~~

with the input program as the first argument and the output program path as the
second argument, Note that we have prepended `autogen_` here, which is necessary
to distinguish this model from the ones we already included in `models.py` in
our Python codebase. You can compile `convolution.daphne` in the same way.

~~~bash
python3 main.py with train_steps=100 gmodel_name=autogen_convolution -F ./runs
~~~

Sacred will store the resulting files in the `./runs` directory.

We provide plotting code for the loss curves in the paper in the `./plots`
directory (WIP).

The Python code base for training is derived from
[FFJORD](https://github.com/rtqichen/ffjord/),



## Augmentation

To augment nodes in the graphical model as described in our paper you can
provide the following argument

~~~bash
python3 main.py with train_steps=100 gmodel_name=autogen_convolution to_augment=[0,1,2,3,4,5] -F ./runs
~~~

Which will add one augmenting node to the nodes one to five.


## TODO

- provide sacred local filesystem loader for the plot routines
- provide plotting code for deconvolution figure

## License for Clojure compiler

Copyright © 2018-2020 Christian Weilbach

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.

This Source Code may also be made available under the following Secondary
Licenses when the conditions for such availability set forth in the Eclipse
Public License, v. 2.0 are satisfied: GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or (at your
option) any later version, with the GNU Classpath Exception which is available
at https://www.gnu.org/software/classpath/license.html.
