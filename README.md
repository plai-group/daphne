# Daphne

> Named after Daphne, the naiad nymph from Greek mythology who transformed into a laurel tree to escape Apollo. Like the mythological transformation that preserved Daphne's essence while changing her form, our compiler transforms probabilistic programs into graphical models of probability flows while preserving their semantic meaning. As water nymphs direct the flow of rivers and streams, our Daphne system guides the flow of probability distributions through transformations that preserve their fundamental properties while enabling efficient inference and representation.

This project provides the `daphne` probabilistic programming compiler. It is
built closely following the book [An Introduction to Probabilistic
Programming](https://arxiv.org/abs/1809.10756). 

The compiler transforms a first order programming language, i.e. a language
without recursion and therefore a fixed number of random variables, in
[Clojure](https://clojure.org/) syntax into an intermediate graph representation
that lends itself to different inference methods, JSON exports and automatic
translation to an amortizing neural network as is described further down.

## Usage

To run this compiler you need to have a JVM and the [clj](https://clojure.org/guides/deps_and_cli) command line tool installed. To see all the options provided by the compiler run:

~~~clojure
clj -M:run graph --help
~~~

in the directory where you have checked out this repository.

To generate the arithmetic circuit program from our paper provided as the
following file in `programs/arithmetic_circuit.daphne`,

~~~clojure
(let [z0 (sample (laplace 20.0 2.0))
      z1 (sample (laplace 10.0 2.0))

      z2 (sample (normal (+ z0 z1) 0.1))
      z3 (sample (normal (* z0 z1) 0.1))

      z4 (sample (normal 7.0 2.0))
      z5 (sample (normal z3 z4))

      x0 (observe (normal (+ z3) 0.1) 0.2)
      x1 (observe (normal z5 0.1) -3.5)]
  [z0 z1 z2 z3 z4 z5])
~~~

you can run 

~~~bash
clj -M:run graph -i programs/arithmetic_circuit.daphne -o output.json
~~~

with the input program as the first argument and the output path as the second
argument. By default the compiler emits JSON, but it can also emit the more
expressive [edn](https://github.com/edn-format/edn) format of Clojure. This will
emit a data structure describing the graphical model structure of the following
form:

~~~javascript
[{},
 {"V":["sample5","sample0","observe6","observe7","sample4","sample2","sample1","sample3"],
  "A":{"sample0":["sample2","sample3"],
       "sample1":["sample2","sample3"],
       "sample4":["sample5"],
       "sample3":["sample5","observe6"],
       "sample5":["observe7"]},
  "P":{"sample0":["sample*",["laplace",20.0,2.0]],
       "sample1":["sample*",["laplace",10.0,2.0]],
       "sample2":["sample*",["normal",["+","sample0","sample1"],0.1]],
       "sample3":["sample*",["normal",["*","sample0","sample1"],0.1]],
       "sample4":["sample*",["normal",7.0,2.0]],
       "sample5":["sample*",["normal","sample3","sample4"]],
       "observe6":["observe*",["normal",["+","sample3"],0.1],0.2],
       "observe7":["observe*",["normal","sample5",0.1],-3.5]},
  "Y":{"observe6":0.2,"observe7":-3.5}},
 ["sample0","sample1","sample2","sample3","sample4","sample5"]]
~~~

The first entry in the triple contains all functions defined in the input file,
here none. The second contains a dictionary of the graphical model with `"V"`
denoting all random variable vertices, `"A"` the forward pointing adjacency
information, `"P"` the symbolic expressions of the link functions in the
deterministic target language and `"Y"` all observe statements. The last entry
in the triple is the return value of the program, here corresponding to
`[z0 z1 z2 z3 z4 z5]`.


Alternatively you can emit a desugared AST (see the book) by using the `desugar` command:

~~~bash
clj -M:run desugar -i programs/arithmetic_circuit.daphne -o output.json
~~~

~~~javascript
[["let",["z0",["sample",["laplace",20.0,2.0]]],
  ["let",["z1",["sample",["laplace",10.0,2.0]]],
   ["let",["z2",["sample",["normal",["+","z0","z1"],0.1]]],
    ["let",["z3",["sample",["normal",["*","z0","z1"],0.1]]],
     ["let",["z4",["sample",["normal",7.0,2.0]]],
      ["let",["z5",["sample",["normal","z3","z4"]]],
       ["let",["x0",["observe",["normal",["+","z3"],0.1],0.2]],
        ["let",["x1",["observe",["normal","z5",0.1],-3.5]],
         ["vector","z0","z1","z2","z3","z4","z5"]]]]]]]]]]
~~~

Note that the let binding is now nested with an expression for each individual
binding and the vector in the final expression is explicitly denoted through a
starting `"vector"` expression to distinguish it from the normal lists. This
format makes it easy to write custom interpreters in any language that can read
JSON.

If you are using `daphne` compiler for the UBC CS532: Probabilistic Programming course
then you are good to go as long as you can run the `lein` commands described above.
You do **not** need to have the "python export" functionality, described below, working.

---

### Python Export

To create a Python class instead of a JSON graphical model you can use the
`python-class` command,

~~~bash
clj -M:run python-class -i programs/arithmetic_circuit.daphne -o autogen_arithmetic_circuit.py
~~~

Note that we have prepended `autogen_` here, which is necessary to distinguish
this model from the ones we already included in `models.py` in our Python
codebase. If you do not use our amortization setup you can pick the filename
freely. You can compile `convolution.daphne` in the same way.


Make sure that you have [hy-lang](https://hylang.org/) for Python 3 installed
and check for `hy2py` on your path. The Python dependencies are documented in
`requirements.txt`.

### CUDA Export

To export to CUDA you can run the following command

~~~bash
clj -M:run cuda -i programs/arithmetic_circuit.daphne -o arithmetic_circuit.cu
~~~

If you have the CUDA compiler nvcc installed you can then compile with

~~~bash
nvcc -o arithmetic_circuit arithmetic_circuit.cu
~~~

and run it to draw 10 samples

~~~bash
./arithmetic_circuit 10
~~~

This will print 10 times value for all sample statements and the log probabilities for each observe statement. If you want to use the data in a different format you can edit [main.cu](resources/cuda/main.cu) to handle IO differently, e.g. torch export.

### Continuous Normalizing Flow training

We can now use the provided Python files to train a neural network with

~~~bash
python3 main.py with train_steps=100 gmodel_name=autogen_convolution -F ./runs
~~~

Sacred will store the resulting files in the `./runs` directory.

We provide plotting code for the loss curves in the paper in the `./plots`
directory (WIP).

The Python code base for training is derived from
[FFJORD](https://github.com/rtqichen/ffjord/),


### Augmentation

To augment nodes in the graphical model as described in our paper you can
provide the following argument

~~~bash
python3 main.py with train_steps=100 gmodel_name=autogen_convolution to_augment=[0,1,2,3,4,5] -F ./runs
~~~

Which will add one augmenting node to the nodes one to five.

## Extended Semantics

The compiler consists of multiple passes and provides an extended partial
evaluation fix point operator that allows expressions such as

~~~clojure
(foreach (count input) ...)
~~~

where input is some value provided to the program and the loop counter
expression can be partially evaluated to an integer at compile time. This
facilitates the provision of a small linear algebra and neural network library
in the [linalg.clj](src/daphne/linalg.clj) file (WIP).

The compiler also integrates the structural [faithful inversion
algorithm](https://arxiv.org/abs/1712.00287) for graphical models, which was
inspired by the book "Probabilistic Graphical Models" by Daphne Koller and Nir
Friedman. This is used by the amortizing Python code.

## Possible Extensions

Clojure recently got fairly seamless bindings including zero copy support to
Clojure tensor libraries for
[Python](https://github.com/clj-python/libpython-clj) and
[Julia](https://github.com/cnuernber/libjulia-clj). This allows interleaving the
expressive metaprogramming facilities of Clojure during the compilation with
many numerical algorithms and facilities of these ecosystems. Clojure itself
also has flexible high performance numerical libraries through the
[uncomplicate](https://uncomplicate.org/) ecosystem that operate on the same
memory layouts as numpy and Julia ndarrays.

## Teaching Examples

Since the codebase is also used to teach the probabilistic programming course
CPSC 539W of [Frank Wood](https://www.cs.ubc.ca/~fwood/) at UBC, it also
contains a few examples and many unit tests corresponding to former exercises.

## Citation

This is also the codebase accompanying our [AISTATS 2020
paper](https://github.com/mlresearch/v108/tree/gh-pages/weilbach20a)

~~~bibtex
@inproceedings{DBLP:conf/aistats/WeilbachBWH20,
  author    = {Christian Weilbach and
               Boyan Beronov and
               William Harvey and
               Frank Wood},
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
