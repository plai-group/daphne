(ns daphne.hy
  "Transpilation code for hy-lang. This code is to address amortized
  inference with the help of the Python optimization stack build on pytorch."
  (:require [daphne.core :as fc]
            [daphne.invert :refer [faithful-program-inversion]]
            [daphne.symbolic-simplify :refer [symbolic-simplify]]
            [loom.graph :as g]
            [backtick :refer [template]]
            [clojure.pprint :refer [pprint]]
            [clojure.core.match :refer [match]]
            [clojure.walk :refer [postwalk]]
            [clojure.string :as str]
            [clojure.java.shell :as sh]))


(def ^:dynamic *gensyms*)

(def ^:dynamic *offset* 0)

(defn- combined-gensym
  "Generates symbol and integer gensyms to refer to tensor positions later."
  [sym]
  ;; filter out all internal gensyms
  (if-not (#{"sample" "observe"} sym)
    (gensym sym)
    (let [f (first @*gensyms*)]
      (swap! *gensyms* rest)
      (if (= sym "observe")
        (symbol (str sym "_" (- f *offset*)))
        (symbol (str sym "_" f))))))

(defn- gensym->cursor [n]
  (let [[sym index] (str/split (name n) #"_")]
    (when (and (#{"sample" "observe"} sym) index)
      (let [sym (symbol sym)
            index (Integer/parseInt index)]
        [sym index]))))

(defn faithful-adjacency-list
  "Returns integer indices for latents and observes as an adjacency list. "
  [code]
  (binding [*gensyms* (atom (range))
           daphne.gensym/*my-gensym* combined-gensym]
    (let [[_ G _] (fc/code->graph code)
          H (:H (faithful-program-inversion G))]
      (sort
       (for [[k vs] (:adj H)
             v vs]
         ;; push symbols to literal numbers
         [(second (gensym->cursor v)) (second (gensym->cursor k))])))))

(defn prog-and-faithful-adjacency-list
  "Returns integer indices for latents and observes as an adjacency list. "
  [code]
  (binding [*gensyms* (atom (range))
            daphne.gensym/*my-gensym* combined-gensym]
    (let [prog (fc/code->graph code)
          [_ G _] prog
          H (:H (faithful-program-inversion G))]
      {:prog prog
       :faithful-adjacency
       (sort
        (for [[k vs] (:adj H)
              v vs]
          ;; push symbols to literal numbers
          [(second (gensym->cursor v)) (second (gensym->cursor k))]))})))

(defn expressions->hy [instructions]
  (postwalk #(get {'normal 'Normal
                  'flip 'Bernoulli
                  'laplace 'Laplace
                  'if 'if
                  '= '=
                  'and 'and
                  'or 'or
                  true 'True
                  false 'False
                  nil 'None
                  'sample* '.sample
                  'observe* '.observe
                  'tanh 'safe_tanh} % %)
            instructions))

(defn- code->hy-instructions [code]
  (->> code
       fc/code->graph
       fc/graph->instructions
       expressions->hy
       butlast))

(defn- observe->sample [n]
  (if (and (seq? n)
           (= (first n) '.observe))
    `(~'.sample ~(second n))
    n))

(defn gensym-comp [a b]
  (let [[_ a-index] (str/split (name a) #"_")
        [_ b-index] (str/split (name b) #"_")
        a-index (Integer/parseInt a-index)
        b-index (Integer/parseInt b-index)]
    (compare a-index b-index)))

(defn create-sampler
  "Creates a sampling routine from the joint distribution."
  [code]
  (binding [*gensyms* (atom (range))
            daphne.gensym/*my-gensym* combined-gensym]
    (let [instructions (code->hy-instructions code)]
      (postwalk
       observe->sample
       `(do ~@(concat
               (map (fn [[sym exp]] `(~'setv ~sym ~exp)) instructions)
               `([;; prior
                  (torch.tensor ~(->> instructions
                                      (map first)
                                      (filter #(re-find #"sample*" (name %)))
                                      (sort gensym-comp)
                                      vec)
                                #_(vec (sort (map first (filter #(re-find #"sample*" (name (first %)))
                                                               instructions)))))
                  ;; likelihood
                  (torch.tensor ~(->> instructions
                                      (map first)
                                      (filter #(re-find #"observe*" (name %)))
                                      (sort gensym-comp)
                                      vec)
                                #_(vec (sort (map first (filter #(re-find #"observe*" (name (first %)))
                                                               instructions)))))])))))))

(defn symbol->slice
  "Extracts the index of a sample or observe expression and generates a slice
  based access into the corresponding tensor."
  [n]
  (if (symbol? n)
    (if-let [[sym index] (gensym->cursor n)]
      ;; advanced numpy style slicing
      `(~'get ~sym [(~'slice ~'None) ~index])
      n)
    n))

(defn- instructions->log_prob
  "Turns instructions with sample or observe statements into log probability
  accumulator."
  [acc-sym exps]
  `(do ~@(concat
          [`(~'setv ~acc-sym ~'(torch.zeros (get sample.shape 0)))]
          (map (fn [[csym exp]]
                 `(~'+= ~acc-sym (.log_prob ~(second exp) ~csym)))
               exps)
          [acc-sym])))

(defn create-log-prob [code sym-regex acc-sym offset]
  (binding [*gensyms* (atom (range))
            *offset* offset
            daphne.gensym/*my-gensym* combined-gensym]
    (->> code
         code->hy-instructions
         (filter #(re-find sym-regex (name (first %))))
         (instructions->log_prob acc-sym)
         (postwalk symbol->slice))))

(defn create-prior [code]
  (create-log-prob code #"sample*" 'log_prior 0))

(defn create-likelihood [code dim-prior]
  (create-log-prob code #"observe*" 'log_likeli dim-prior))

(defn create-rand-adjacency [target-count dim-latent dim-condition]
  (loop [prob 0.5
         i 0]
    (let [rand-adjacency (distinct
                          (concat
                           (for [i (range dim-latent)
                                 j (range (+ dim-latent dim-condition))
                                 :when (< (rand) prob)]
                             [i j])))
          num-conns (count rand-adjacency)]
      (when (= i 1000)
        (throw (ex-info "Cannot find random graph. " {:final-prob prob
                                                      :num-conns num-conns
                                                      :target-count target-count})))
      (cond (= num-conns target-count)
            (vec rand-adjacency)

            (< num-conns target-count)
            (recur (min (max (+ prob 0.01) 0) 1) (inc i))

            :else
            (recur (min (max (- prob 0.01) 0) 1) (inc i))))))


(defn code->hy [code]
  (let [prog (fc/code->graph code)
        [rho G E] prog
        ;;faithful-inversion (faithful-program-inversion G)
        dim-condition (count (:Y G))
        dim-latent (- (count (:V G)) dim-condition)
        self-adjacency (for [i (range dim-latent)] [i i])
        faithful-adjacency (concat (faithful-adjacency-list code) self-adjacency)
        rand-adjacency (concat (create-rand-adjacency (count faithful-adjacency)
                                                      dim-latent
                                                      dim-condition)
                               self-adjacency)]

    (template
     ;; hy-lang syntactic embedding
     ;; chose immutable Object for scoping
     (defclass FlowModel []
       "Autogenerated, do not edit. "
       (setv dim_latent ~dim-latent)
       (setv dim_condition ~dim-condition)
       (setv faithful_adjacency ~(vec faithful-adjacency))
       (setv rand_adjacency ~(vec rand-adjacency))
       ;; for debugging/tracking purposes
       (setv src ~(with-out-str (clojure.pprint/pprint code)))

       ;; old hy syntax (2019?)
       ;; "Autogenerated, do not edit. "
       ;; [dim_latent ~dim-latent
       ;;  dim_condition ~dim-condition
       ;;  src ~(with-out-str (clojure.pprint/pprint code)) ;; for debugging/tracking purposes
       ;;  faithful_adjacency ~(vec faithful-adjacency)
       ;;  rand_adjacency ~(vec rand-adjacency)]

       ;; use Giry Monad interface (?)
       ;; Void -> [Tensor[dim_latent], Tensor[dim_condition]]
       (defn sample [self]
         ~(create-sampler code))

       ;; Tensor[dim_latent], Tensor[dim_condition] -> LogProb
       (defn log_likelihood [self sample observe]
         ~(create-likelihood code dim-latent))

       ;; Tensor[dim_latent] -> LogProb
       (defn log_prior [self sample]
         ~(create-prior code)))))) 

(def python-imports
  '((import torch)
    (import math)
    (import [torch.distributions [Normal Bernoulli Laplace Uniform]])))


(defn foppl->python
  "Compiles FOPPL to Python flow representation."
  [code]
  (let [hy-code       (code->hy code)
        python-output (->>
                       (with-out-str
                         (doseq [i python-imports]
                           (pprint i))
                         (println)
                         (pprint
                          '(defn safe-tanh [x]
                             (if (isinstance x float)
                               (math.tanh x)
                               (torch.tanh x))))
                         (println)
                         (pprint hy-code))
                       (sh/sh "hy2py" :in)
                       :out)]
    ;; TODO HACK patch for stupid unary + injection in recent hy 0.18+ versions,
    ;; which is breaking pytorch tensor addition :(
    (.replace python-output "(+(" "((")))
