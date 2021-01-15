(ns daphne.reverse-diff
  "Reverse mode auto-diff."
  (:require [anglican.runtime :refer [observe* normal]]
            [daphne.gensym :refer [*my-gensym*]])
  (:import [anglican.runtime normal-distribution]))

;; The following code so far follows
;; http://www-bcl.cs.may.ie/~barak/papers/toplas-reverse.pdf
;; and Griewank A. Evaluating derivatives. 2008.


;; Proposed roadmap

;; 1. generalization
;;    + function composition (boundary type)
;;    + arbitrary Anglican style nested values
;;    + external primitive functions
;;    + integrate into CPS trafo of Anglican
;;
;; 2. implement tape version through operator overloading
;;    following diffsharp
;;
;; 3. linear algebra support
;;    + extend to core.matrix
;;    + support simple deep learning style composition
;;
;; 4. performance optimizations

(set! *warn-on-reflection* true)

(comment
  (set! *unchecked-math* :warn-on-boxed))


;; some aliasing for formula sanity

(def ** (fn [x p] (Math/pow x p)))

(def sqrt (fn [x] (Math/sqrt x)))

(def log (fn [x] (Math/log x)))

(def exp (fn [x] (Math/exp x)))

(def pow (fn [x p] (Math/pow x p)))

(def sin (fn [x] (Math/sin x)))

(def cos (fn [x] (Math/cos x)))


(defn normpdf [x mu sigma]
  (let [x (double x)
        mu (double mu)
        sigma (double sigma)]
    (+ (- (/ (* (- x mu) (- x mu))
             (* 2.0 (* sigma sigma))))
       (* -0.5 (Math/log (* 2.0 (* 3.141592653589793 (* sigma sigma))))))))


(defn term? [exp]
  (or (number? exp)
      (symbol? exp)))

(defn dispatch-exp [exp p]
  (assert (seq? exp) "All differentiation happens on arithmetic expressions.")
  (assert (zero? (count (filter seq? exp))) "Differentiation works on flat (not-nested) expressions only.")
  (keyword (name (first exp))))


;; derivative definitions

(defmulti partial-deriv dispatch-exp)

(defmethod partial-deriv :+ [[_ & args] p]
  (seq (into '[+]
             (reduce (fn [nargs a]
                       (if (= a p)
                         (conj nargs 1)
                         nargs))
                     []
                     args))))


(defmethod partial-deriv :- [[_ & args] p]
  (seq (into '[-]
             (reduce (fn [nargs a]
                       (if (= a p)
                         (conj nargs 1)
                         (conj nargs 0)))
                     []
                     args))))


(defmethod partial-deriv :* [[_ & args] p]
  (let [pn (count (filter #(= % p) args))]
    (seq (into ['* pn (list 'pow p (dec pn))]
               (filter #(not= % p) args)))))


(defmethod partial-deriv :/ [[_ & [a b]] p]
  ;; TODO support any arity
  (if (= a p)
    (if (= b p)
      0
      (list '/ 1 b))
    (if (= b p)
      (list '- (list '* a (list 'pow b -2)))
      0)))

(defmethod partial-deriv :sin [[_ a] p]
  (if (= a p)
    (list 'cos a)
    0))

(defmethod partial-deriv :cos [[_ a] p]
  (if (= a p)
    (list 'sin a)
    0))

(defmethod partial-deriv :exp [[_ a] p]
  (if (= a p)
    (list 'exp a)
    0))

(defmethod partial-deriv :log [[_ a] p]
  (if (= a p)
    (list '/ 1 a)
    0))


(defmethod partial-deriv :pow [[_ & [base expo]] p]
  (if (= base p)
    (if (= expo p)
      ;; TODO p^p only defined for p > 0
      (list '* (list '+ 1 (list 'log p))
            (list 'pow p p))
      (list '* expo (list 'pow p (list 'dec expo))))
    (if (= expo p)
      (list '* (list 'log base) (list 'pow base p))
      0)))


(defmethod partial-deriv :normpdf [[_ x mu sigma] p]
  (cond (= x p)
        (list '*
              (list '- (list '/ 1 (list '* sigma sigma)))
              (list '- x mu))

        (= mu p)
        (list '*
              (list '- (list '/ 1 (list '* sigma sigma)))
              (list '- mu x))

        (= sigma p)
        (list '-
              (list '*
                    (list '/ 1 (list '* sigma sigma sigma))
                    (list 'pow (list '- x mu) 2))
              (list '/ 1 sigma))

        :else
        0))


(def empty-tape {:forward [] :backward []})


(defn adjoint-sym [sym]
  (symbol (str sym "_")))

(defn tape-expr
  "The tape returns flat variable assignments for forward and backward pass.
   It allows multiple assignments following Griewank p. 125 or chapter 3.2.

  Once lambdas are supported this should be A-normal form of code."
  [bound sym exp tape]
  (cond (and (seq? exp)
             (= (first exp) 'if))

        (let [[_ condition then else] exp
              {:keys [forward backward]} tape
              then-s (*my-gensym* "then")
              else-s (*my-gensym* "else")
              {then-forward :forward
               then-backward :backward} (tape-expr bound then-s then empty-tape)
              {else-forward :forward
               else-backward :backward} (tape-expr bound else-s else empty-tape)
              if-forward (concat (map (fn [[s e]] [s (list 'if condition e 0)])
                                      then-forward)
                                 (map (fn [[s e]] [s (list 'if-not condition e 0)])
                                      else-forward))
              if-backward (concat (map (fn [[s e]] [s (list 'if condition e s)])
                                       then-backward)
                                  (map (fn [[s e]] [s (list 'if-not condition e s)])
                                       else-backward))]
          {:forward (vec (concat forward
                                 if-forward
                                 [[sym (list 'if condition then-s else-s)]]))
           :backward (vec (concat backward
                                  if-backward
                                  [[(adjoint-sym then-s) (adjoint-sym sym)]
                                   [(adjoint-sym else-s) (adjoint-sym sym)]]))})

        :else
        (let [[f & args] exp
              new-gensyms (atom [])
              nargs (map (fn [a] (if (term? a) a
                                    (let [ng (*my-gensym* "v")]
                                      (swap! new-gensyms conj ng)
                                      ng))) args)
              nexp (conj nargs f)
              {:keys [forward backward]}
              (reduce (fn [{:keys [forward backward] :as tape} [s a]]
                        (if (term? a)
                          tape
                          (tape-expr bound s a tape)))
                      tape
                      (partition 2 (interleave nargs args)))
              bound (into bound (map first forward))]
          {:forward
           (conj forward
                 [sym nexp])
           :backward
           (vec (concat backward
                        ;; reverse chain-rule (backpropagator)
                        (for [a (distinct nargs)
                              :when (bound a) ;; we only do backward on our vars
                              :let [a-back (adjoint-sym a)]]
                          [a-back
                           (list '+ a-back
                                 (list '*
                                       (adjoint-sym sym)
                                       (partial-deriv nexp a)))])
                        ;; initialize new variables with 0
                        (map (fn [a]
                               [(adjoint-sym a) 0])
                             @new-gensyms)))})))


(defn adjoints [args]
  (mapv (fn [a] (symbol (str (name a) "_")))
        args))

(defn init-adjoints [args]
  (->> (interleave (adjoints args) (repeat 0))
     (partition 2)
     (apply concat)))

(defn reverse-diff*
  "Splice the tape "
  [args code]
  (let [{:keys [forward backward]} (tape-expr (into #{} args)
                                             (*my-gensym* "v")
                                              code
                                              {:forward
                                               []
                                               :backward
                                               []})
         ret (first (last forward))]
    (list 'fn args
          (list 'let (vec (apply concat forward))
                [ret
                 (list 'fn [(symbol (str ret "_"))]
                       (list 'let
                             (vec
                              (concat (init-adjoints args)
                                      (apply concat
                                             (reverse
                                              backward))))
                             (adjoints args)))]))))


;; numeric gradient for checks

(defmacro fnr [args code]
  `~(reverse-diff* args code))

(defn addd [exprl i d]
  (if (= i 0)
    (reduce conj [`(~'+ ~d ~(first exprl))] (subvec exprl 1))
    (reduce conj (subvec exprl 0 i)
            (reduce conj [`(~'+ ~d ~(get exprl i))] (subvec exprl (+ i 1)))))) 

(defn finite-difference-expr [expr args i d]
  `(~'/ (~'- (~expr ~@(addd args i d)) (~expr ~@args)) ~d)) 

(defn finite-difference-grad [expr]
  (let [[op args body] expr
        d (*my-gensym*)
        fdes (mapv #(finite-difference-expr expr args % d) (range (count args)))
        argsyms (map (fn [x] `(~'quote ~x)) args)]
    `(~'fn [~@args]
      (~'let [~d 0.001]
       ~fdes
       #_~(zipmap argsyms fdes))))) 





