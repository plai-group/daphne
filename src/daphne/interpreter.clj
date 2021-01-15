(ns daphne.interpreter
  (:refer-clojure :exclude [eval])
  (:require [anglican.runtime :refer :all]
            [daphne.desugar :refer [desugar]]))

(def ^:dynamic *primitive-procedures*
  "primitive procedures for Anglican semantics" ;; TODO check implications of this choice
  (let [;; higher-order procedures cannot be primitive
        exclude '#{loop
                   map reduce
                   filter keep keep-indexed remove
                   repeatedly
                   every? not-any? some
                   every-pred some-fn
                   comp juxt partial}
        ;; runtime namespaces
        runtime-namespaces '[clojure.core anglican.runtime #_daphne.primitives]]
    (into {} (keep (fn [[k v]]
                 (when (and (not (exclude k))
                            (fn? (var-get v)))
                   [k v]))
               (mapcat ns-publics runtime-namespaces)))))


(def empty-env *primitive-procedures*)


;; TODO
;; do we need vector seq and map? beyond destructuring?
;; sigma summing correctly?
;; use local environment also for procedure definitions
;; what are constants
;; desugar in HOPPL
;; TODO fix proper primitive lookup in FOPPL
;; repeatedly



(defn dispatch-eval [exp sigma l]
  (cond (or (number? exp)
            (nil? exp)
            (string? exp)
            (boolean? exp)
            (keyword? exp))
        :constant

        (symbol? exp)
        :variable


        (and (list? exp)
             (= (first exp) 'fn))
        :fn

        (and (list? exp)
             (= (first exp) 'defn))
        :defn

        (and (list? exp)
             (= (first exp) 'let))
        :let

        (and (list? exp)
             (= (first exp) 'if))
        :if

        (and (list? exp)
             (= (first exp) 'sample))
        :sample

        (and (list? exp)
             (= (first exp) 'observe))
        :observe

        (list? exp)
        :application

        (seq? exp)
        :seq

        (map? exp)
        :map

        (vector? exp)
        :vector

        :else (throw (ex-info "Not supported." {:exp exp}))))


(defmulti eval dispatch-eval)

;; Big-step rules from Algorithm 6

(defmethod eval :constant
  [c sigma l]
  [c sigma])

(defmethod eval :variable
  [v sigma l]
  [(get l v) sigma])

(defmethod eval :vector
  [v sigma l]
  (let [v' (mapv #(eval % sigma l) v)
        vs (mapv first v')
        sigma' (reduce + (map second v'))]
    [vs sigma']))

(defmethod eval :map
  [v sigma l]
  (reduce (fn [[m sigma] [k v]]
            (let [k' (eval k sigma l)
                  v' (eval v sigma l)]
              [(assoc m (first k') (first v'))
               (+ sigma (second k') (second v'))]))
          [{} 0.0]
          v))

(defmethod eval :let
  [[_ [v1 e1] e0] sigma l]
  (let [res (eval e1 sigma l)
        [c1 sigma] res]
    (eval e0 sigma (assoc l v1 c1))))

(defmethod eval :if
  [[_ e1 e2 e3] sigma l]
  (let [[e1' sigma] (eval e1 sigma l)]
    (if e1'
      (eval e2 sigma l)
      (eval e3 sigma l))))

(defn make-procedure [env args body]
  [:proc env args body])

(defn unpack-procedure [proc]
  (let [[_ env args body] proc]
    [@env args body]))

(defn procedure? [l exp]
  (vector? (get l exp)))

(defmethod eval :fn
  [lambda sigma l]
  ;; TODO allow multiple body expressions
  (if (symbol? (second lambda))
    (let [[_ name args body] lambda
          env (atom l)
          proc (make-procedure env args body)]
      (swap! env assoc name proc)
      [proc sigma])
    (let [[_ args body] lambda]
      [(make-procedure (atom l) args body) sigma])))


(defmethod eval :application
  [[e0 & es] sigma l]
  (let [es' (map #(eval % 0.0 l) es)
        cs (map first es')
        sigma (reduce + sigma (map second es'))]
    (cond (procedure? l e0)
          (let [[env args body] (unpack-procedure (get l e0))
                l' (reduce (fn [l [v c]] (assoc l v c))
                           env
                           (partition 2 (interleave args cs)))]
            (eval body sigma l'))

          (list? e0)
          (let [[env args body] (unpack-procedure (first (eval e0 sigma l)))
                l' (reduce (fn [l [v c]] (assoc l v c))
                           env
                           (partition 2 (interleave args cs)))]
            (eval body sigma l'))

          :else
          [(apply (get l e0) cs) sigma])))


(defmethod eval :sample
  [[_ e] sigma l]
  (let [[d sigma] (eval e sigma l)]
    [(sample* d) sigma]))

(defmethod eval :observe
  [[_ e1 e2] sigma l]
  (let [[d1 sigma1] (eval e1 sigma l)
        [c2 sigma2] (eval e2 sigma l)
        sigma' (+ sigma1 sigma2 (observe* d1 c2))]
    [c2 sigma']))


(defn collect-defns [exps env]
  @(reduce (fn [env [_ name args body]]
             (let [proc (make-procedure env args (desugar body))]
               (swap! env assoc name proc)
               env))
           (atom env)
           exps))

(defn likelihood-weighting [exp]
  (let [env (collect-defns (butlast exp) *primitive-procedures*)]
    (repeatedly #(eval (desugar (last exp)) 0.0 env))))




