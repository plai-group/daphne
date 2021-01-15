(ns daphne.desugar
  (:require [daphne.gensym :refer [*my-gensym*]]
            [clojure.core.memoize :as memoize]))

(defn dispatch-desugar
  [exp]
  (cond (and (list? exp)
             (= (first exp) 'let))
        :let

        (and (list? exp)
             (= (first exp) 'loop))
        :loop

        (and (list? exp)
             (= (first exp) 'foreach))
        :foreach

        (and (list? exp)
             (= (first exp) 'if))
        :if

        (map? exp)
        :map

        (list? exp)
        :list

        (seq? exp)
        :seq

        (vector? exp)
        :vector

        :else :unrelated))


(defmulti desugar dispatch-desugar)

(def mem-desugar (memoize/lu desugar :lu/threshold 10000))

(defmethod desugar :let
  [exp]
  (let [[_ bindings & body] exp]
    ((fn expand-bindings [[f & r]]
       (let [[b v] f]
         (if f
           (do
             (list 'let [b (mem-desugar v)]
                   (expand-bindings r)))
           ((fn expand-body [[f & r]]
              (if-not (empty? r)
                (list 'let [(*my-gensym* "dontcare") (mem-desugar f)]
                      (expand-body r))
                (desugar f)))
            body))))
     (partition 2 bindings))))


(defn expand-loop
  [i c acc f es]
  (if (= i c)
    acc
    (let [new-acc (*my-gensym* "acc")]
      (list 'let [new-acc (apply list (concat (list f i acc) es))]
            (expand-loop (inc i) c new-acc f es)))))


(defmethod desugar :loop
  [exp]
  (let [[_ c acc f & es] exp
        as (map (fn [_] (*my-gensym* "a")) es)]
    (if-not (int? c)
      exp
      (mem-desugar
       (list 'let (vec (interleave as es))
             (expand-loop 0 c acc f as))))))


(defmethod desugar :foreach
  [exp]
  (let [[_ c bindings & body] exp
        bindings (mapv mem-desugar bindings)
        body (map mem-desugar body)]
    (if-not (int? c)
      exp
      (mem-desugar
       (vec (for [i (range c)]
              (apply list
                     (concat
                      (list 'let
                            (vec (apply concat
                                        (for [[v e] (partition 2 bindings)]
                                          [v (list 'get e i)]))))
                      body))))))))


(defmethod desugar :if
  [exp]
  (let [[_ cond then else] exp]
    (list 'if (mem-desugar cond)
          (mem-desugar then)
          (mem-desugar else))))

(defmethod desugar :map [exp]
  (into {} (map (fn [[k v]]
                  [(mem-desugar k)
                   (mem-desugar v)])
                exp)))

(defmethod desugar :list [exp]
  (apply list (map #(mem-desugar %) exp)))

(defmethod desugar :seq [exp]
  (map #(mem-desugar %) exp))

(defmethod desugar :vector [exp]
  (mapv #(mem-desugar %) exp))

(defmethod desugar :unrelated [exp]
  exp)

