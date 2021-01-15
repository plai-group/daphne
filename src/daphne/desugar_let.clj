(ns daphne.desugar-let
  (:require [daphne.gensym :refer [*my-gensym*]]
            [clojure.core.memoize :as memoize]))

(defn dispatch-desugar-let
  [exp]
  (cond (and (list? exp)
             (= (first exp) 'let))
        :let

        (map? exp)
        :map

        (list? exp)
        :list

        (seq? exp)
        :seq

        (vector? exp)
        :vector

        :else :unrelated))


(defmulti desugar-let dispatch-desugar-let)

(def mem-desugar-let (memoize/lu desugar-let :lu/threshold 10000))

(defmethod desugar-let :let
  [exp]
  (let [[_ bindings & body] exp]
    ((fn expand-bindings [[f & r]]
       (let [[b v] f]
         (if f
           (do
             (list 'let [b (mem-desugar-let v)]
                   (expand-bindings r)))
           ((fn expand-body [[f & r]]
              (if-not (empty? r)
                (list 'let [(*my-gensym* "dontcare") (mem-desugar-let f)]
                      (expand-body r))
                (mem-desugar-let f)))
            body))))
     (partition 2 bindings))))


(defmethod desugar-let :map [exp]
  (into {} (map (fn [[k v]]
                  [(mem-desugar-let k)
                   (mem-desugar-let v)])
                exp)))

(defmethod desugar-let :list [exp]
  (apply list (map #(mem-desugar-let %) exp)))

(defmethod desugar-let :seq [exp]
  (map #(mem-desugar-let %) exp))

(defmethod desugar-let :vector [exp]
  (mapv #(mem-desugar-let %) exp))

(defmethod desugar-let :unrelated [exp]
  exp)




