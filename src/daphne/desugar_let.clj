(ns daphne.desugar-let
  (:require [daphne.gensym :refer [*my-gensym*]]))

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

(defmethod desugar-let :let
  [exp]
  (let [[_ bindings & body] exp]
    ((fn expand-bindings [[f & r]]
       (let [[b v] f]
         (if f
           (do
             (list 'let [b (desugar-let v)]
                   (expand-bindings r)))
           ((fn expand-body [[f & r]]
              (if-not (empty? r)
                (list 'let [(*my-gensym* "dontcare") (desugar-let f)]
                      (expand-body r))
                (desugar-let f)))
            body))))
     (partition 2 bindings))))


(defmethod desugar-let :map [exp]
  (into {} (map (fn [[k v]]
                  [(desugar-let k)
                   (desugar-let v)])
                exp)))

(defmethod desugar-let :list [exp]
  (apply list (map #(desugar-let %) exp)))

(defmethod desugar-let :seq [exp]
  (map #(desugar-let %) exp))

(defmethod desugar-let :vector [exp]
  (mapv #(desugar-let %) exp))

(defmethod desugar-let :unrelated [exp]
  exp)




