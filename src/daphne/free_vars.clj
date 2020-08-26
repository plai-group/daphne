(ns daphne.free-vars
  (:require [clojure.set :as set]))

(defn dispatch-free-vars
  [exp _]
  (cond (and (seq? exp)
             (= (first exp) 'let))
        :let

        (map? exp)
        :map

        (or (vector? exp) (seq? exp))
        :seq

        (symbol? exp)
        :symbol

        :else :unrelated))

(defmulti free-vars dispatch-free-vars)

(defmethod free-vars :symbol [s bound]
  (if (bound s) #{} #{s}))

(defmethod free-vars :seq [exp bound]
  (set (mapcat #(free-vars % bound) exp)))

(defmethod free-vars :map [exp bound]
  (set/union (set (mapcat #(free-vars % bound) (keys exp)))
             (set (mapcat #(free-vars % bound) (vals exp)))))

(defmethod free-vars :unrelated [exp bound]
  #{})

(defmethod free-vars :let [exp bound]
  (let [[_ [s v] e] exp]
    (set/union (free-vars e (conj bound s))
               (free-vars v bound))))





