(ns daphne.core
  "This namespace contains an implementation of a FOPPL compiler following chapter
  3.1 of 'An Introduction to Probabilistic Programming' by van de Meent et al."
  (:require [clojure.set :as set]
            [anglican.runtime :refer :all]
            [daphne.primitives :refer :all]
            [daphne.analyze :refer [analyze empty-env empty-graph]]))


(defn invert-graph [G]
  (reduce (fn [acc m] (merge-with set/union acc m))
          {}
          (for [[p children] G
                c children]
            {c #{p}})))

(defn topo-sort [{:keys [V A]}]
  (let [terminals
        (loop [terminals []
               A A
               V V]
          (let [ts (filter (comp empty? (invert-graph A)) V)
                V (set/difference V (set ts))]
            (if (empty? V)
              (into terminals ts)
              (recur (into terminals ts)
                     (select-keys A V)
                     V))))]
    terminals))

(defn graph->instructions [[_ G E]]
  (conj
   (vec
    (for [t (topo-sort G)]
      [t ((:P G) t)]))
   [:return E]))

(defn eval-instructions [instructions]
  (reduce (fn [acc [s v]]
            (binding [*ns* (find-ns 'daphne.core)]
              (conj acc [s ((eval `(fn [{:syms ~(vec (take-nth 2 (apply concat acc)))}]
                                     ~v))
                            (into {} acc))])))
          []
          instructions))

(defn program->graph [p]
  (reduce (fn [[rho _ _] exp]
            (analyze rho true exp))
          [empty-env empty-graph nil]
          p))

(defn count-vertices [G]
  (count (:V G)))

(defn count-edges [G]
  (count (apply concat (vals (:A G)))))

(defn sample-from-prior [G]
  (-> G
     graph->instructions
     eval-instructions))

(defn observes->samples [instructions]
  (reduce (fn [acc ix]
            (let [[sym v] ix]
              (if (re-find #"observe\d+" (name sym))
                (if (= (first v) 'if)
                  (let [[_ cond [_ dist _] _] v]
                    (binding [*ns* (find-ns 'daphne.core)]
                      (if (eval (list 'let (vec (apply concat acc))
                                      cond))
                        (conj acc [sym (list 'sample* dist)])
                        acc)))
                  (let [[_ dist _] v]
                    (conj acc [sym (list 'sample* dist)])))
                (conj acc ix))))
          []
          instructions))

(defn sample-from-joint [G]
  (-> G
     graph->instructions
     observes->samples
     eval-instructions))

(defn code->graph [code]
  (->> code
       program->graph))

(defn count-graph [code]
  (let [[_ G _] (code->graph code)]
    [(count-vertices G) (count-edges G)]))
