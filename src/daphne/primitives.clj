(ns daphne.primitives
  (:require [clojure.core.matrix :as m]
            [anglican.runtime :refer [tanh]]))

(defn append [& args] (apply conj args))

(defn mat-mul [& args] (apply m/mmul args))
(defn mat-add [& args] (apply m/add args))
(defn mat-transpose [& args] (apply m/transpose args))
(defn mat-tanh [M] (m/emap tanh M))
(defn mat-relu [M] (m/emap (fn [x] (if (> x 0) x 0)) M))
(defn mat-repmat [M r c]
  (let [R (reduce (partial m/join-along 0) (repeat r M))]
    (reduce (partial m/join-along 1) (repeat c R))))

(defn rand-tensor [[dim & dims]]
  (vec
   (for [x (range dim)]
     (if (seq dims)
       (rand-tensor dims)
       (rand)))))

