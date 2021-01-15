(ns daphne.synthesize
  (:require [anglican.runtime :refer :all]
            [anglican.core :refer :all]
            [anglican.emit :refer :all]
            [daphne.hy :refer [foppl->python]])) 

(defn sample-sym? [sym]
  (re-find #"smpl\d+" (name sym)))

(def operations ['+ '- '* #_'/])

(def distributions ['laplace 'normal])

(def non-linear-fns ['tanh #_'relu 'identity])

(def standard-devs [0.1 1.0 2.0 10.0])

(def machine-precision 1e-8)

(defn sample-link-function [vars-in-scope observe?]
  (let [operand (rand-nth operations)
        nlf (rand-nth non-linear-fns)
        args `(~@(when observe? [(rand-nth vars-in-scope)])
               ~@(take (sample* (categorical (if observe?
                                              [[1 0.6] [2 0.4]]
                                              [[1 0.1] [2 0.5] [3 0.4]])))
                      (repeatedly (fn []
                                    (if (> (rand) 0.8)
                                      (sample* (normal 0 3))
                                      (rand-nth vars-in-scope))))))
        op (if (> (count args) 1)
             `(~operand ~@args)
             (first args))]
    (if (= nlf 'identity)
      op
      `(~nlf ~op))))

(defn sample-distribution [mean]
  (let [dist (rand-nth (seq distributions))]
    (if (= dist 'dirac)
      `(normal ~mean machine-precision)
      `(~dist ~mean ~(if (and (seq? mean)
                              (= (first mean) 'tanh))
                       0.1
                       (rand-nth standard-devs))))))

;; between 10 and 20
(defn arithmetic-circuit [min-num-samples num-rand-vars]
  (let [num-samples (sample* (uniform-discrete min-num-samples (dec num-rand-vars)))
        samples (for [n (range num-samples)] (gensym "smpl"))
        num-observes (- num-rand-vars num-samples)
        observes (for [n (range num-observes)] (gensym "obs"))]
    `(~'let [
             ~@(->>
               (into [] (concat samples observes))
               (reduce
                (fn [acc sym]
                  (let [vars-in-scope (filter #(not= '_ %) (map first acc)) ;; filter samples
                        sample? (sample-sym? sym)]
                    (conj acc
                          (if (seq vars-in-scope)
                            (if sample?
                              [sym `(~'sample
                                     ~(sample-distribution (sample-link-function vars-in-scope false)))]
                              ['_ `(~'observe
                                    ~(sample-distribution (sample-link-function vars-in-scope true))
                                   1.23456)])
                            ;; pick some interesting starting point for the prior
                            [sym `(~'sample (~'laplace ~(sample* (normal 0 3)) 3))]))))
                [])
               (apply concat))]
      ;; return not-needed atm.
      ~(vec samples))))


(comment
  (clojure.pprint/pprint
   (arithmetic-circuit 4 8)) 

  (require '[daphne.core :as core]) 

  (->> (arithmetic-circuit 4 8)
     pr-str
     read-string
     list
     #_foppl->python
     core/code->graph
     #_clojure.pprint/pprint
     ) 

)

