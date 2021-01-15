(ns daphne.interpreter-test
  (:refer-clojure :exclude [eval])
  (:require  [clojure.test :refer [deftest testing is]]
             [anglican.stat :refer [empirical-expectation]]
             [anglican.emit :refer [query]]
             [anglican.core :refer [doquery]]
             [anglican.runtime :refer :all]
             [daphne.test-helpers :refer [err?]]
             [daphne.interpreter :refer :all]))


(deftest interpreter-test
  (testing "Testing HOPPL interpreter functions."
    (is (= [1 0.0]
           (eval 1 0.0 empty-env)))
    (is (= [5 0.0]
           (eval 'v 0.0 {'v 5})))
    (is (= [3 0.0]
           (eval '(let [v 3] v) 0.0 empty-env)))
    (is (= [1 0.0]
           (eval '(if true 1 2) 0.0 empty-env)))
    (is (= [2 0.0]
           (eval '(if false 1 2) 0.0 empty-env)))
    (is (= [false 0.0]
           (eval '(= 1 2) 0.0 empty-env)))
    (is (= [true 0.0]
           (eval '(sample (flip 1.0)) 0.0 empty-env)))
    (is (= [[1 2 3] 0.0]
           (eval '[1 2 3] 0.0 empty-env)))
    (is (= [{1 2 3 4} 0.0]
           (eval '{1 2 3 4} 0.0 empty-env)))))



(defn tol? [foppl anglican]
  (if (vector? foppl)
    (not (some false?
               (mapv
                #(< (/ (Math/abs (- %1 %2))
                       %2)
                   0.05)
                foppl anglican)))
    (< (/ (Math/abs (- foppl anglican))
          anglican)
       0.05)))

(deftest likelihood-weighting-base-test
  (testing "Likelihood-weighting test"
    (is (= '([false 0.0] [false 0.0]) 
           (take 2 (likelihood-weighting '((= 1 2))))))

    (is (tol? (->>
               (likelihood-weighting '((let [p 0.01
                                             dist (flip p)
                                             until-success (fn until-success [p n]
                                                             (if (sample dist)
                                                               n
                                                               (until-success p (+ n 1))))]
                                         (until-success p 0))))
               (take 10000)
               (empirical-expectation identity)) 
              (->>
               (doquery :smc
                        (query []
                               (let [p 0.01
                                     dist (flip p)
                                     until-success (fn until-success [p n]
                                                     (if (sample dist)
                                                       n
                                                       (until-success p (+ n 1))))]
                                 (until-success p 0)))
                        []
                        :number-of-particles 10000)
               (take 10000)
               (map (fn [{:keys [result log-weight]}] [result log-weight]))
               (empirical-expectation identity))))))


(deftest marsaglia-test
  (testing "Marsaglia example"
    (is (tol? (->>
               (likelihood-weighting '((defn marsaglia-normal [mean var]
                                         (let [d (uniform-continuous -1.0 1.0)
                                               x (sample d)
                                               y (sample d)
                                               s (+ (* x x ) (* y y ))]
                                           (if (< s 1)
                                             (+ mean (* (sqrt var)
                                                        (* x (sqrt (* -2 (/ (log s) s))))))
                                             (marsaglia-normal mean var))))
                                       (let [mu (marsaglia-normal 1 5)
                                             sigma (sqrt 2)
                                             lik (normal mu sigma)]
                                         (observe lik 8)
                                         (observe lik 9)
                                         mu)))
               (take 10000)
               (empirical-expectation identity)) 
              (->>
               (doquery :smc
                        (query []
                               (let [marsaglia-normal (fn marsaglia-normal [mean var]
                                                        (let [d (uniform-continuous -1.0 1.0)
                                                              x (sample d)
                                                              y (sample d)
                                                              s (+ (* x x ) (* y y ))]
                                                          (if (< s 1)
                                                            (+ mean (* (sqrt var)
                                                                       (* x (sqrt (* -2 (/ (log s) s))))))
                                                            (marsaglia-normal mean var))))
                                     mu (marsaglia-normal 1 5)
                                     sigma (sqrt 2)
                                     lik (normal mu sigma)]
                                 (observe lik 8)
                                 (observe lik 9)
                                 mu))
                        []
                        :number-of-particles 10000)
               (take 10000)
               (map (fn [{:keys [result log-weight]}] [result log-weight]))
               (empirical-expectation identity)))))) 


(deftest reduce-test
  (testing "Reduce example"
    (is (tol? (->>
               (likelihood-weighting '((defn reduce [f acc s]
                                          (if (seq s)
                                            (reduce f (f acc (first s)) (rest s))
                                            acc))
                                       (let [observations [0.9 0.8 0.7 0.0
                                                           -0.025 -5.0 -2.0 -0.1
                                                           0.0 0.13 0.45 6
                                                           0.2 0.3 -1 -1]
                                              init-dist (discrete [1.0 1.0 1.0])
                                              trans-dists {0 (discrete [0.1 0.5 0.4])
                                                           1 (discrete [0.2 0.2 0.6])
                                                           2 (discrete [0.15 0.15 0.7])}
                                              obs-dists {0 (normal -1 1)
                                                         1 (normal 1 1)
                                                         2 (normal 0 1)}]
                                          (reduce
                                           (fn [states obs]
                                             (let [state (sample (get trans-dists (peek states)))]
                                               (observe (get obs-dists state) obs)
                                               (conj states state)))
                                           [(sample init-dist)]
                                           observations))))
               (take 100000)
               (empirical-expectation identity)) 
              (->>
               (doquery :smc
                        (query []
                               (let [observations [0.9 0.8 0.7 0.0 -0.025 -5.0 -2.0 -0.1 0.0 0.13 0.45 6 0.2 0.3 -1 -1]
                                                          init-dist (discrete [1.0 1.0 1.0])
                                                          trans-dists {0 (discrete [0.1 0.5 0.4])
                                                                       1 (discrete [0.2 0.2 0.6])
                                                                       2 (discrete [0.15 0.15 0.7])}
                                                          obs-dists {0 (normal -1 1)
                                                                     1 (normal 1 1)
                                                                     2 (normal 0 1)}]
                                                      (reduce
                                                       (fn [states obs]
                                                         (let [state (sample (get trans-dists
                                                                                  (peek states)))]
                                                           (observe (get obs-dists state) obs)
                                                           (conj states state)))
                                                       [(sample init-dist)]
                                                       observations)))
                        []
                        :number-of-particles 10000)
               (take 500000)
               (map (fn [{:keys [result log-weight]}] [result log-weight]))
               (empirical-expectation identity))))))
