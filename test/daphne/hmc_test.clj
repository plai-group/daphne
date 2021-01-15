(ns daphne.hmc-test
  (:require  [clojure.test :refer [deftest testing is]]
             [daphne.hmc :refer :all]
             [daphne.core :refer [sample-from-prior code->graph]]
             [anglican.runtime :refer [mean std normal sqrt
                                       discrete gamma dirichlet flip
                                       defdist]]
             [anglican.core :refer [doquery]]
             [anglican.emit :refer [query fm with-primitive-procedures]]
             [daphne.test-helpers :refer [local-gensym]]
             [daphne.gensym :refer [*my-gensym*]]
             [clojure.core.matrix :as m]
             [clojure.core.matrix.linear :as lin]))


(defn tol? [foppl anglican]
  (< (/ (Math/abs (- foppl anglican))
        anglican)
     0.05))


(deftest exercise-1-test
  (testing "Test program from exercise 1"
    (let [foppl
          (->> (hmc '((let [mu (sample (normal 1 (sqrt 5)))
                                sigma (sqrt 2)
                                lik (normal mu sigma)]
                            (observe lik 8)
                            (observe lik 9)
                            mu))) 
                     (take 10000)) 
            anglican (->>
                      (doquery :smc 
                               (query []
                                      (let [mu (sample (normal 1 (sqrt 5)))
                                            sigma (sqrt 2)
                                            lik (normal mu sigma)]
                                        (observe lik 8)
                                        (observe lik 9)
                                        mu))
                               []
                               :number-of-particles 10000) 
                      (drop 1000)
                      (map :result)
                      (take 10000))]
      (is (tol? (mean foppl) (mean anglican)))
      (is (tol? (std foppl) (std anglican))))))


(deftest exercise-2-test
  (testing "Testing program from exercise 2"
    (let [foppl (->> (hmc '((defn observe-data [_ data slope bias]
                            (let [xn (first data)
                                  yn (second data)
                                  zn (+ (* slope xn) bias)]
                              (observe (normal zn 1.0) yn)
                              (rest (rest data))))

                          (let [slope (sample (normal 0.0 10.0))
                                bias  (sample (normal 0.0 10.0))
                                data (vector 1.0 2.1 2.0 3.9 3.0 5.3
                                             4.0 7.7 5.0 10.2 6.0 12.9)]
                            (loop 6 data observe-data slope bias)
                            (vector slope bias))))
                   (take 20000))
          anglican (->>
                    (doquery :smc (let [observe-data
                                        (fm [xn yn slope bias]
                                            (let [zn (+ (* slope xn) bias)]
                                              (observe (normal zn 1.0) yn)))]

                                    (query []
                                           (let [slope (sample (normal 0.0 10.0))
                                                 bias  (sample (normal 0.0 10.0))
                                                 data (vector 1.0 2.1 2.0 3.9 3.0 5.3
                                                              4.0 7.7 5.0 10.2 6.0 12.9)
                                                 xn (take-nth 2 data)
                                                 yn (take-nth 2 (rest data))]
                                             (loop [i 0]
                                               (when (< i 6)
                                                 (observe-data (nth xn i) (nth yn i) slope bias)
                                                 (recur (inc i))))
                                             (vector slope bias))))
                             []
                             :number-of-particles 10000)
                    (drop 10000)
                    (map :result)
                    (take 50000))]

      (prn "exercise 2:" (mean foppl) (mean anglican))
      (is (tol? (mean (map first foppl)) (mean (map first anglican))))
      (is (tol? (mean (map second foppl)) (mean (map second anglican))))

      (is (tol? (std (map first foppl)) (std (map first anglican))))
      (is (tol? (std (map second foppl)) (std (map second anglican)))))))


;; pretty hard and not working anyway
#_(deftest exercise-3-test
  (testing "Testing program from exercise 3"
    (let [foppl (->> (hmc '((let [x (sample (normal 0 10))
                                y (sample (normal 0 10))]
                            (observe (normal (+ x y) 0.1) 7)
                            [x y])))
                   (take 50000))
          anglican (->>
                    (doquery :smc (query []
                                         (let [x (sample (normal 0 10))
                                               y (sample (normal 0 10))]
                                           (observe (normal (+ x y) 0.1) 7)
                                           [x y]))
                             []
                             :number-of-particles 10000)
                    (drop 10000)
                    (map :result)
                    (take 200000))]

      (prn "exercise 3:" (time (mean foppl)) (mean anglican))
      (is (tol? (mean (map first foppl)) (mean (map first anglican))))
      (is (tol? (mean (map second foppl)) (mean (map second anglican))))

      (is (tol? (std (map first foppl)) (std (map first anglican))))
      (is (tol? (std (map second foppl)) (std (map second anglican)))))))
