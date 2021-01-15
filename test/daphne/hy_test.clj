(ns daphne.hy-test
  "Transpilation code for hy-lang. This code is to address amortized
  inference with the help of the Python optimization stack build on pytorch."
  (:require [daphne.hy :refer :all]
            [clojure.test :refer :all]
            [backtick :refer [template]]))


(deftest faithful-adjacency-list-test
  (testing "Testing mapping to tensor indices by redirecting gensym."
    (let [code '((defn observe-data [_ data slope bias]
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
                   (vector slope bias)))]
      (is (= '([0 1] [0 2] [0 3] [0 4] [0 5] [0 6] [0 7]
               [1 2] [1 3] [1 4] [1 5] [1 6] [1 7])
             (faithful-adjacency-list code))))))


(deftest expressions->hy-test
  (testing "Expression translation to hy (and torch) primitives."
    (is (= '[[sample27746 (.sample (Bernoulli 0.8))]
             [sample27742 (.sample (Bernoulli 0.5))]
             [sample27750 (.sample (Bernoulli 0.2))]
             [observe27914
              (.observe
               (if (= sample27742 True) (Bernoulli 0.1) (Bernoulli 0.5))
               True)]
             [observe28067
              (.observe
               (if
                   (and
                    True
                    (= (if (= sample27742 True) sample27746 sample27750) True))
                 (Bernoulli 0.99)
                 (if
                     (and
                      False
                      (= (if (= sample27742 True) sample27746 sample27750) False))
                   (Bernoulli 0.0)
                   (if
                       (or
                        True
                        (= (if (= sample27742 True) sample27746 sample27750) True))
                     (Bernoulli 0.9)
                     None)))
               True)]]
           (expressions->hy '[[sample27746 (sample* (flip 0.8))]
                              [sample27742 (sample* (flip 0.5))]
                              [sample27750 (sample* (flip 0.2))]
                              [observe27914 (observe* (if (= sample27742 true) (flip 0.1) (flip 0.5)) true)]
                              [observe28067
                               (observe*
                                (if (and true (= (if (= sample27742 true) sample27746 sample27750) true))
                                  (flip 0.99)
                                  (if (and false (= (if (= sample27742 true) sample27746 sample27750) false))
                                    (flip 0.0)
                                    (if (or true (= (if (= sample27742 true) sample27746 sample27750) true))
                                      (flip 0.9)
                                      nil)))
                                true)]])))))
