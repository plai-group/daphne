(ns daphne.desugar-test
  (:require [daphne.desugar :refer :all]
            [clojure.test :refer [deftest testing is]]))


(deftest desugaring-test
  (testing "Testing desugaring functionality"
    (is (= '(let [a1 e1] (let [a2 e2] (let [a3 e3] a3)))
           (desugar
            '(let [a1 e1
                   a2 e2
                   a3 e3]
               a3))))

    (is (= '[a b]
           (let [[_ [_ v1] v2]
                 (desugar '(let []
                             a b))]
             [v1 v2])))


    (is (= '{(let [a 1] (let [b 2] b)) (let [c 2] (let [d 4] d))}
           (desugar '{(let [a 1
                            b 2] b)
                      (let [c 2
                            d 4] d)})))

    (is (= '[(let [v (get [1 2 3] 0)] (+ v 1))
             (let [v (get [1 2 3] 1)] (+ v 1))]
           (desugar
            '(foreach 2 [v [1 2 3]]
                      (+ v 1)))))

    (is (= [1 2]
           (let [[_ [_ v1] [_ [_ v2]]] (desugar '(loop 1 acc + 1 2))]
             [v1 v2])))))
