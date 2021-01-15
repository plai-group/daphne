(ns daphne.substitute-test
  (:require [clojure.test :refer :all]
            [daphne.substitute :refer :all]))


(deftest substitute-test
  (testing "Testing substitute functionality."
    (is (= (substitute 'x 'x 3) 3))

    (is (= (substitute '(+ x 1) 'x 3)
           '(+ 3 1)))

    (is (= (substitute '{(+ x 1) 5} 'x 3)
           '{(+ 3 1) 5}))

    (is (= (substitute '(let [x 5] (+ x y)) 'y 3)
           '(let [x 5] (+ x 3))))

    (is (= (substitute '(let [x 5] (+ x y)) 'x 3)
           '(let [x 5] (+ x y))))))
