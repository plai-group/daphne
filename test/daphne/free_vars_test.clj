(ns daphne.free-vars-test
  (:require [daphne.free-vars :refer :all]
            [daphne.analyze :refer [*primitive-procedures*]]
            [clojure.test :refer [deftest testing is]]))



(deftest free-vars-test
  (testing "Testing free var detection."
    (is (= #{}
           (free-vars 1 #{})))
    (is (= #{'x}
           (free-vars 'x #{})))

    (is (= '#{x y}
           (free-vars '{[x 1 +] [2 y]}
                      *primitive-procedures*)))

    (is (= #{'x}
           (free-vars '(+ 1 (* x 1)) *primitive-procedures*)))

    (is (= #{'z}
           (free-vars '(let [x 1
                             y 2]
                         [x z]) *primitive-procedures*)))))
