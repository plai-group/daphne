(ns daphne.symbolic-simplify-test
  (:require [daphne.symbolic-simplify :refer :all]
            [clojure.test :refer [deftest testing is]]))

(deftest symbolic-simplify-test
  (testing "Testing symbolic simplification."

    (is (= '(+ b 1)
           (symbolic-simplify '(+ b (first [1 a])))))

    (is (= '(2 b)
           (symbolic-simplify '(rest (first (second [a [[1 2 b]]]))))))

    (is (= '(2 b)
           (symbolic-simplify '(rest (first (second [(normal 1 1) [[1 2 b]]]))))))


    (is (= '(normal 1 1)
           (symbolic-simplify '(normal 1 1))))


    (is (= '(2)
           (symbolic-simplify '(rest (rest [1 data 2])))))

    (is (= '{(2) 5}
           (symbolic-simplify '{(rest (rest [1 data 2])) 5})))


    (is (= '(rest (rest data))
           (symbolic-simplify '(rest (rest data)))))

    (is (= '[foo bar baz]
           (symbolic-simplify '(append (append [foo] bar) baz))))))



