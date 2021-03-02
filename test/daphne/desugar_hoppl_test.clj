(ns daphne.desugar-hoppl-test
  (:require [daphne.desugar-hoppl :refer :all]
            [clojure.test :refer [deftest testing is]]))


(deftest desugar-hoppl-test-semantics
  (testing "Semantics: Inline all naming constructs as lambdas."
    (is (= 5
           (eval (desugar-hoppl '(let [x 1 y 3] (+ x y 1))))))
    (is (= 4
           (eval (desugar-hoppl-global '[(defn add [a b] (+ a b))
                                         (let [a (add 2 3)]
                                           (- a 1))]))))))


(deftest desugar-hoppl-test-syntax
  (testing "Syntax: Inline all naming constructs as lambdas."
    (is (= (desugar-hoppl '(let [x 1 y 3] (+ x y 1)))
          '((fn [x] ((fn [y] (+ x y 1)) 3)) 1)))
    (is (= (desugar-hoppl-global '[(defn add [a b] (+ a b))
                                   (let [a (add 2 3)]
                                     (- a 1))])
          '((fn [add] ((fn [a] (- a 1)) (add 2 3))) (fn [a b] (+ a b)))))))
