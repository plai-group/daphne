(ns daphne.desugar-hoppl-test
  (:require [daphne.desugar-hoppl :refer :all]
            [daphne.gensym :refer [*my-gensym*]]
            [clojure.test :refer [deftest testing is]]))


(deftest desugar-hoppl-test-semantics
  (testing "Semantics: Inline all naming constructs as lambdas."
    (is (= 5
           (eval (desugar-hoppl '(let [x 1 y 3] (+ x y 1))))))
    (is (= 4
           (eval (desugar-hoppl-global '[(defn add [a b] (+ a b))
                                         (let [a (add 2 3)]
                                           (- a 1))]))))
    (is (= 3
           (eval (desugar-hoppl-global '[(loop 3 0 (fn [i c] (+ c 1)))]))))))


(deftest desugar-hoppl-test-syntax
  (testing "Syntax: Inline all naming constructs as lambdas."
    (is (=
         (let [gensyms (atom (range))]
           (binding [*my-gensym* (fn [s]
                                   (let [f (first @gensyms)]
                                     (swap! gensyms rest)
                                     (symbol (str s f))))]
             (desugar-hoppl '(let [x 1 y 3] (+ x y 1)))))
         '((fn let0 [x] ((fn let1 [y] (+ x y 1)) 3)) 1)))
    (is (= (let [gensyms (atom (range))]
             (binding [*my-gensym* (fn [s]
                                     (let [f (first @gensyms)]
                                       (swap! gensyms rest)
                                       (symbol (str s f))))]
               (desugar-hoppl-global '[(defn add [a b] (+ a b))
                                       (let [a (add 2 3)]
                                         (- a 1))])))
           '((fn let0 [loop-helper]
              ((fn let1 [add] ((fn let2 [a] (- a 1)) (add 2 3)))
               (fn add [a b] (+ a b))))
            (fn loop-helper [i c v g]
              (if (= i c) v (loop-helper (+ i 1) c (g i v) g))))))
    (is (=
         (let [gensyms (atom (range))]
           (binding [*my-gensym* (fn [s]
                                   (let [f (first @gensyms)]
                                     (swap! gensyms rest)
                                     (symbol (str s f))))]
             (desugar-hoppl-global '[(loop 3 0 (fn [i c] (+ c 1)))])))
         '((fn let0 [loop-helper]
             (let [bound 3
                   initial-value 0
                   g (fn loop1 [i w] ((fn [i c] (+ c 1)) i w))]
               (loop-helper 0 bound initial-value g)))
           (fn loop-helper [i c v g]
             (if (= i c) v (loop-helper (+ i 1) c (g i v) g))))))))
