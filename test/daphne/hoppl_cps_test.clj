(ns daphne.desugar-hoppl-test
  (:require [daphne.hoppl-cps :refer :all]
            [daphne.desugar-hoppl :refer [desugar-hoppl]]
            [daphne.gensym :refer [*my-gensym*]]
            [clojure.test :refer [deftest testing is are]]))


(defn hoppl-cps-helper [exp]
  (let [gensyms (atom (range))]
    (binding [*my-gensym* (fn [s]
                            (let [f (first @gensyms)]
                              (swap! gensyms rest)
                              (symbol (str s f))))]
      (hoppl-cps exp 'k123))))

(deftest hoppl-cps-syntax-test
  (testing "Test HOPPL CPS syntax."
    (is (= (hoppl-cps-helper '(fn [x] x))
           '(k123 (fn [x k0] (k0 x)))))

    (is (= (hoppl-cps-helper '(fn [x] (+ 1 x)))
           '(k123 (fn [x k0] (+ 1 x k0)))))

    (is (= (hoppl-cps-helper '(sqrt (+ (* x x) (* y y))))
           '(* x x
             (fn [cps0]
               (* y y
                  (fn [cps1]
                    (+ cps0 cps1
                       (fn [cps2]
                         (sqrt cps2 k123)))))))))

    (is (= (hoppl-cps-helper '(+ 3 ((fn [x] x) 2)))
           '((fn [x k0] (k0 x)) 2 (fn [cps1] (+ 3 cps1 k123)))))


    (is (= (hoppl-cps-helper '(if (= 2 2) (+ 2 1) 3))
           '(= 2 2 (fn [cps0] (if cps0 (+ 2 1 k123) (k123 3))))))

    (is (= (hoppl-cps-helper '((fn [x] (+ 2 (if (even? x) (+ 1 2) 3))) 5)) 
           '((fn [x k0]
               (even?
                x
                (fn [ifcps1]
                  (if ifcps1
                    (+ 1 2 (fn [cps4] (+ 2 cps4 k0)))
                    ((fn [cps4] (+ 2 cps4 k0)) 3)))))
             5
             k123)))))


(def return-atom (atom ::fail))

(def k-return (fn [x] (reset! return-atom x)))

(def +& (fn [a b k] (k (+ a b))))

(def >& (fn [a b k] (k (> a b))))

(defn eval-cps [exp]
  (in-ns 'daphne.desugar-hoppl-test)
  (eval
   (hoppl-cps (desugar-hoppl exp)
              'k-return)))

(deftest hoppl-cps-semantics-test
  (testing "Testing the CPS transformed expressions."
    (eval-cps '(let [m (fn [x]
                          (if (>& 0 x)
                            (fn [y] (+& 1 y))
                            (fn [z] (+& 2 z))))]
                  ((m 1) 2)))
    (is (= @return-atom 4))))
