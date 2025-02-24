(ns daphne.factor-graph-test
  (:require [daphne.factor-graph :refer :all]
            [daphne.desugar-let :refer [desugar-let]]
            [clojure.walk :refer [prewalk]]
            [clojure.test :refer [deftest testing is]]))

(defn strip-dontcare-numbers [exp]
  (prewalk (fn [x]
             (if (symbol? x)
               (let [nm (name x)]
                 (if-let [[_ prefix] (re-matches #"^(dontcare)\d+$" nm)]
                   (symbol prefix)
                   x))
               x))
           exp))
;;
;; Test data definitions based on the examples in the source file
;;

;; 1) A program that gets source‐code transformed.
(def prog-1
  '(let [s1 (sample (normal 0 1.0))
         s2 (sample (normal 0 1.0))
         delta (- s1 s2)
         epsilon 0.1
         w (> delta epsilon)
         y true]
     (observe (dirac w) y)
     [s1 s2]))

;; Expected output (as shown in the file’s comment) in flattened form.
(def expected-transformed-prog-1
  '(let [s1 (sample (normal 0 1.0))
         s2 (sample (normal 0 1.0))
         delta (sample (dirac (- s1 s2)))
         epsilon 0.1
         w (sample (dirac (> delta epsilon)))
         y true]
     (observe (dirac w) y)
     [s1 s2]))

;; 2) A directed graph (dag) used for mapping to a factor graph.
(def dag-1
  '[{}
    {:V #{sample4 sample2 sample1 observe5 sample3}
     :A {sample2 #{sample3}
         sample1 #{sample3}
         sample3 #{sample4}
         sample4 #{observe5}}
     :P {sample1 (sample* (normal 0 1.0))
         sample2 (sample* (normal 0 1.0))
         sample3 (sample* (dirac (- sample1 sample2)))
         sample4 (sample* (dirac (> sample3 0.1)))
         observe5 (observe* (dirac sample4) true)}
     :Y {observe5 true}}
    (vector sample1 sample2)])

(def expected-factor-graph-from-dag-1
  '[{}
    {:X [sample4 sample2 sample1 sample3]
     :F [f-sample4 f-sample2 f-sample1 f-observe5 f-sample3]
     :A {sample2 f-sample2
         sample1 f-sample1
         sample3 f-sample3
         sample4 f-sample4}
     :psi {f-sample4 (sample* (dirac (> sample3 0.1)))
           f-sample2 (sample* (normal 0 1.0))
           f-sample1 (sample* (normal 0 1.0))
           f-observe5 (observe* (dirac sample4) true)
           f-sample3 (sample* (dirac (- sample1 sample2)))}}
    (vector sample1 sample2)])

;; 3) Factor graph with spurious (substitution) variables -- example 1.
(def factor-graph-1
  '[{}
    {:X [x y z]
     :F [f1 f2 f3]
     :A {x f1
         y f2
         z f3}
     :psi {f1 (normal x 0 1)
           f2 (normal y 0 1)
           f3 (dirac x y)}}
    []])

(def expected-cruft-removed-1
  '[{}
    {:X [y]
     :F [f1 f2]
     :A {f1 #{y}
         f2 #{y}}
     :psi {f1 (normal y 0 1)
           f2 (normal y 0 1)}}
    []])

;; 4) Factor graph with spurious (substitution) factors -- example 2.
(def factor-graph-2
  '[{}
    {:X [x y z]
     :F [f1 f2 f3 f4 f5]
     :A {x f1
         y f2
         z f3}
     :psi {f1 (normal x 0 1)
           f2 (normal y 0 1)
           f3 (normal z 0 1)
           f4 (dirac x y)
           f5 (dirac y z)}}
    []])

(def expected-cruft-removed-2
  '[{}
    {:X [z]
     :F [f1 f2 f3]
     :A {f1 #{z}
         f2 #{z}
         f3 #{z}}
     :psi {f1 (normal z 0 1)
           f2 (normal z 0 1)
           f3 (normal z 0 1)}}
    []])

;;
;; Tests
;;

(deftest test-source-code-transformation
  (testing "source-code transformation wraps deterministic subexpressions correctly"
    (let [actual (first (source-code-transformation [prog-1]))
          expected (desugar-let expected-transformed-prog-1)]
      (println "actual:")
      (println actual)
      (is (= (strip-dontcare-numbers actual)
             (strip-dontcare-numbers expected))))))

(deftest test-graph->factor-graph
  (testing "graph->factor-graph maps the directed graph to a factor graph correctly"
    (let [computed (graph->factor-graph dag-1)]
      (is (= computed expected-factor-graph-from-dag-1)))))

(deftest test-remove-cruft-factor-graph-1
  (testing "remove-cruft removes spurious variables and factors (example 1)"
    (is (= (remove-cruft factor-graph-1)
           expected-cruft-removed-1))))

(deftest test-remove-cruft-factor-graph-2
  (testing "remove-cruft removes spurious factors (example 2)"
    (is (= (remove-cruft factor-graph-2)
           expected-cruft-removed-2))))
