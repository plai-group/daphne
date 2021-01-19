(ns daphne.reverse-diff-test
  (:require [daphne.reverse-diff :refer :all]
            [clojure.test :refer [deftest testing is]]
            [daphne.gensym :refer [*my-gensym*]]
            [daphne.test-helpers :refer [local-gensym err?]]))



(deftest partial-deriv-tests
  (testing "Testing partial derivatives."

    (is (= '(- 0 0 1)
           (partial-deriv '(- 1 2 x) 'x)))

    (is (= '(- 1 0)
           (partial-deriv '(- x 1) 'x)))

    (is (= '(* 1 (pow x 0) 1 2)
           (partial-deriv '(* 1 2 x) 'x)))

    (is (= '(- (* 2 (pow x -2)))
           (partial-deriv '(/ 2 x) 'x)))

    (is (= '(/ 1 2)
           (partial-deriv '(/ x 2) 'x)))

    (is (= 0
           (partial-deriv '(/ 1 2) 'x)))

    (is (= 0
           (partial-deriv '(/ x x) 'x)))

    (is (= '(/ 1 y)
           (partial-deriv '(/ x y) 'x)))


    (is (= '(* 2 (pow x (dec 2)))
           (partial-deriv '(pow x 2) 'x)))

    (is (= '(* (log 2) (pow 2 x))
           (partial-deriv '(pow 2 x) 'x)))

    (is (= '(* (+ 1 (log x)) (pow x x))
           (partial-deriv '(pow x x) 'x)))


    (is (= '(* (- (/ 1 (* sigma sigma))) (- x mu))
           (partial-deriv '(normpdf x mu sigma) 'x)))

    (is (= '(* (- (/ 1 (* sigma sigma))) (- mu x))
           (partial-deriv '(normpdf x mu sigma) 'mu)))

    (is (= '(- (* (/ 1 (* sigma sigma sigma)) (pow (- x mu) 2)) (/ 1 sigma)) 
           (partial-deriv '(normpdf x mu sigma) 'sigma)))

    ))


(deftest tape-expr-test
  (testing "Testing tape transformation."
    (binding [*my-gensym* (local-gensym)]
      (is (= {:forward '[[v4 (if (> x x) (+ 5 x) 0)]
                         [then2 (if (> x x) (* 2 v4) 0)]
                         [else3 (if-not (> x x) (+ x 5) 0)]
                         [v1 (if (> x x) then2 else3)]],
              :backward '[[x_ (if (> x x) (+ x_ (* v4_ (+ 1))) x_)]
                          [v4_ (if (> x x) (+ v4_ (* then2_ (* 1 (pow v4 0) 2))) v4_)]
                          [v4_ (if (> x x) 0 v4_)]
                          [x_ (if-not (> x x) (+ x_ (* else3_ (+ 1))) x_)]
                          [then2_ v1_]
                          [else3_ v1_]]}
               (tape-expr #{'x}
                          (*my-gensym* "v")
                          '(if (> x x) (* 2 (+ 5 x)) (+ x 5)) ;; TODO support non-seqs
                          {:forward
                           []
                           :backward
                           []}))))

    (binding [*my-gensym* (local-gensym)]
      (is (= {:forward '[[v2 (- 3 x)]
                         [v3 (x y)]
                         [v1 (* v2 v3)]],
              :backward '[[x_ (+ x_ (* v2_ (- 0 1)))]
                          [v2_ (+ v2_ (* v1_ (* 1 (pow v2 0) v3)))]
                          [v3_ (+ v3_ (* v1_ (* 1 (pow v3 0) v2)))]
                          [v2_ 0]
                          [v3_ 0]]}
             (tape-expr #{'x}
                        (*my-gensym* "v")
                        '(* (- 3 x) ( x y))
                        {:forward
                         []
                         :backward
                         []}))))))

(deftest reverse-diff-test
  (testing "Testing reverse symbolic trafo."
    (binding  [*my-gensym* (local-gensym)]
      (is (= '(fn [x]
                (let [v2 (+ x x) v1 (* v2 y)]
                  [v1 (fn [v1_]
                        (let [x_ 0
                              v2_ 0
                              v2_ (+ v2_ (* v1_ (* 1 (pow v2 0) y)))
                              x_ (+ x_ (* v2_ (+ 1 1)))]
                          [x_]))]))
             (reverse-diff* '[x] '(* (+ x x) y)))))))


(deftest fnr-macro-test
  (testing "Macro expension for reverse-diff macro."
    (binding [*my-gensym* (local-gensym)
              ;; WTF, why do i need this namespace binding? works in REPL without
              *ns* (find-ns 'daphne.reverse-diff)]
      (is (= '(fn [x y]
                (let [v2 (* x x) v1 (+ v2)]
                  [v1 (fn [v1_]
                     (let [x_ 0
                           y_ 0
                           v2_ 0
                           v2_ (+ v2_ (* v1_ (+ 1)))
                           x_ (+ x_ (* v2_ (* 2 (pow x 1))))]
                       [x_ y_]))]))
             (macroexpand-1
              '(fnr [x y]
                    (+ (* x x)))))))))

(defn check-gradient [code values]
  (let [num-grad (apply
                  (eval
                   (finite-difference-grad code))
                  values)
        [res bp] (apply
                  (eval (apply reverse-diff* (rest code)))
                  values)
        rev-grad (bp 1.0)]
    #_(prn #_code "Forward: " res " Grad: " rev-grad)
    (and (not (err? res (apply (eval code) values))) 
         (zero? (count (filter true? (map err? num-grad rev-grad)))))))

(deftest foppl-examples-test
  (testing "Testing examples from the lecture."
    (binding [*ns* (find-ns 'daphne.reverse-diff)]
      (is (check-gradient '(fn [x] (exp (sin x))) [3.2]))

      (is (check-gradient '(fn [x y] (+ (* x x) (sin x))) [5.1 8.7]))

      (is (check-gradient '(fn [x] (if (> x 5) (* x x) (+ x 18))) [3]))

      (is (check-gradient '(fn [x] (if (> x 5) (* x x) (+ x 18))) [6]))

      (is (check-gradient '(fn [x] (log x)) [2.7]))

      (is (check-gradient '(fn [x mu sigma]
                             (+ (- 0 (/ (* (- x mu) (- x mu))
                                        (* 2 (* sigma sigma))))
                                (* (- 0 (/ 1 2)) (log (* 2 (* 3.141592653589793 (* sigma sigma)))))))
                          [3.1 -2.5 8]))

      (is (check-gradient '(fn [x mu sigma] (normpdf x mu sigma)) [3.1 -2.5 8]))


      (is (check-gradient '(fn [x1 x2 x3] (+ (+ (normpdf x1 2 5)
                                               (if (> x2 7)
                                                 (normpdf x2 0 1)
                                                 (normpdf x2 10 1)))
                                            (normpdf x3 -4 10)))
                          [1.2 2.1 4]))

      (is (check-gradient '(fn [sample38647 sample38648]
                             (clojure.core/-
                              (clojure.core/+ (normpdf 5.3 (+ (* sample38647 3.0) sample38648) 1.0)
                                              (normpdf 2.1 (+ (* sample38647 1.0) sample38648) 1.0)
                                              (normpdf 3.9 (+ (* sample38647 2.0) sample38648) 1.0)
                                              (normpdf 12.9 (+ (* sample38647 6.0) sample38648) 1.0)
                                              (normpdf 7.7 (+ (* sample38647 4.0) sample38648) 1.0)
                                              (normpdf 10.2 (+ (* sample38647 5.0) sample38648) 1.0)
                                              (normpdf sample38647 0.0 10.0)
                                              (normpdf sample38648 0.0 10.0))))
                          [-2 3]))

      )))

(deftest foppl-exercise-test
  (testing "Testing examples from the exercise."
    (binding [*ns* (find-ns 'daphne.reverse-diff)]
      (is (check-gradient '(fn [x] (exp (sin x))) [0]))

      (is (check-gradient '(fn [x y] (+ (* x x) (sin x))) [0 10]))

      (is (check-gradient '(fn [x] (if (> x 5) (* x x) (+ x 18))) [5.000001]))

      (is (check-gradient '(fn [x] (log x)) [0.1]))

      (is (check-gradient '(fn [x mu sigma]
                             (+ (- 0 (/ (* (- x mu) (- x mu))
                                        (* 2 (* sigma sigma))))
                                (* (- 0 (/ 1 2)) (log (* 2 (* 3.141592653589793 (* sigma sigma)))))))
                          [10 0 2]))

      (is (check-gradient '(fn [x mu sigma] (normpdf x mu sigma)) [10 0 2]))


      (is (check-gradient '(fn [x1 x2 x3] (+ (+ (normpdf x1 2 5)
                                               (if (> x2 7)
                                                 (normpdf x2 0 1)
                                                 (normpdf x2 10 1)))
                                            (normpdf x3 -4 10)))
                          [2 7.01 5]))

      )))




