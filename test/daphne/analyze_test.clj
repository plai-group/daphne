(ns daphne.analyze-test
  (:require [daphne.analyze :refer :all]
            [daphne.desugar :refer [desugar]]
            [clojure.walk :refer [postwalk]]
            [clojure.test :refer [deftest testing is]]))


(defn remove-gensym [exp]
  (postwalk (fn [x]
              (cond (and (symbol? x)
                         (re-matches #"sample\d+" (name x)))
                    :gsample

                    (and (symbol? x)
                         (re-matches #"observe\d+" (name x)))
                    :gobserve

                    :else x))
            exp))

(deftest analyze-test
  (testing "Testing analyze."

    (is (= [{} {:V #{}, :A {}, :P {}, :Y {}} 5]
           (analyze empty-env false 5)))

    (is (= [{} {:V #{}, :A {}, :P {}, :Y {}} 42]
           (analyze empty-env false '(let [foo 42] foo))))

    (is (= [{} {:V #{}, :A {}, :P {}, :Y {}} '(if 1 2 3)]
           (analyze empty-env false '(if 1 2 3))))

    (is (= '[{}
             {:V #{:gsample}, :A {}, :P {:gsample (sample* (normal 0 1))}, :Y {}}
             {:gsample (if 1 2 3)}]
           (remove-gensym
            (analyze empty-env false '{(sample (normal 0 1)) (if 1 2 3)})))) 

    (is (= '[{}
             {:V #{:gsample}, :A {}, :P {:gsample (sample* (normal 0 1))}, :Y {}}
             :gsample]
           (remove-gensym
            (analyze empty-env false '(sample (normal 0 1))))))


    (is (= '[{} {:V #{:gobserve}, :A {}, :P {:gobserve 1}, :Y {:gobserve 1}} :gobserve]
           (remove-gensym
            (analyze empty-env false '(observe (normal 0 1) 1)))))

  ;; example programs from https://www.cs.ubc.ca/~fwood/CS532W-539W/homework/4.html

    (is (= '[{}
             {:V #{:gsample}, :A {}, :P {:gsample (sample* (normal 0 1))}, :Y {}}
             :gsample]
           (remove-gensym
            (analyze empty-env false '(let [x (sample (normal 0 1))]
                                        x)))))

    (is (= '[{} {:V #{}, :A {}, :P {}, :Y {}} [1 2 3]]
           (analyze empty-env false '(vector 1 2 3))))


    (is (= '[{}
             {:V #{:gsample :gobserve},
              :A {:gsample #{:gobserve}},
              :P {:gsample (sample* (normal 0 3)), :gobserve 1},
              :Y {:gobserve 2}}
             [:gsample :gsample]]
           (remove-gensym
            (analyze empty-env false (desugar '(let [data (vector 1 2 (sample (normal 1 1)))
                                                     a (conj [] (sample (normal 0 2)))
                                                     b (conj a (sample (normal 0 3)))]
                                                 (observe (normal (second b) 4) (first (rest data)))
                                                 b))))))
    ))

