(ns daphne.address-transformation-test
  (:require [daphne.address-transformation :refer :all]
            [daphne.gensym :refer [*my-gensym*]]
            [clojure.test :refer [deftest testing is]]))


(deftest address-transformation-test
  (testing "Test injection of address data structure."
    (is (=
         (let [gensyms (atom (range))]
           (binding [*my-gensym* (fn [s]
                                   (let [f (first @gensyms)]
                                     (swap! gensyms rest)
                                     (symbol (str s f))))]
             (address-trafo '((fn [x] (sample (normal (+ x 1) 1))) 2) 'alpha5)))
         '((fn [alpha5 x]
             (sample
              (push-address alpha5 addr1)
              (normal
               (push-address alpha5 addr2)
               (+ (push-address alpha5 addr3) x 1)
               1)))
           (push-address alpha5 addr0)
           2)))))


(comment
  (let [push-address (fn [alpha addr]
                       (swap! alpha conj addr)
                       alpha)
        alpha5       (atom [])]
    ((fn [alpha5 x]
       (+
        (push-address alpha5 'addr1)
        2 x))
     (push-address alpha5 'addr0)
     2))

  )
